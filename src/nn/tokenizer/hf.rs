//! Hugging Face `tokenizer.json` (the format of the `tokenizers` library)
//! for BPE models.
//!
//! The pipeline mirrors `tokenizers`: added tokens are split out first, then
//! each remaining section is normalized, pre-tokenized and byte-pair
//! encoded; a post-processor adds special tokens (e.g. `<s>`), and decoding
//! runs the decoder chain. Supported components:
//!
//! * normalizers: `NFC`, `NFD`, `NFKC`, `NFKD`, `Lowercase`, `Prepend`,
//!   `Replace`, `Strip`, `Sequence`
//! * pre-tokenizers: `ByteLevel`, `Split`, `Metaspace`, `Digits`,
//!   `Whitespace`, `WhitespaceSplit`, `Sequence`
//! * model: `BPE` (with `byte_fallback`, `fuse_unk`, `ignore_merges`)
//! * post-processors: `TemplateProcessing`, `ByteLevel`,
//!   `RobertaProcessing`, `BertProcessing`, `Sequence`
//! * decoders: `ByteLevel`, `Metaspace`, `Replace`, `ByteFallback`, `Fuse`,
//!   `Strip`, `Sequence`
//!
//! Anything else (WordPiece / Unigram models, other components) is reported
//! as an error when loading.

use super::byte_to_char_table;
use fancy_regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::Mutex;
use unicode_normalization::UnicodeNormalization;

/// GPT-2's pre-tokenization regex, used by the `ByteLevel` pre-tokenizer.
const GPT2_PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn regex(pattern: &str) -> io::Result<Regex> {
    Regex::new(pattern).map_err(|e| invalid(format!("bad regex {pattern:?}: {e}")))
}

fn kind(v: &Value) -> &str {
    v["type"].as_str().unwrap_or("")
}

/// A literal string or a regex (`{"String": ..}` / `{"Regex": ..}`).
enum Pattern {
    Str(String),
    Re(Regex),
}

impl Pattern {
    fn parse(v: &Value) -> io::Result<Self> {
        if let Some(s) = v["String"].as_str() {
            Ok(Pattern::Str(s.to_string()))
        } else if let Some(r) = v["Regex"].as_str() {
            Ok(Pattern::Re(regex(r)?))
        } else {
            Err(invalid(format!("bad pattern {v}")))
        }
    }

    /// Byte ranges of the non-overlapping, non-empty matches in `s`.
    fn find_all(&self, s: &str) -> Vec<(usize, usize)> {
        match self {
            Pattern::Str(p) if p.is_empty() => vec![],
            Pattern::Str(p) => s.match_indices(p.as_str()).map(|(i, m)| (i, i + m.len())).collect(),
            Pattern::Re(r) => r.find_iter(s).filter_map(Result::ok).filter(|m| !m.as_str().is_empty()).map(|m| (m.start(), m.end())).collect(),
        }
    }

    fn replace_all(&self, s: &str, with: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let mut last = 0;
        for (a, b) in self.find_all(s) {
            out.push_str(&s[last..a]);
            out.push_str(with);
            last = b;
        }
        out.push_str(&s[last..]);
        out
    }
}

enum Normalizer {
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
    Lowercase,
    Prepend(String),
    Replace(Pattern, String),
    Strip { left: bool, right: bool },
}

impl Normalizer {
    fn parse(v: &Value, out: &mut Vec<Normalizer>) -> io::Result<()> {
        if v.is_null() {
            return Ok(());
        }
        let n = match kind(v) {
            "Sequence" => {
                for n in v["normalizers"].as_array().into_iter().flatten() {
                    Self::parse(n, out)?;
                }
                return Ok(());
            }
            "NFC" => Normalizer::Nfc,
            "NFD" => Normalizer::Nfd,
            "NFKC" => Normalizer::Nfkc,
            "NFKD" => Normalizer::Nfkd,
            "Lowercase" => Normalizer::Lowercase,
            "Prepend" => Normalizer::Prepend(v["prepend"].as_str().unwrap_or("").to_string()),
            "Replace" => Normalizer::Replace(Pattern::parse(&v["pattern"])?, v["content"].as_str().unwrap_or("").to_string()),
            "Strip" => Normalizer::Strip {
                left: v["strip_left"].as_bool().unwrap_or(true),
                right: v["strip_right"].as_bool().unwrap_or(true),
            },
            other => return Err(invalid(format!("unsupported normalizer '{other}'"))),
        };
        out.push(n);
        Ok(())
    }

    fn apply(&self, s: String) -> String {
        match self {
            Normalizer::Nfc => s.nfc().collect(),
            Normalizer::Nfd => s.nfd().collect(),
            Normalizer::Nfkc => s.nfkc().collect(),
            Normalizer::Nfkd => s.nfkd().collect(),
            Normalizer::Lowercase => s.to_lowercase(),
            Normalizer::Prepend(p) if !s.is_empty() => format!("{p}{s}"),
            Normalizer::Prepend(_) => s,
            Normalizer::Replace(p, with) => p.replace_all(&s, with),
            Normalizer::Strip { left, right } => {
                let t = if *left { s.trim_start() } else { &s };
                let t = if *right { t.trim_end() } else { t };
                t.to_string()
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum SplitBehavior {
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
}

/// Split `s` at `matches` (sorted, non-overlapping byte ranges).
fn split_at_matches(s: &str, matches: &[(usize, usize)], behavior: SplitBehavior) -> Vec<String> {
    // alternating (is_match, range) segments covering `s`
    let mut segs: Vec<(bool, usize, usize)> = Vec::new();
    let mut last = 0;
    for &(a, b) in matches {
        if a > last {
            segs.push((false, last, a));
        }
        segs.push((true, a, b));
        last = b;
    }
    if last < s.len() {
        segs.push((false, last, s.len()));
    }
    let mut out: Vec<String> = Vec::new();
    match behavior {
        SplitBehavior::Removed => out.extend(segs.iter().filter(|g| !g.0).map(|g| s[g.1..g.2].to_string())),
        SplitBehavior::Isolated => out.extend(segs.iter().map(|g| s[g.1..g.2].to_string())),
        SplitBehavior::Contiguous => {
            let mut prev_match = false;
            for &(m, a, b) in &segs {
                match out.last_mut() {
                    Some(l) if m && prev_match => l.push_str(&s[a..b]),
                    _ => out.push(s[a..b].to_string()),
                }
                prev_match = m;
            }
        }
        SplitBehavior::MergedWithPrevious => {
            for &(m, a, b) in &segs {
                match out.last_mut() {
                    Some(l) if m => l.push_str(&s[a..b]),
                    _ => out.push(s[a..b].to_string()),
                }
            }
        }
        SplitBehavior::MergedWithNext => {
            let mut pending = String::new();
            for &(m, a, b) in &segs {
                pending.push_str(&s[a..b]);
                if !m {
                    out.push(std::mem::take(&mut pending));
                }
            }
            if !pending.is_empty() {
                out.push(pending);
            }
        }
    }
    out
}

#[derive(Clone, Copy, PartialEq)]
enum PrependScheme {
    Always,
    First,
    Never,
}

enum PreTokenizer {
    ByteLevel { add_prefix_space: bool, use_regex: bool },
    Split { pattern: Pattern, behavior: SplitBehavior, invert: bool },
    Metaspace { replacement: char, prepend: PrependScheme, split: bool },
    Digits { individual: bool },
    Whitespace,
    WhitespaceSplit,
}

fn prepend_scheme(v: &Value) -> PrependScheme {
    match v["prepend_scheme"].as_str() {
        Some("first") => PrependScheme::First,
        Some("never") => PrependScheme::Never,
        Some(_) => PrependScheme::Always,
        // older files: `add_prefix_space`
        None if v["add_prefix_space"].as_bool() == Some(false) => PrependScheme::Never,
        None => PrependScheme::Always,
    }
}

impl PreTokenizer {
    fn parse(v: &Value, out: &mut Vec<PreTokenizer>) -> io::Result<()> {
        if v.is_null() {
            return Ok(());
        }
        let p = match kind(v) {
            "Sequence" => {
                for p in v["pretokenizers"].as_array().into_iter().flatten() {
                    Self::parse(p, out)?;
                }
                return Ok(());
            }
            "ByteLevel" => PreTokenizer::ByteLevel {
                add_prefix_space: v["add_prefix_space"].as_bool().unwrap_or(false),
                use_regex: v["use_regex"].as_bool().unwrap_or(true),
            },
            "Split" => PreTokenizer::Split {
                pattern: Pattern::parse(&v["pattern"])?,
                behavior: match v["behavior"].as_str().unwrap_or("Isolated") {
                    "Removed" => SplitBehavior::Removed,
                    "Isolated" => SplitBehavior::Isolated,
                    "MergedWithPrevious" => SplitBehavior::MergedWithPrevious,
                    "MergedWithNext" => SplitBehavior::MergedWithNext,
                    "Contiguous" => SplitBehavior::Contiguous,
                    other => return Err(invalid(format!("unsupported split behavior '{other}'"))),
                },
                invert: v["invert"].as_bool().unwrap_or(false),
            },
            "Metaspace" => PreTokenizer::Metaspace {
                replacement: v["replacement"].as_str().and_then(|s| s.chars().next()).unwrap_or('▁'),
                prepend: prepend_scheme(v),
                split: v["split"].as_bool().unwrap_or(true),
            },
            "Digits" => PreTokenizer::Digits { individual: v["individual_digits"].as_bool().unwrap_or(false) },
            "Whitespace" => PreTokenizer::Whitespace,
            "WhitespaceSplit" => PreTokenizer::WhitespaceSplit,
            other => return Err(invalid(format!("unsupported pre-tokenizer '{other}'"))),
        };
        out.push(p);
        Ok(())
    }

    /// Split one piece. `first`: the piece starts the input text (for `Metaspace`'s "first" scheme).
    fn apply(&self, piece: String, first: bool, re: &Regexes) -> Vec<String> {
        match self {
            PreTokenizer::ByteLevel { add_prefix_space, use_regex } => {
                let piece = if *add_prefix_space && !piece.starts_with(' ') { format!(" {piece}") } else { piece };
                let parts = if *use_regex {
                    split_at_matches(&piece, &Pattern::find_re(&re.gpt2, &piece), SplitBehavior::Isolated)
                } else {
                    vec![piece]
                };
                parts.iter().map(|p| p.bytes().map(|b| re.byte_to_char[b as usize]).collect()).collect()
            }
            PreTokenizer::Split { pattern, behavior, invert } => {
                let mut m = pattern.find_all(&piece);
                if *invert {
                    m = complement(&m, piece.len());
                }
                split_at_matches(&piece, &m, *behavior)
            }
            PreTokenizer::Metaspace { replacement, prepend, split } => {
                let mut s: String = piece.chars().map(|c| if c == ' ' { *replacement } else { c }).collect();
                let add = match prepend {
                    PrependScheme::Always => true,
                    PrependScheme::First => first,
                    PrependScheme::Never => false,
                };
                if add && !s.starts_with(*replacement) {
                    s.insert(0, *replacement);
                }
                if !*split {
                    return vec![s];
                }
                let m: Vec<(usize, usize)> =
                    s.char_indices().filter(|&(_, c)| c == *replacement).map(|(i, c)| (i, i + c.len_utf8())).collect();
                split_at_matches(&s, &m, SplitBehavior::MergedWithNext)
            }
            PreTokenizer::Digits { individual } => {
                let mut m: Vec<(usize, usize)> = Vec::new();
                for (i, c) in piece.char_indices().filter(|(_, c)| c.is_ascii_digit()) {
                    match m.last_mut() {
                        Some(last) if !*individual && last.1 == i => last.1 = i + c.len_utf8(),
                        _ => m.push((i, i + c.len_utf8())),
                    }
                }
                split_at_matches(&piece, &m, SplitBehavior::Isolated)
            }
            PreTokenizer::Whitespace => {
                Pattern::find_re(&re.whitespace, &piece).into_iter().map(|(a, b)| piece[a..b].to_string()).collect()
            }
            PreTokenizer::WhitespaceSplit => piece.split_whitespace().map(String::from).collect(),
        }
    }
}

impl Pattern {
    fn find_re(r: &Regex, s: &str) -> Vec<(usize, usize)> {
        r.find_iter(s).filter_map(Result::ok).filter(|m| !m.as_str().is_empty()).map(|m| (m.start(), m.end())).collect()
    }
}

fn complement(m: &[(usize, usize)], len: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut last = 0;
    for &(a, b) in m {
        if a > last {
            out.push((last, a));
        }
        last = b;
    }
    if last < len {
        out.push((last, len));
    }
    out
}

/// Compiled helpers shared by the pre-tokenizers.
struct Regexes {
    gpt2: Regex,
    whitespace: Regex,
    byte_to_char: [char; 256],
}

enum Decoder {
    ByteLevel,
    Metaspace { replacement: char, strip_first: bool },
    Replace(Pattern, String),
    ByteFallback,
    Fuse,
    Strip { content: char, start: usize, stop: usize },
}

impl Decoder {
    fn parse(v: &Value, out: &mut Vec<Decoder>) -> io::Result<()> {
        if v.is_null() {
            return Ok(());
        }
        let d = match kind(v) {
            "Sequence" => {
                for d in v["decoders"].as_array().into_iter().flatten() {
                    Self::parse(d, out)?;
                }
                return Ok(());
            }
            "ByteLevel" => Decoder::ByteLevel,
            "Metaspace" => Decoder::Metaspace {
                replacement: v["replacement"].as_str().and_then(|s| s.chars().next()).unwrap_or('▁'),
                strip_first: prepend_scheme(v) != PrependScheme::Never,
            },
            "Replace" => Decoder::Replace(Pattern::parse(&v["pattern"])?, v["content"].as_str().unwrap_or("").to_string()),
            "ByteFallback" => Decoder::ByteFallback,
            "Fuse" => Decoder::Fuse,
            "Strip" => Decoder::Strip {
                content: v["content"].as_str().and_then(|s| s.chars().next()).unwrap_or(' '),
                start: v["start"].as_u64().unwrap_or(0) as usize,
                stop: v["stop"].as_u64().unwrap_or(0) as usize,
            },
            other => return Err(invalid(format!("unsupported decoder '{other}'"))),
        };
        out.push(d);
        Ok(())
    }

    fn apply(&self, tokens: Vec<String>, char_to_byte: &HashMap<char, u8>) -> Vec<String> {
        match self {
            Decoder::ByteLevel => {
                let mut bytes = Vec::new();
                for c in tokens.iter().flat_map(|t| t.chars()) {
                    match char_to_byte.get(&c) {
                        Some(&b) => bytes.push(b),
                        None => bytes.extend_from_slice(c.to_string().as_bytes()),
                    }
                }
                vec![String::from_utf8_lossy(&bytes).into_owned()]
            }
            Decoder::Metaspace { replacement, strip_first } => tokens
                .into_iter()
                .enumerate()
                .map(|(i, t)| {
                    let t: String = t.chars().map(|c| if c == *replacement { ' ' } else { c }).collect();
                    match t.strip_prefix(' ') {
                        Some(rest) if i == 0 && *strip_first => rest.to_string(),
                        _ => t,
                    }
                })
                .collect(),
            Decoder::Replace(p, with) => tokens.iter().map(|t| p.replace_all(t, with)).collect(),
            Decoder::ByteFallback => {
                let mut out = Vec::new();
                let mut bytes: Vec<u8> = Vec::new();
                let flush = |bytes: &mut Vec<u8>, out: &mut Vec<String>| {
                    if !bytes.is_empty() {
                        out.push(String::from_utf8_lossy(bytes).into_owned());
                        bytes.clear();
                    }
                };
                for t in tokens {
                    match byte_token(&t) {
                        Some(b) => bytes.push(b),
                        None => {
                            flush(&mut bytes, &mut out);
                            out.push(t);
                        }
                    }
                }
                flush(&mut bytes, &mut out);
                out
            }
            Decoder::Fuse => vec![tokens.concat()],
            Decoder::Strip { content, start, stop } => tokens
                .into_iter()
                .map(|t| {
                    let chars: Vec<char> = t.chars().collect();
                    let a = chars.iter().take(*start).take_while(|&&c| c == *content).count();
                    let b = chars.len() - chars.iter().rev().take(*stop).take_while(|&&c| c == *content).count();
                    chars[a..b.max(a)].iter().collect()
                })
                .collect(),
        }
    }
}

/// `<0x41>` -> `0x41`.
fn byte_token(t: &str) -> Option<u8> {
    let hex = t.strip_prefix("<0x")?.strip_suffix('>')?;
    (hex.len() == 2).then(|| u8::from_str_radix(hex, 16).ok()).flatten()
}

struct AddedToken {
    content: String,
    id: usize,
    special: bool,
    lstrip: bool,
    rstrip: bool,
}

/// A tokenizer loaded from a Hugging Face `tokenizer.json` (BPE models).
///
/// ```no_run
/// use ruai::tokenizer::HfTokenizer;
/// let tok = HfTokenizer::from_file("data/smollm/tokenizer.json").unwrap();
/// let ids = tok.encode("Hello world", true); // with the model's special tokens (e.g. BOS)
/// assert_eq!(tok.decode(&ids, true), "Hello world");
/// ```
pub struct HfTokenizer {
    vocab: HashMap<String, usize>,
    id_to_token: Vec<String>,
    ranks: HashMap<(String, String), usize>,
    unk: Option<usize>,
    byte_fallback: bool,
    fuse_unk: bool,
    ignore_merges: bool,
    added: Vec<AddedToken>,
    normalizers: Vec<Normalizer>,
    pre_tokenizers: Vec<PreTokenizer>,
    decoders: Vec<Decoder>,
    prefix: Vec<usize>,
    suffix: Vec<usize>,
    re: Regexes,
    char_to_byte: HashMap<char, u8>,
    cache: Mutex<HashMap<String, Vec<usize>>>,
}

/// Special-token ids a post-processor adds before / after a single sequence.
fn parse_post_processor(v: &Value, vocab: &HashMap<String, usize>, prefix: &mut Vec<usize>, suffix: &mut Vec<usize>) -> io::Result<()> {
    if v.is_null() {
        return Ok(());
    }
    let id_of = |t: &Value| -> io::Result<usize> {
        // `[token, id]` pairs (Roberta/Bert) or token names
        if let Some(id) = t[1].as_u64() {
            return Ok(id as usize);
        }
        let name = t.as_str().or(t[0].as_str()).unwrap_or("");
        vocab.get(name).copied().ok_or_else(|| invalid(format!("post-processor token '{name}' not in vocab")))
    };
    match kind(v) {
        "Sequence" => {
            for p in v["processors"].as_array().into_iter().flatten() {
                parse_post_processor(p, vocab, prefix, suffix)?;
            }
        }
        "ByteLevel" => {}
        "RobertaProcessing" | "BertProcessing" => {
            prefix.push(id_of(&v["cls"])?);
            suffix.push(id_of(&v["sep"])?);
        }
        "TemplateProcessing" => {
            let mut seen_seq = false;
            for item in v["single"].as_array().into_iter().flatten() {
                if item.get("Sequence").is_some() {
                    seen_seq = true;
                } else if let Some(name) = item["SpecialToken"]["id"].as_str() {
                    let ids: Vec<usize> = match v["special_tokens"][name]["ids"].as_array() {
                        Some(a) => a.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect(),
                        None => vec![id_of(&Value::from(name))?],
                    };
                    if seen_seq { suffix.extend(ids) } else { prefix.extend(ids) }
                }
            }
        }
        other => return Err(invalid(format!("unsupported post-processor '{other}'"))),
    }
    Ok(())
}

impl HfTokenizer {
    /// Load `dir/tokenizer.json`.
    pub fn from_dir(dir: impl AsRef<Path>) -> io::Result<Self> {
        Self::from_file(dir.as_ref().join("tokenizer.json"))
    }

    /// Load a `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }

    /// Parse the contents of a `tokenizer.json`.
    pub fn from_json(json: &str) -> io::Result<Self> {
        let v: Value = serde_json::from_str(json).map_err(|e| invalid(format!("tokenizer.json: {e}")))?;
        let m = &v["model"];
        let model_type = m["type"].as_str().unwrap_or("BPE"); // old files omit the type
        if model_type != "BPE" {
            return Err(invalid(format!("unsupported model '{model_type}' (only BPE is supported)")));
        }
        for (k, ok) in [("dropout", m["dropout"].is_null()), ("continuing_subword_prefix", m["continuing_subword_prefix"].as_str().unwrap_or("").is_empty()), ("end_of_word_suffix", m["end_of_word_suffix"].as_str().unwrap_or("").is_empty())] {
            if !ok {
                return Err(invalid(format!("unsupported BPE option '{k}'")));
            }
        }
        let mut vocab: HashMap<String, usize> = m["vocab"]
            .as_object()
            .ok_or_else(|| invalid("model.vocab missing"))?
            .iter()
            .map(|(k, id)| (k.clone(), id.as_u64().unwrap_or(0) as usize))
            .collect();
        let mut ranks = HashMap::new();
        for (r, merge) in m["merges"].as_array().into_iter().flatten().enumerate() {
            let pair = match merge {
                Value::String(s) => s.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string())),
                Value::Array(a) => a.first().and_then(Value::as_str).zip(a.get(1).and_then(Value::as_str)).map(|(a, b)| (a.to_string(), b.to_string())),
                _ => None,
            };
            let pair = pair.ok_or_else(|| invalid(format!("bad merge {merge}")))?;
            ranks.entry(pair).or_insert(r);
        }
        let mut added = Vec::new();
        for t in v["added_tokens"].as_array().into_iter().flatten() {
            let content = t["content"].as_str().unwrap_or("").to_string();
            let id = t["id"].as_u64().ok_or_else(|| invalid("added token without id"))? as usize;
            vocab.insert(content.clone(), id);
            added.push(AddedToken {
                content,
                id,
                special: t["special"].as_bool().unwrap_or(false),
                lstrip: t["lstrip"].as_bool().unwrap_or(false),
                rstrip: t["rstrip"].as_bool().unwrap_or(false),
            });
        }
        // longest first, so overlapping added tokens match greedily
        added.sort_by_key(|t| std::cmp::Reverse(t.content.len()));
        let size = vocab.values().max().map_or(0, |m| m + 1);
        let mut id_to_token = vec![String::new(); size];
        for (t, &id) in &vocab {
            id_to_token[id] = t.clone();
        }
        let unk = m["unk_token"].as_str().and_then(|u| vocab.get(u).copied());
        let (mut normalizers, mut pre_tokenizers, mut decoders) = (Vec::new(), Vec::new(), Vec::new());
        Normalizer::parse(&v["normalizer"], &mut normalizers)?;
        PreTokenizer::parse(&v["pre_tokenizer"], &mut pre_tokenizers)?;
        Decoder::parse(&v["decoder"], &mut decoders)?;
        let (mut prefix, mut suffix) = (Vec::new(), Vec::new());
        parse_post_processor(&v["post_processor"], &vocab, &mut prefix, &mut suffix)?;
        let byte_to_char = byte_to_char_table();
        Ok(HfTokenizer {
            vocab,
            id_to_token,
            ranks,
            unk,
            byte_fallback: m["byte_fallback"].as_bool().unwrap_or(false),
            fuse_unk: m["fuse_unk"].as_bool().unwrap_or(false),
            ignore_merges: m["ignore_merges"].as_bool().unwrap_or(false),
            added,
            normalizers,
            pre_tokenizers,
            decoders,
            prefix,
            suffix,
            re: Regexes { gpt2: regex(GPT2_PATTERN)?, whitespace: regex(r"\w+|[^\w\s]+")?, byte_to_char },
            char_to_byte: byte_to_char.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect(),
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Number of token ids (max id + 1, added tokens included).
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Id of a token string (vocabulary or added token).
    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.vocab.get(token).copied()
    }

    /// Token string of an id.
    pub fn id_to_token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(String::as_str).filter(|t| !t.is_empty())
    }

    /// Split `text` into plain sections and added tokens.
    fn split_added<'a>(&self, text: &'a str) -> Vec<Result<&'a str, usize>> {
        let mut out: Vec<Result<&str, usize>> = Vec::new();
        let (mut start, mut i) = (0, 0);
        let mut rstrip_next = false;
        while i < text.len() {
            let hit = self.added.iter().find(|t| !t.content.is_empty() && text[i..].starts_with(&t.content));
            match hit {
                Some(t) => {
                    let mut sec = &text[start..i];
                    if rstrip_next {
                        sec = sec.trim_start();
                    }
                    if t.lstrip {
                        sec = sec.trim_end();
                    }
                    if !sec.is_empty() {
                        out.push(Ok(sec));
                    }
                    out.push(Err(t.id));
                    i += t.content.len();
                    start = i;
                    rstrip_next = t.rstrip;
                }
                None => i += text[i..].chars().next().map_or(1, char::len_utf8),
            }
        }
        let sec = if rstrip_next { text[start..].trim_start() } else { &text[start..] };
        if !sec.is_empty() {
            out.push(Ok(sec));
        }
        out
    }

    /// Byte-pair encode one pre-token.
    fn bpe(&self, word: &str) -> Vec<usize> {
        if let Some(ids) = self.cache.lock().unwrap().get(word) {
            return ids.clone();
        }
        let ids = match self.vocab.get(word) {
            Some(&id) if self.ignore_merges => vec![id],
            _ => {
                let mut parts: Vec<String> = word.chars().map(String::from).collect();
                while parts.len() > 1 {
                    let best = (0..parts.len() - 1)
                        .filter_map(|i| self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())).map(|&r| (r, i)))
                        .min();
                    let Some((_, i)) = best else { break };
                    let (a, b) = (parts[i].clone(), parts[i + 1].clone());
                    let mut merged = Vec::with_capacity(parts.len());
                    let mut j = 0;
                    while j < parts.len() {
                        if j + 1 < parts.len() && parts[j] == a && parts[j + 1] == b {
                            merged.push(format!("{a}{b}"));
                            j += 2;
                        } else {
                            merged.push(parts[j].clone());
                            j += 1;
                        }
                    }
                    parts = merged;
                }
                let mut ids = Vec::with_capacity(parts.len());
                let mut last_unk = false;
                for p in &parts {
                    if let Some(&id) = self.vocab.get(p) {
                        ids.push(id);
                        last_unk = false;
                        continue;
                    }
                    let bytes: Option<Vec<usize>> = self
                        .byte_fallback
                        .then(|| p.bytes().map(|b| self.vocab.get(&format!("<0x{b:02X}>")).copied()).collect())
                        .flatten();
                    match (bytes, self.unk) {
                        (Some(b), _) => {
                            ids.extend(b);
                            last_unk = false;
                        }
                        (None, Some(unk)) => {
                            if !(self.fuse_unk && last_unk) {
                                ids.push(unk);
                            }
                            last_unk = true;
                        }
                        (None, None) => panic!("token {p:?} is not in the vocabulary and there is no unk token"),
                    }
                }
                ids
            }
        };
        self.cache.lock().unwrap().insert(word.to_string(), ids.clone());
        ids
    }

    /// Encode `text`. With `add_special_tokens`, the post-processor's
    /// tokens (e.g. BOS) are added.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<usize> {
        let mut out = Vec::new();
        if add_special_tokens {
            out.extend(&self.prefix);
        }
        for (k, sec) in self.split_added(text).into_iter().enumerate() {
            let sec = match sec {
                Ok(s) => s,
                Err(id) => {
                    out.push(id);
                    continue;
                }
            };
            let normalized = self.normalizers.iter().fold(sec.to_string(), |s, n| n.apply(s));
            let mut pieces = vec![normalized];
            for p in &self.pre_tokenizers {
                pieces = pieces.into_iter().enumerate().flat_map(|(i, s)| p.apply(s, k == 0 && i == 0, &self.re)).collect();
            }
            for piece in pieces.iter().filter(|p| !p.is_empty()) {
                out.extend(self.bpe(piece));
            }
        }
        if add_special_tokens {
            out.extend(&self.suffix);
        }
        out
    }

    /// Decode ids to text; `skip_special_tokens` drops special added tokens.
    pub fn decode(&self, ids: &[usize], skip_special_tokens: bool) -> String {
        let tokens: Vec<String> = ids
            .iter()
            .filter(|&&id| !(skip_special_tokens && self.added.iter().any(|t| t.special && t.id == id)))
            .filter_map(|&id| self.id_to_token(id).map(String::from))
            .collect();
        self.decoders.iter().fold(tokens, |t, d| d.apply(t, &self.char_to_byte)).concat()
    }
}

//! GPT-2 byte-level BPE tokenizer (compatible with `vocab.json` +
//! `merges.txt` from Hugging Face / OpenAI).
//!
//! Text is split with GPT-2's pre-tokenization pattern
//! `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//! (implemented by hand: it needs a lookahead), every piece is mapped
//! byte-by-byte to printable characters, and BPE merges are applied by rank.

use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::sync::Mutex;

/// GPT-2's `<|endoftext|>` token id.
pub const EOT: usize = 50256;

/// GPT-2's reversible byte -> printable-char table (`bytes_to_unicode`).
fn byte_to_char_table() -> [char; 256] {
    let mut printable: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut chars: Vec<u32> = printable.clone();
    let mut extra = 0;
    for b in 0..256u32 {
        if !printable.contains(&b) {
            printable.push(b);
            chars.push(256 + extra);
            extra += 1;
        }
    }
    let mut table = ['\0'; 256];
    for (b, c) in printable.iter().zip(chars) {
        table[*b as usize] = char::from_u32(c).unwrap();
    }
    table
}

/// Split text like GPT-2's regex (see module docs).
pub fn pre_tokenize(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let byte_at = |i: usize| if i < n { chars[i].0 } else { text.len() };
    let is_letter = |c: char| c.is_alphabetic();
    let is_number = |c: char| c.is_numeric();
    let is_other = |c: char| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i].1;
        // contractions
        if c == '\'' {
            let rest = &text[chars[i].0..];
            if let Some(len) = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"].iter().find(|p| rest.starts_with(**p)).map(|p| p.len()) {
                out.push(&rest[..len]);
                i += len; // contractions are ASCII
                continue;
            }
        }
        // optional single space followed by a run of letters / numbers / other symbols
        let j = if c == ' ' && i + 1 < n { i + 1 } else { i };
        let class: Option<&dyn Fn(char) -> bool> = match chars[j].1 {
            c2 if is_letter(c2) => Some(&is_letter),
            c2 if is_number(c2) => Some(&is_number),
            c2 if is_other(c2) => Some(&is_other),
            _ => None,
        };
        if let Some(class) = class {
            let mut e = j;
            while e < n && class(chars[e].1) {
                e += 1;
            }
            out.push(&text[chars[i].0..byte_at(e)]);
            i = e;
            continue;
        }
        // whitespace: \s+(?!\S) leaves the last space for the next word; else \s+
        let mut e = i;
        while e < n && chars[e].1.is_whitespace() {
            e += 1;
        }
        if e < n && e - i > 1 {
            e -= 1;
        }
        out.push(&text[chars[i].0..byte_at(e)]);
        i = e;
    }
    out
}

/// GPT-2 byte-level BPE tokenizer (`vocab.json` + `merges.txt`).
pub struct Gpt2Tokenizer {
    encoder: HashMap<String, usize>,
    decoder: Vec<String>,
    ranks: HashMap<(String, String), usize>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    cache: Mutex<HashMap<String, Vec<usize>>>,
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

impl Gpt2Tokenizer {
    /// Load `dir/vocab.json` and `dir/merges.txt`.
    pub fn from_dir(dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = dir.as_ref();
        Self::from_files(dir.join("vocab.json"), dir.join("merges.txt"))
    }

    /// Load from explicit `vocab.json` and `merges.txt` paths.
    pub fn from_files(vocab: impl AsRef<Path>, merges: impl AsRef<Path>) -> io::Result<Self> {
        Self::from_strings(&std::fs::read_to_string(vocab)?, &std::fs::read_to_string(merges)?)
    }

    /// From the contents of `vocab.json` (token -> id) and `merges.txt`.
    pub fn from_strings(vocab_json: &str, merges_txt: &str) -> io::Result<Self> {
        let encoder: HashMap<String, usize> =
            serde_json::from_str(vocab_json).map_err(|e| invalid(format!("bad vocab.json: {e}")))?;
        let size = encoder.values().max().map_or(0, |m| m + 1);
        let mut decoder = vec![String::new(); size];
        for (tok, &id) in &encoder {
            decoder[id] = tok.clone();
        }
        let mut ranks = HashMap::new();
        for line in merges_txt.lines().filter(|l| !l.starts_with("#version") && !l.trim().is_empty()) {
            let (a, b) = line.split_once(' ').ok_or_else(|| invalid(format!("bad merges line: {line:?}")))?;
            let r = ranks.len();
            ranks.insert((a.to_string(), b.to_string()), r);
        }
        let byte_to_char = byte_to_char_table();
        let char_to_byte = byte_to_char.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        Ok(Gpt2Tokenizer { encoder, decoder, ranks, byte_to_char, char_to_byte, cache: Mutex::new(HashMap::new()) })
    }

    /// Number of token ids (max id + 1).
    pub fn vocab_size(&self) -> usize {
        self.decoder.len()
    }

    /// Byte-pair-merge one pre-token (already mapped to printable chars).
    fn bpe(&self, word: &str) -> Vec<usize> {
        if let Some(ids) = self.cache.lock().unwrap().get(word) {
            return ids.clone();
        }
        let mut parts: Vec<String> = word.chars().map(String::from).collect();
        while parts.len() > 1 {
            // the adjacent pair with the lowest merge rank
            let best = (0..parts.len() - 1)
                .filter_map(|i| self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())).map(|&r| (r, i)))
                .min();
            let Some((_, i)) = best else { break };
            let (a, b) = (parts[i].clone(), parts[i + 1].clone());
            // merge every occurrence of (a, b), left to right
            let mut merged = Vec::with_capacity(parts.len());
            let mut i = 0;
            while i < parts.len() {
                if i + 1 < parts.len() && parts[i] == a && parts[i + 1] == b {
                    merged.push(format!("{a}{b}"));
                    i += 2;
                } else {
                    merged.push(parts[i].clone());
                    i += 1;
                }
            }
            parts = merged;
        }
        let ids: Vec<usize> = parts
            .iter()
            .map(|p| *self.encoder.get(p).unwrap_or_else(|| panic!("token {p:?} missing from vocab")))
            .collect();
        self.cache.lock().unwrap().insert(word.to_string(), ids.clone());
        ids
    }

    /// Encode text to token ids (panics if a BPE token is missing from the vocab).
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut out = Vec::new();
        for piece in pre_tokenize(text) {
            let mapped: String = piece.bytes().map(|b| self.byte_to_char[b as usize]).collect();
            out.extend(self.bpe(&mapped));
        }
        out
    }

    /// Decode ids to text (invalid UTF-8 from a partial character is replaced with U+FFFD).
    pub fn decode(&self, ids: &[usize]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    /// Raw bytes of `ids` (useful for streaming output that may split a UTF-8 character).
    pub fn decode_bytes(&self, ids: &[usize]) -> Vec<u8> {
        ids.iter()
            .map(|&id| self.decoder.get(id).map_or("", String::as_str))
            .flat_map(str::chars)
            .map(|c| self.char_to_byte.get(&c).copied().unwrap_or(b'?'))
            .collect()
    }
}

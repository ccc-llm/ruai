//! Hugging Face `tokenizer.json` support (`HfTokenizer`): byte-level BPE
//! (GPT-2 / Llama 3 / SmolLM style), SentencePiece-style BPE with byte
//! fallback (Llama 2 / TinyLlama style), added tokens, post-processors.
//!
//! With `data/gpt2/tokenizer.json` present (scripts/download_gpt2.sh), the
//! real GPT-2 file is also checked against `Gpt2Tokenizer`.

use ruai::tokenizer::{Gpt2Tokenizer, HfTokenizer};
use serde_json::{json, Value};
use std::path::Path;

/// GPT-2's byte -> char table (reproduced to build test vocabularies).
fn byte_chars() -> Vec<char> {
    let mut bytes: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut chars = bytes.clone();
    let mut extra = 0;
    for b in 0..256 {
        if !bytes.contains(&b) {
            bytes.push(b);
            chars.push(256 + extra);
            extra += 1;
        }
    }
    chars.into_iter().map(|c| char::from_u32(c).unwrap()).collect()
}

/// Byte-level vocabulary: the 256 byte chars plus the merged tokens.
fn byte_level_vocab(merges: &[&str]) -> (Value, Vec<String>) {
    let mut vocab: Vec<String> = byte_chars().iter().map(|c| c.to_string()).collect();
    for m in merges {
        vocab.push(m.replace(' ', ""));
    }
    let map: serde_json::Map<String, Value> = vocab.iter().enumerate().map(|(i, t)| (t.clone(), json!(i))).collect();
    (Value::Object(map), vocab)
}

fn tokenizer(v: Value) -> HfTokenizer {
    HfTokenizer::from_json(&v.to_string()).unwrap()
}

const GPT2_MERGES: [&str; 9] = ["h e", "l l", "he ll", "hell o", "Ġ w", "Ġw o", "r l", "Ġwo rl", "Ġworl d"];

fn gpt2_style() -> (HfTokenizer, Gpt2Tokenizer) {
    let (vocab, list) = byte_level_vocab(&GPT2_MERGES);
    let eot = list.len();
    let hf = tokenizer(json!({
        "added_tokens": [{"id": eot, "content": "<|endoftext|>", "special": true}],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true},
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": false, "vocab": vocab, "merges": GPT2_MERGES},
    }));
    let vocab_json = serde_json::to_string(&list.iter().enumerate().map(|(i, t)| (t.clone(), i)).collect::<std::collections::HashMap<_, _>>()).unwrap();
    let g = Gpt2Tokenizer::from_strings(&vocab_json, &GPT2_MERGES.join("\n")).unwrap();
    (hf, g)
}

const SAMPLES: [&str; 7] = [
    "hello world",
    "hello\n\n  world  ",
    "héllo wörld 你好 🎉",
    "it's 2024!! don't",
    "tabs\tand\r\nnewlines",
    "",
    "  leading and trailing  ",
];

#[test]
fn byte_level_matches_gpt2_tokenizer() {
    let (hf, g) = gpt2_style();
    assert_eq!(hf.encode("hello world", false), vec![256 + 3, 256 + 8]);
    for s in SAMPLES {
        assert_eq!(hf.encode(s, true), g.encode(s), "{s:?}");
        assert_eq!(hf.decode(&hf.encode(s, false), false), s, "{s:?}");
    }
    // added tokens are matched before pre-tokenization
    let eot = hf.token_to_id("<|endoftext|>").unwrap();
    assert_eq!(hf.encode("hello<|endoftext|> world", false), vec![256 + 3, eot, 256 + 8]);
    assert_eq!(hf.decode(&[256 + 3, eot], false), "hello<|endoftext|>");
    assert_eq!(hf.decode(&[256 + 3, eot], true), "hello");
}

/// SentencePiece-style BPE as in Llama 2 / TinyLlama: `▁` for spaces,
/// `<0xXX>` byte fallback, `<s>` added by the post-processor.
fn llama2_style(metaspace: bool) -> HfTokenizer {
    let mut vocab: Vec<String> = vec!["<unk>".into(), "<s>".into(), "</s>".into()];
    vocab.extend((0..=255u8).map(|b| format!("<0x{b:02X}>")));
    let merges = ["▁ h", "e l", "l o", "▁h el", "▁hel lo", "▁ w", "o r", "▁w or", "l d", "▁wor ld"];
    for c in ["▁", "h", "e", "l", "o", "w", "r", "d"] {
        vocab.push(c.into());
    }
    for m in merges {
        vocab.push(m.replace(' ', ""));
    }
    let map: serde_json::Map<String, Value> = vocab.iter().enumerate().map(|(i, t)| (t.clone(), json!(i))).collect();
    let (normalizer, pre_tokenizer) = if metaspace {
        (Value::Null, json!({"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": false}))
    } else {
        (
            json!({"type": "Sequence", "normalizers": [
                {"type": "Prepend", "prepend": "▁"},
                {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}]}),
            Value::Null,
        )
    };
    let decoder = if metaspace {
        json!({"type": "Sequence", "decoders": [
            {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": false},
            {"type": "ByteFallback"}, {"type": "Fuse"}]})
    } else {
        json!({"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"}, {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0}]})
    };
    tokenizer(json!({
        "added_tokens": [
            {"id": 0, "content": "<unk>", "special": true},
            {"id": 1, "content": "<s>", "special": true},
            {"id": 2, "content": "</s>", "special": true}],
        "normalizer": normalizer,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": {"type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [], "special_tokens": {"<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}}},
        "decoder": decoder,
        "model": {"type": "BPE", "dropout": null, "unk_token": "<unk>", "byte_fallback": true, "fuse_unk": true,
                  "vocab": Value::Object(map), "merges": merges},
    }))
}

#[test]
fn sentencepiece_style_bpe_with_byte_fallback() {
    for metaspace in [false, true] {
        let tok = llama2_style(metaspace);
        let id = |t: &str| tok.token_to_id(t).unwrap();
        // "hello world" -> "▁hello", "▁world"
        let ids = tok.encode("hello world", false);
        assert_eq!(ids, vec![id("▁hello"), id("▁world")], "metaspace={metaspace}");
        assert_eq!(tok.encode("hello world", true), vec![1, id("▁hello"), id("▁world")]);
        // characters outside the vocabulary fall back to their UTF-8 bytes
        let ids = tok.encode("hé", false);
        assert_eq!(ids, vec![id("▁h"), id("<0xC3>"), id("<0xA9>")]);
        for s in ["hello world", "hé wörld 你好", "hello  world", "x"] {
            let ids = tok.encode(s, true);
            assert_eq!(tok.decode(&ids, true), s, "metaspace={metaspace} {s:?}");
        }
        // special tokens in the text, and skipping them when decoding
        let ids = tok.encode("hello</s>", false);
        assert_eq!(ids, vec![id("▁hello"), 2]);
        assert_eq!(tok.decode(&[1, id("▁hello"), 2], false), "<s> hello</s>");
        assert_eq!(tok.decode(&[1, id("▁hello"), 2], true), "hello");
    }
}

#[test]
fn split_regex_with_lookahead_like_llama3() {
    // Llama 3's pattern (needs lookahead); digits are grouped by up to three
    let pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    // "3 4" has the best rank: without the split "12345" would become 12|34|5
    let merges = ["3 4", "1 2", "Ġ w", "Ġw o"];
    let (vocab, list) = byte_level_vocab(&merges);
    let tok = tokenizer(json!({
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}]},
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE", "ignore_merges": true, "vocab": vocab, "merges": merges},
    }));
    let id = |t: &str| list.iter().position(|x| x == t).unwrap();
    assert_eq!(tok.encode("12345", false), vec![id("12"), id("3"), id("4"), id("5")]);
    assert_eq!(tok.encode(" wo", false), vec![id("Ġwo")]);
    for s in SAMPLES {
        assert_eq!(tok.decode(&tok.encode(s, false), false), s, "{s:?}");
    }
}

#[test]
fn digits_pre_tokenizer_like_smollm() {
    let merges = ["1 2", "Ġ 1"];
    let (vocab, list) = byte_level_vocab(&merges);
    let tok = tokenizer(json!({
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Digits", "individual_digits": true},
            {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}]},
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
    }));
    let id = |t: &str| list.iter().position(|x| x == t).unwrap();
    // individual digits: "12" is never merged
    assert_eq!(tok.encode("12", false), vec![id("1"), id("2")]);
    assert_eq!(tok.decode(&tok.encode("a 123 b", false), false), "a 123 b");
}

#[test]
fn unsupported_components_are_reported() {
    let err = HfTokenizer::from_json(&json!({"model": {"type": "WordPiece", "vocab": {}}}).to_string()).err().unwrap();
    assert!(err.to_string().contains("WordPiece"), "{err}");
    let err = HfTokenizer::from_json(
        &json!({"normalizer": {"type": "BertNormalizer"}, "model": {"type": "BPE", "vocab": {}, "merges": []}}).to_string(),
    )
    .err()
    .unwrap();
    assert!(err.to_string().contains("BertNormalizer"), "{err}");
}

#[test]
fn real_gpt2_tokenizer_json_matches_vocab_and_merges() {
    let dir = Path::new("data/gpt2");
    if !dir.join("tokenizer.json").exists() || !dir.join("vocab.json").exists() {
        eprintln!("skipping: data/gpt2/tokenizer.json not found (run scripts/download_gpt2.sh)");
        return;
    }
    let hf = HfTokenizer::from_dir(dir).unwrap();
    let g = Gpt2Tokenizer::from_dir(dir).unwrap();
    assert_eq!(hf.vocab_size(), 50257);
    assert_eq!(hf.encode("The dog", false), vec![464, 3290]);
    for s in SAMPLES.iter().chain(&["The quick brown fox jumps over the lazy dog.", "Ünïcödé — “quotes” 1,234.56"]) {
        assert_eq!(hf.encode(s, false), g.encode(s), "{s:?}");
        assert_eq!(hf.decode(&hf.encode(s, false), false), *s);
    }
}

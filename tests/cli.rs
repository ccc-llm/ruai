//! The `ruai` command line tool, run as a subprocess.

use ruai::nn::llama::{Llama, LlamaConfig};
use ruai::{rand, Device};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn ruai(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ruai")).args(args).env_remove("RUSTNN_DEVICE").output().unwrap()
}

fn text(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn temp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ruai_cli_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn help_version_and_errors() {
    let o = ruai(&[]);
    assert!(o.status.success());
    assert!(text(&o).contains("usage: ruai <command>") && text(&o).contains("nn "));
    assert_eq!(text(&ruai(&["--version"])).trim(), format!("ruai {}", env!("CARGO_PKG_VERSION")));
    assert!(text(&ruai(&["help", "nn"])).contains("generate --model DIR"));
    let o = ruai(&["no-such-command"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("ruai-no-such-command"));
    let o = ruai(&["nn", "generate"]);
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).contains("needs --model"));
    let o = ruai(&["nn", "info", "x", "--bogus", "1"]);
    assert!(String::from_utf8_lossy(&o.stderr).contains("unknown option --bogus"));
}

#[cfg(unix)]
#[test]
fn unknown_commands_run_ruai_name_from_path() {
    use std::os::unix::fs::PermissionsExt;
    let dir = temp_dir("ext");
    let script = dir.join("ruai-hello");
    std::fs::write(&script, "#!/bin/sh\necho \"hello from plugin: $*\"\nexit 7\n").unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    let path = format!("{}:{}", dir.display(), std::env::var("PATH").unwrap_or_default());
    let o = Command::new(env!("CARGO_BIN_EXE_ruai")).args(["hello", "a", "b c"]).env("PATH", path).output().unwrap();
    assert_eq!(text(&o), "hello from plugin: a b c\n");
    assert_eq!(o.status.code(), Some(7), "the plugin's exit code is passed through");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn nn_devices_and_info() {
    let o = ruai(&["nn", "devices"]);
    assert!(o.status.success());
    assert!(text(&o).starts_with("cpu    available"));

    rand::manual_seed(0);
    let dir = temp_dir("info");
    let m = Llama::new(LlamaConfig::new(64, 32, 2, 4, 2, 16), Device::Cpu);
    m.save_pretrained(&dir).unwrap();
    let o = ruai(&["nn", "info", dir.to_str().unwrap()]);
    let out = text(&o);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(out.contains("model_type llama"));
    let n: usize = ruai::nn::Module::num_parameters(&m);
    assert!(out.contains(&format!("elements: {n} ")), "{out}");
    assert!(out.contains("--all"));
    assert!(text(&ruai(&["nn", "info", dir.to_str().unwrap(), "--all"])).contains("model.norm.weight"));
    // a single file works too
    let f = dir.join("model.safetensors");
    assert!(text(&ruai(&["nn", "info", f.to_str().unwrap()])).contains("tensors: "));
    std::fs::remove_dir_all(&dir).ok();
}

/// A byte-level tokenizer.json covering all 256 bytes (no merges).
fn byte_tokenizer(dir: &Path) {
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
    let vocab: serde_json::Map<String, serde_json::Value> =
        chars.iter().enumerate().map(|(i, &c)| (char::from_u32(c).unwrap().to_string(), i.into())).collect();
    let tj = serde_json::json!({
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
    });
    std::fs::write(dir.join("tokenizer.json"), tj.to_string()).unwrap();
}

#[test]
fn nn_generate_with_a_small_model() {
    rand::manual_seed(1);
    let dir = temp_dir("gen");
    Llama::new(LlamaConfig::new(256, 32, 2, 4, 2, 64), Device::Cpu).save_pretrained(&dir).unwrap();
    byte_tokenizer(&dir);
    let args = ["nn", "generate", "--model", dir.to_str().unwrap(), "--prompt", "hi", "-n", "5"];
    let o = ruai(&args);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(text(&o).starts_with("hi"));
    assert!(String::from_utf8_lossy(&o.stderr).contains("5 tokens in"));
    // greedy is deterministic; int8 / int4 and sampling run too
    assert_eq!(text(&o), text(&ruai(&args)));
    for extra in [&["--int8"][..], &["--int4"], &["--temperature", "0.8", "--top-k", "5", "--seed", "3"]] {
        let o = ruai(&[&args[..], extra].concat());
        assert!(o.status.success(), "{extra:?}: {}", String::from_utf8_lossy(&o.stderr));
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn nn_generate_real_gpt2() {
    if !Path::new("data/gpt2/model.safetensors").exists() {
        eprintln!("skipping: data/gpt2 not found (run scripts/download_gpt2.sh)");
        return;
    }
    let o = ruai(&["nn", "generate", "--model", "data/gpt2", "--prompt", "The dog", "-n", "18"]);
    assert!(o.status.success());
    // Hugging Face's own greedy continuation (test_lm_generate_gpt2)
    assert_eq!(text(&o), "The dog was found in a field near the intersection of West and West Streets.\n\nThe dog\n");
}

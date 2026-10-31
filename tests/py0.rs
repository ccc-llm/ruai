//! py0 (`ruai py`): every program in tests/py0 is translated, built and run,
//! and must print what CPython prints (or be rejected with the expected
//! message, for tests/py0/err). Needs Python >= 3.12 (`PY2RS_PYTHON`, else
//! python3.14 / 3.13 / 3.12) and cargo; skipped without them.

use std::process::Command;

fn python() -> Option<String> {
    let candidates = std::env::var("PY2RS_PYTHON").ok().into_iter().chain(["python3.14", "python3.13", "python3.12", "python3"].map(String::from));
    candidates.into_iter().find(|p| {
        Command::new(p)
            .args(["-c", "import sys; assert sys.version_info >= (3, 12)"])
            .output()
            .is_ok_and(|o| o.status.success())
    })
}

#[test]
fn corpus_matches_cpython() {
    let Some(py) = python() else {
        eprintln!("skipping: no Python >= 3.12 found");
        return;
    };
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/py0");
    let out = Command::new(env!("CARGO_BIN_EXE_ruai")).args(["py", "test", dir]).env("PY2RS_PYTHON", &py).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "ruai py test failed\n{stdout}\n{stderr}");
}

/// The WebAssembly backend (`ruai py test --wasm`): every program it compiles
/// must print what CPython prints. Needs Node.js 22+ (WasmGC); skipped without it.
#[test]
fn wasm_corpus_matches_cpython() {
    let Some(py) = python() else {
        eprintln!("skipping: no Python >= 3.12 found");
        return;
    };
    let node = Command::new("node").args(["-e", "process.exit(+process.versions.node.split('.')[0] >= 22 ? 0 : 1)"]).output();
    if !node.is_ok_and(|o| o.status.success()) {
        eprintln!("skipping: no Node.js 22+ found");
        return;
    }
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/py0");
    let out = Command::new(env!("CARGO_BIN_EXE_ruai")).args(["py", "test", "--wasm", dir]).env("PY2RS_PYTHON", &py).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "ruai py test --wasm failed\n{stdout}\n{}", String::from_utf8_lossy(&out.stderr));
    // most of the corpus compiles (programs using Rust-backed packages do not)
    let n: usize = stdout.lines().last().and_then(|l| l.strip_prefix("wasm: ")).and_then(|l| l.split(' ').next()).and_then(|n| n.parse().ok()).unwrap_or(0);
    assert!(n >= 200, "only {n} programs compiled to wasm\n{stdout}");
}

#[test]
fn run_emit_check() {
    let dir = std::env::temp_dir().join(format!("ruai_py0_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("hello.py");
    std::fs::write(&f, "def twice(n: int) -> int:\n    return 2 * n\n\nprint(twice(21), [i * i for i in range(4)])\n").unwrap();
    let ruai = |args: &[&str]| Command::new(env!("CARGO_BIN_EXE_ruai")).args(args).output().unwrap();
    let f = f.to_str().unwrap();
    let emit = ruai(&["py", "emit", f]);
    assert!(emit.status.success());
    let rust = String::from_utf8_lossy(&emit.stdout);
    assert!(rust.contains("use ruai::pyrt;") && rust.contains("fn twice(n: i64) -> i64"), "{rust}");
    assert!(ruai(&["py", "check", f]).status.success());
    let run = ruai(&["py", "run", f]);
    assert_eq!(String::from_utf8_lossy(&run.stdout), "42 [0, 1, 4, 9]\n", "{}", String::from_utf8_lossy(&run.stderr));
    // a rejected program: clear error, exit code 1
    std::fs::write(dir.join("bad.py"), "x: int = 'text'\n").unwrap();
    let bad = ruai(&["py", "check", dir.join("bad.py").to_str().unwrap()]);
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("bad.py"));
    std::fs::remove_dir_all(&dir).ok();
}

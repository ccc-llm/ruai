//! `ruai agent` (agge in py0, src/agent/*.py): the embedded translation is up to
//! date, and the command behaves like the same program under CPython, offline
//! (the echo model) and against a fake LLM server (tests/agent/fake_llm.py).
//! The CPython half needs Python >= 3.12 (`PY2RS_PYTHON`, else python3.14 /
//! 3.13 / 3.12) and is skipped without it.

#![cfg(feature = "agent")]

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn translation_is_fresh() {
    let out = Command::new(env!("CARGO_BIN_EXE_ruai"))
        .args(["py", "emit", "src/agent/agent.py", "--module"])
        .current_dir(ROOT)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let committed = std::fs::read_to_string(Path::new(ROOT).join("src/cli/agent_gen.rs")).unwrap();
    assert!(
        String::from_utf8_lossy(&out.stdout) == committed,
        "src/cli/agent_gen.rs is out of date: run\n  ruai py emit src/agent/agent.py --module > src/cli/agent_gen.rs\nand rebuild"
    );
}

fn python() -> Option<String> {
    let candidates = std::env::var("PY2RS_PYTHON").ok().into_iter().chain(["python3.14", "python3.13", "python3.12", "python3"].map(String::from));
    candidates.into_iter().find(|p| {
        Command::new(p)
            .args(["-c", "import sys; assert sys.version_info >= (3, 12)"])
            .output()
            .is_ok_and(|o| o.status.success())
    })
}

/// One step: arguments, and stdin to pipe in (None: no stdin).
type Step = (Vec<&'static str>, Option<&'static str>);

fn offline_steps() -> Vec<Step> {
    let s = |a: &[&'static str]| (a.to_vec(), None);
    let p = |input: &'static str, a: &[&'static str]| (a.to_vec(), Some(input));
    vec![
        s(&["status"]),
        s(&["init", "-m", "echo"]),
        s(&["status"]),
        s(&["provider"]),
        s(&["graph", "list"]),
        s(&["graph", "validate", "solo"]),
        s(&["graph", "validate", "code"]),
        s(&["graph", "validate", "book"]),
        s(&["graph", "validate", "brainstorm"]),
        s(&["run", "solo", "hello world"]),
        s(&["graph", "new", "mine", "--from", "code"]),
        s(&["graph", "add-node", "mine", "docs", "--agent", "writer", "--after", "implement"]),
        s(&["graph", "add-edge", "mine", "test", "implement", "--when", "RESULT: FAIL", "--max", "2"]),
        s(&["graph", "validate", "mine"]),
        s(&["graph", "mermaid", "mine"]),
        s(&["graph", "show", "mine", "--json"]),
        s(&["graph", "rm-node", "mine", "docs"]),
        s(&["graph", "rm", "mine"]),
        s(&["code", "-n", "x", "-M", "review=nvidia:z-ai/glm-5.3-flash"]),
        s(&["agent", "add", "translator", "--base", "writer", "--description", "翻譯", "--model", "echo"]),
        s(&["agent", "list"]),
        s(&["agent", "show", "translator"]),
        s(&["team", "add", "docs", "--members", "outliner", "translator", "editor"]),
        s(&["team", "list"]),
        s(&["model", "list"]),
        s(&["model", "add", "claude", "anthropic", "claude-sonnet-5-5", "--temperature", "0.5"]),
        s(&["model", "test", "echo", "--prompt", "ping"]),
        s(&["model", "default", "nope"]),
        s(&["config", "approval"]),
        s(&["config", "max_steps", "7"]),
        s(&["config", "language", "zh-TW"]),
        s(&["dir", "list"]),
        s(&["log"]),
        s(&["log", "last"]),
        s(&["log", "last", "main"]),
        s(&["ask", "just answer"]),
        s(&["ask", "-v", "<write_file path=\"a.txt\">from ask</write_file> <read_file path=\"a.txt\"/> <list_files/>"]),
        s(&["ask", "<shell>echo from shell</shell>"]),
        s(&["ask", "-y", "<shell>echo from shell</shell> <search pattern=\"from\"/>"]),
        s(&["ask", "-t", "code", "<delegate agent=\"coder\">write it</delegate><delegate agent=\"ghost\">x</delegate>"]),
        s(&["run", "book", "- one\n- two"]),
        s(&["run", "code", "build it"]),
        s(&["run", "brainstorm", "plan"]),
        s(&["bogus"]),
        s(&["graph", "frob", "x"]),
        s(&["ask", "--nope", "x"]),
        s(&["ask"]),
        p("/agents\n/model echo\n@coder ping\n/team code\n/yes\nhi\n/exit\nnever-sent\n", &["chat"]),
        p("<write_file path=\"w.txt\">hi</write_file>\n", &["chat"]),
        p("hello\n/save out.txt\n/compact\n/graphs\n/graph code\n/dirs\n/clear\n/help\n/nope\n@ghost x\n/team zz\n", &["chat"]),
        p("/run solo do it\n", &["chat", "-a", "writer"]),
        s(&["session", "list"]),
        s(&["help"]),
    ]
}

fn network_steps(port: u16) -> Vec<Step> {
    let base: &'static str = Box::leak(format!("http://127.0.0.1:{port}/v1").into_boxed_str());
    let s = |a: &[&'static str]| (a.to_vec(), None);
    vec![
        s(&["init", "-m", "echo"]),
        s(&["model", "add", "fo", "openai", "fake-model", "--base-url", base, "--api-key-env", "FAKE_KEY"]),
        s(&["model", "add", "fa", "anthropic", "fake-model", "--base-url", base, "--api-key-env", "FAKE_KEY"]),
        s(&["model", "add", "nokey", "openai", "x", "--base-url", base, "--api-key-env", "NO_SUCH_KEY_RUAI"]),
        s(&["model", "test", "fo", "--prompt", "hello there"]),
        s(&["model", "test", "fa", "--prompt", "hello anthropic"]),
        s(&["ask", "-m", "fo", "non streaming"]),
        s(&["ask", "-m", "fo", "--stream", "streaming"]),
        s(&["ask", "-m", "fa", "anthropic non streaming"]),
        s(&["ask", "-m", "fa", "--stream", "anthropic streaming"]),
        s(&["ask", "-m", "fo", "--stream", "please CALLTOOL now"]),
        s(&["ask", "-m", "fo", "FAIL500 then ok"]),
        s(&["ask", "-m", "nokey", "x"]),
        s(&["run", "solo", "-M", "main=fa", "graph via anthropic"]),
    ]
}

/// Runs the steps in a fresh directory; the transcript, with paths and times masked.
fn transcript(cmd: &[String], steps: &[Step], dir: &Path, env: &[(&str, String)]) -> String {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut log = String::new();
    for (args, input) in steps {
        // run and session ids are timestamps: keep them one second apart
        if matches!(args.first(), Some(&"run" | &"code")) || input.is_some() {
            std::thread::sleep(std::time::Duration::from_millis(1100));
        }
        let mut c = Command::new(&cmd[0]);
        c.args(&cmd[1..]).args(args).current_dir(dir).env("NO_COLOR", "1").env("AGGE_HOME", dir.join("home"));
        c.env_remove("NVIDIA_API_KEY").env("PYTHONPATH", Path::new(ROOT).join("py0/python"));
        for (k, v) in env {
            c.env(k, v);
        }
        c.stdin(if input.is_some() { Stdio::piped() } else { Stdio::null() }).stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child = c.spawn().unwrap();
        if let Some(i) = input {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(i.as_bytes()).unwrap();
        }
        let out = child.wait_with_output().unwrap();
        log += &format!(
            "$ {:?}\n{}{}[exit {}]\n",
            args,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
            out.status.code().unwrap_or(-1)
        );
    }
    for f in ["a.txt", "w.txt", "out.txt", "tool.txt"] {
        if let Ok(t) = std::fs::read_to_string(dir.join(f)) {
            log += &format!("== {f}\n{t}\n");
        }
    }
    let dir = dir.to_string_lossy();
    let masked = log.replace(dir.as_ref(), "DIR");
    let time = regex::Regex::new(r"\d{8}-\d{6}|\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?|\d+\.\ds · ").unwrap();
    time.replace_all(&masked, "TIME").into_owned()
}

fn compare(name: &str, steps: &[Step], env: &[(&str, String)], py: &str) {
    let tmp = std::env::temp_dir().join(format!("ruai_agent_{}_{}", name, std::process::id()));
    let ruai = vec![env!("CARGO_BIN_EXE_ruai").to_string(), "agent".to_string()];
    let cpython = vec![py.to_string(), Path::new(ROOT).join("src/agent/agent.py").to_string_lossy().into_owned()];
    let a = transcript(&ruai, steps, &tmp.join("ruai"), env);
    let b = transcript(&cpython, steps, &tmp.join("cpython"), env);
    if a != b {
        let diff: Vec<String> =
            a.lines().zip(b.lines()).enumerate().filter(|(_, (x, y))| x != y).take(10).map(|(i, (x, y))| format!("line {}:\n  ruai:    {x}\n  cpython: {y}", i + 1)).collect();
        panic!("ruai agent and CPython differ ({} vs {} lines)\n{}", a.lines().count(), b.lines().count(), diff.join("\n"));
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn offline_matches_cpython() {
    let Some(py) = python() else {
        eprintln!("skipping: no Python >= 3.12 found");
        return;
    };
    compare("offline", &offline_steps(), &[], &py);
}

/// The fake LLM server, killed when dropped.
struct Server(std::process::Child, u16);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn fake_llm(py: &str) -> Server {
    let script: PathBuf = Path::new(ROOT).join("tests/agent/fake_llm.py");
    let mut child = Command::new(py).arg("-I").arg(script).stdout(Stdio::piped()).spawn().unwrap();
    let mut line = String::new();
    BufReader::new(child.stdout.as_mut().unwrap()).read_line(&mut line).unwrap();
    let port = line.trim().parse().unwrap();
    Server(child, port)
}

#[test]
fn network_matches_cpython() {
    let Some(py) = python() else {
        eprintln!("skipping: no Python >= 3.12 found");
        return;
    };
    let server = fake_llm(&py);
    let steps = network_steps(server.1);
    compare("network", &steps, &[("FAKE_KEY", "k".to_string())], &py);
    drop(server);
}

//! `subprocess.run`: a command's exit code and output, like CPython's
//! `subprocess.run(..., capture_output=True, text=True)`.

use crate::pyrt::err::{os_err, PyErr};
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const TIMEOUT_EXPIRED: &[&str] = &["TimeoutExpired", "SubprocessError", "Exception", "BaseException"];

/// The result of `subprocess.run`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompletedProcess {
    pub args: String,
    pub returncode: i64,
    pub stdout: String,
    pub stderr: String,
}

impl CompletedProcess {
    /// `p.returncode`
    pub fn returncode(&self) -> i64 {
        self.returncode
    }
    /// `p.stdout` (empty without `capture_output`)
    pub fn stdout(&self) -> String {
        self.stdout.clone()
    }
    /// `p.stderr` (empty without `capture_output`)
    pub fn stderr(&self) -> String {
        self.stderr.clone()
    }
    /// `p.args`
    pub fn args(&self) -> String {
        self.args.clone()
    }
}

fn shell_words(cmd: &str) -> Result<Vec<String>, PyErr> {
    super::shlex::split(cmd).map_err(|e| PyErr::new(crate::pyrt::err::VALUE_ERROR, e))
}

/// `subprocess.run(args, shell=False, cwd=None, capture_output=False, text=False, timeout=None, input=None)`.
/// With `shell=False` the string is split like `shlex.split`.
pub fn run(
    args: &str,
    shell: bool,
    cwd: &Option<String>,
    capture_output: bool,
    _text: bool,
    timeout: Option<f64>,
    input: &Option<String>,
) -> Result<CompletedProcess, PyErr> {
    let mut cmd = if shell {
        let mut c = Command::new("/bin/sh");
        c.arg("-c").arg(args);
        c
    } else {
        let w = shell_words(args)?;
        let Some(prog) = w.first() else {
            return Err(PyErr::new(crate::pyrt::err::INDEX_ERROR, "list index out of range".to_string()));
        };
        let mut c = Command::new(prog);
        c.args(&w[1..]);
        c
    };
    if let Some(d) = cwd {
        cmd.current_dir(d);
    }
    cmd.stdin(if input.is_some() { Stdio::piped() } else { Stdio::inherit() });
    if capture_output {
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    }
    crate::pyrt::out::flush();
    let prog = if shell { "/bin/sh".to_string() } else { shell_words(args)?.remove(0) };
    let mut child = cmd.spawn().map_err(|e| os_err(&e, &prog))?;
    if let Some(s) = input.clone() {
        let mut stdin = child.stdin.take().unwrap();
        std::thread::spawn(move || {
            use std::io::Write;
            let _ = stdin.write_all(s.as_bytes());
        });
    }
    // read the pipes in threads so a full pipe never blocks the child
    let reader = |p: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(mut p) = p {
                let _ = p.read_to_end(&mut buf);
            }
            String::from_utf8_lossy(&buf).into_owned()
        })
    };
    let out = reader(child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let err = reader(child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let start = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait().map_err(|e| os_err(&e, &prog))? {
            break s;
        }
        if let Some(t) = timeout {
            if start.elapsed().as_secs_f64() > t {
                let _ = child.kill();
                let _ = child.wait();
                let secs = if t.fract() == 0.0 { format!("{}", t as i64) } else { format!("{}", t) };
                return Err(PyErr::new(TIMEOUT_EXPIRED, format!("Command '{}' timed out after {} seconds", args, secs)));
            }
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    #[cfg(unix)]
    let code = {
        use std::os::unix::process::ExitStatusExt;
        status.code().map(|c| c as i64).unwrap_or_else(|| -(status.signal().unwrap_or(0) as i64))
    };
    #[cfg(not(unix))]
    let code = status.code().unwrap_or(-1) as i64;
    Ok(CompletedProcess { args: args.to_string(), returncode: code, stdout: out.join().unwrap_or_default(), stderr: err.join().unwrap_or_default() })
}

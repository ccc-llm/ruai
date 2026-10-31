//! More of `sys` than py0 builds in: declared by `src/py/lib/sys.pyi`, where
//! `sys.stdin.isatty` is `stdin_isatty`.

use std::io::{IsTerminal, Write};

/// `sys.stdin.isatty()`.
pub fn stdin_isatty() -> bool {
    std::io::stdin().is_terminal()
}

/// `sys.stdout.isatty()`.
pub fn stdout_isatty() -> bool {
    std::io::stdout().is_terminal()
}

/// `sys.stderr.write(s)`: what `print` wrote so far comes first.
pub fn stderr_write(s: &str) -> i64 {
    crate::pyrt::out::flush();
    let mut e = std::io::stderr();
    let _ = e.write_all(s.as_bytes());
    let _ = e.flush();
    s.chars().count() as i64
}

/// `sys.stdout.flush()`.
pub fn stdout_flush() {
    crate::pyrt::out::flush();
}

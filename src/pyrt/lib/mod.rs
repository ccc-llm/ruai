//! Python's standard library for py0, in Rust: each module mirrors the CPython
//! module of the same name (the same results, checked against CPython by the
//! programs in `tests/py0/py/std_*.py`), and is declared to py0 by a stub
//! `src/py/lib/NAME.pyi`, so a py0 program simply writes `import shlex`.

pub mod fnmatch;
pub mod html;
mod html_tables;
pub mod os;
#[cfg(feature = "pyrt-http")]
pub mod ruai_http;
pub mod ruai_json;
pub mod shlex;
pub mod subprocess;
pub mod sys;
pub mod time;

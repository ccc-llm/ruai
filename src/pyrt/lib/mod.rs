//! Python's standard library for py0, in Rust: each module mirrors the CPython
//! module of the same name (the same results, checked against CPython by the
//! programs in `tests/py0/py/std_*.py`), and is declared to py0 by a stub
//! `src/py/lib/NAME.pyi`, so a py0 program simply writes `import shlex`.

pub mod fnmatch;
#[cfg(feature = "pyrt-gymnasium")]
pub mod gymnasium;
pub mod html;
mod html_tables;
#[cfg(feature = "pyrt-numpy")]
pub mod numpy;
pub mod os;
#[cfg(feature = "pyrt-http")]
pub mod ruai_http;
pub mod ruai_json;
pub mod shlex;
#[cfg(feature = "pyrt-sklearn")]
pub mod sklearn;
pub mod subprocess;
pub mod sys;
pub mod time;
#[cfg(feature = "pyrt-torch")]
pub mod torch;

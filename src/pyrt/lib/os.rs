//! More of `os` and `os.path` than py0 builds in (`os.path.exists`,
//! `os.path.join`, `os.remove`): declared by `src/py/lib/os.pyi`, where
//! `os.environ.get` is `environ_get` and `os.path.isdir` is `path_isdir`.

use crate::pyrt::err::{os_err, PyErr};

/// `os.getcwd()`.
pub fn getcwd() -> String {
    std::env::current_dir().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default()
}

/// `os.environ.get(key, default=None)`.
pub fn environ_get(key: &str, default: &Option<String>) -> Option<String> {
    std::env::var(key).ok().or_else(|| default.clone())
}

/// `os.listdir(path)`: the names in a directory (in the order the OS lists them, like CPython).
pub fn listdir(path: &str) -> Result<Vec<String>, PyErr> {
    let rd = std::fs::read_dir(path).map_err(|e| os_err(&e, path))?;
    Ok(rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
}

/// `os.makedirs(path, exist_ok=False)`.
pub fn makedirs(path: &str, exist_ok: bool) -> Result<(), PyErr> {
    let p = std::path::Path::new(path);
    if p.exists() {
        if exist_ok && p.is_dir() {
            return Ok(());
        }
        return Err(PyErr::new(crate::pyrt::err::FILE_EXISTS_ERROR, format!("[Errno 17] File exists: '{}'", path)));
    }
    std::fs::create_dir_all(p).map_err(|e| os_err(&e, path))
}

/// `os.rmdir(path)`.
pub fn rmdir(path: &str) -> Result<(), PyErr> {
    std::fs::remove_dir(path).map_err(|e| os_err(&e, path))
}

/// `os.rename(src, dst)`.
pub fn rename(src: &str, dst: &str) -> Result<(), PyErr> {
    std::fs::rename(src, dst).map_err(|e| os_err(&e, src))
}

/// `os.path.isdir(p)`.
pub fn path_isdir(p: &str) -> bool {
    std::path::Path::new(p).is_dir()
}

/// `os.path.isfile(p)`.
pub fn path_isfile(p: &str) -> bool {
    std::path::Path::new(p).is_file()
}

/// `os.path.isabs(p)`.
pub fn path_isabs(p: &str) -> bool {
    p.starts_with('/')
}

/// `os.path.getsize(p)`.
pub fn path_getsize(p: &str) -> Result<i64, PyErr> {
    std::fs::metadata(p).map(|m| m.len() as i64).map_err(|e| os_err(&e, p))
}

/// `os.path.normpath(p)` (POSIX): `.` and `..` resolved textually.
pub fn path_normpath(p: &str) -> String {
    if p.is_empty() {
        return ".".into();
    }
    // POSIX allows exactly two leading slashes to mean something else
    let lead = if p.starts_with("//") && !p.starts_with("///") {
        2
    } else if p.starts_with('/') {
        1
    } else {
        0
    };
    let mut parts: Vec<&str> = vec![];
    for c in p.split('/') {
        match c {
            "" | "." => {}
            ".." if lead == 0 && (parts.is_empty() || parts.last() == Some(&"..")) => parts.push(c),
            ".." => {
                parts.pop();
            }
            c => parts.push(c),
        }
    }
    let s = "/".repeat(lead) + &parts.join("/");
    if s.is_empty() {
        ".".into()
    } else {
        s
    }
}

/// `os.path.abspath(p)`.
pub fn path_abspath(p: &str) -> String {
    if path_isabs(p) {
        path_normpath(p)
    } else {
        path_normpath(&format!("{}/{}", getcwd(), p))
    }
}

/// `os.path.realpath(p)`: symbolic links resolved (as far as the path exists).
pub fn path_realpath(p: &str) -> String {
    let abs = path_abspath(p);
    // resolve the longest existing prefix, keep the rest
    let mut head = std::path::PathBuf::from(&abs);
    let mut tail: Vec<String> = vec![];
    loop {
        if let Ok(c) = std::fs::canonicalize(&head) {
            let mut out = c.to_string_lossy().into_owned();
            for t in tail.iter().rev() {
                if !out.ends_with('/') {
                    out.push('/');
                }
                out.push_str(t);
            }
            return path_normpath(&out);
        }
        match (head.file_name().map(|f| f.to_string_lossy().into_owned()), head.parent()) {
            (Some(f), Some(parent)) => {
                tail.push(f);
                head = parent.to_path_buf();
            }
            _ => return abs,
        }
    }
}

/// `os.path.expanduser(p)`.
pub fn path_expanduser(p: &str) -> String {
    if p != "~" && !p.starts_with("~/") {
        return p.to_string();
    }
    match std::env::var("HOME") {
        Ok(h) => {
            let h = h.trim_end_matches('/');
            let h = if h.is_empty() { "/" } else { h };
            if p == "~" {
                h.to_string()
            } else if h == "/" {
                p[1..].to_string()
            } else {
                format!("{}{}", h, &p[1..])
            }
        }
        Err(_) => p.to_string(),
    }
}

/// `os.path.dirname(p)`.
pub fn path_dirname(p: &str) -> String {
    let i = p.rfind('/').map(|i| i + 1).unwrap_or(0);
    let head = &p[..i];
    if !head.is_empty() && head.chars().any(|c| c != '/') {
        head.trim_end_matches('/').to_string()
    } else {
        head.to_string()
    }
}

/// `os.path.basename(p)`.
pub fn path_basename(p: &str) -> String {
    p.rsplit('/').next().unwrap_or("").to_string()
}

/// `os.path.splitext(p)`.
pub fn path_splitext(p: &str) -> (String, String) {
    let base = p.rfind('/').map(|i| i + 1).unwrap_or(0);
    let name = &p[base..];
    let lead = name.len() - name.trim_start_matches('.').len();
    match name[lead..].rfind('.') {
        Some(j) => {
            let k = base + lead + j;
            (p[..k].to_string(), p[k..].to_string())
        }
        None => (p.to_string(), String::new()),
    }
}

/// `os.path.relpath(path, start)`.
pub fn path_relpath(path: &str, start: &str) -> String {
    let a = path_abspath(path);
    let b = path_abspath(start);
    let a: Vec<&str> = a.split('/').filter(|s| !s.is_empty()).collect();
    let b: Vec<&str> = b.split('/').filter(|s| !s.is_empty()).collect();
    let common = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let mut parts: Vec<&str> = vec![".."; b.len() - common];
    parts.extend(&a[common..]);
    if parts.is_empty() {
        ".".into()
    } else {
        parts.join("/")
    }
}

/// `os.getpid()`.
pub fn getpid() -> i64 {
    std::process::id() as i64
}

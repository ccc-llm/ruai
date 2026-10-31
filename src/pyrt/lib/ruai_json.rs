//! `ruai.json`: a JSON value for py0 programs (`from ruai.json import Json`).
//!
//! py0 values have static types, so `json.loads` (whose result could be
//! anything) is outside the subset; `Json.loads` returns a `Json` instead,
//! read with `get` / `at` / `str_or` / ... and written back by `dumps` exactly
//! like CPython's `json.dumps`. Its CPython stand-in is `py0/python/ruai/json.py`.
//! Values have value semantics: `get` returns a copy.

use crate::pyrt::err::{PyErr, VALUE_ERROR};
use indexmap::IndexMap;

/// A JSON value.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Json {
    #[default]
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Json>),
    Dict(IndexMap<String, Json>),
}

static NULL: Json = Json::Null;

impl Json {
    /// `Json.loads(s)`: parse like `json.loads` (ValueError on bad input).
    pub fn loads(s: &str) -> Result<Json, PyErr> {
        let mut p = Parser { s: s.chars().collect(), i: 0 };
        p.ws();
        let v = p.value()?;
        p.ws();
        if p.i < p.s.len() {
            return Err(p.fail("Extra data", p.i));
        }
        Ok(v)
    }
    /// `Json.null()`
    pub fn null() -> Json {
        Json::Null
    }
    /// `Json.of_bool(b)`
    pub fn of_bool(b: bool) -> Json {
        Json::Bool(b)
    }
    /// `Json.of_int(i)`
    pub fn of_int(i: i64) -> Json {
        Json::Int(i)
    }
    /// `Json.of_float(x)`
    pub fn of_float(x: f64) -> Json {
        Json::Float(x)
    }
    /// `Json.of_str(s)`
    pub fn of_str(s: &str) -> Json {
        Json::Str(s.to_string())
    }
    /// `Json.of_strs(xs)`: a list of strings.
    pub fn of_strs(xs: &[String]) -> Json {
        Json::List(xs.iter().map(|s| Json::Str(s.clone())).collect())
    }
    /// `Json.new_list()`
    pub fn new_list() -> Json {
        Json::List(vec![])
    }
    /// `Json.new_dict()`
    pub fn new_dict() -> Json {
        Json::Dict(IndexMap::new())
    }

    /// `"null"`, `"bool"`, `"int"`, `"float"`, `"str"`, `"list"` or `"dict"`.
    pub fn kind(&self) -> String {
        match self {
            Json::Null => "null",
            Json::Bool(_) => "bool",
            Json::Int(_) => "int",
            Json::Float(_) => "float",
            Json::Str(_) => "str",
            Json::List(_) => "list",
            Json::Dict(_) => "dict",
        }
        .to_string()
    }
    /// `v.is_null()`
    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }
    /// The string, or `default` when this is not a string.
    pub fn str_or(&self, default: &str) -> String {
        match self {
            Json::Str(s) => s.clone(),
            _ => default.to_string(),
        }
    }
    /// The integer, or `default` when this is not an int.
    pub fn int_or(&self, default: i64) -> i64 {
        match self {
            Json::Int(i) => *i,
            _ => default,
        }
    }
    /// The number (an int converted), or `default` when this is not a number.
    pub fn float_or(&self, default: f64) -> f64 {
        match self {
            Json::Int(i) => *i as f64,
            Json::Float(x) => *x,
            _ => default,
        }
    }
    /// The bool, or `default` when this is not a bool.
    pub fn bool_or(&self, default: bool) -> bool {
        match self {
            Json::Bool(b) => *b,
            _ => default,
        }
    }
    /// Like Python's `str(v)`: a string as it is, anything else as JSON
    /// (`ensure_ascii=False`).
    pub fn text(&self) -> String {
        match self {
            Json::Str(s) => s.clone(),
            v => v.dumps(None, false),
        }
    }
    /// Python truthiness: null, false, 0, "", [] and {} are false.
    pub fn truthy(&self) -> bool {
        match self {
            Json::Null => false,
            Json::Bool(b) => *b,
            Json::Int(i) => *i != 0,
            Json::Float(x) => *x != 0.0,
            Json::Str(s) => !s.is_empty(),
            Json::List(v) => !v.is_empty(),
            Json::Dict(d) => !d.is_empty(),
        }
    }
    fn get_ref(&self, key: &str) -> &Json {
        match self {
            Json::Dict(d) => d.get(key).unwrap_or(&NULL),
            _ => &NULL,
        }
    }
    /// `d[key]` of a dict (a copy), null when it is missing or this is not a dict.
    pub fn get(&self, key: &str) -> Json {
        self.get_ref(key).clone()
    }
    /// A string member, or `default` (`d.get(key, default)` for strings).
    pub fn get_str(&self, key: &str, default: &str) -> String {
        self.get_ref(key).str_or(default)
    }
    /// `key in d`
    pub fn has(&self, key: &str) -> bool {
        matches!(self, Json::Dict(d) if d.contains_key(key))
    }
    /// The keys of a dict, in order (empty for anything else).
    pub fn keys(&self) -> Vec<String> {
        match self {
            Json::Dict(d) => d.keys().cloned().collect(),
            _ => vec![],
        }
    }
    /// `xs[i]` of a list (negative from the end), null when out of range.
    pub fn at(&self, i: i64) -> Json {
        match self {
            Json::List(v) => {
                let j = if i < 0 { i + v.len() as i64 } else { i };
                if j >= 0 && (j as usize) < v.len() {
                    v[j as usize].clone()
                } else {
                    Json::Null
                }
            }
            _ => Json::Null,
        }
    }
    /// The elements of a list, or the values of a dict.
    pub fn items(&self) -> Vec<Json> {
        match self {
            Json::List(v) => v.clone(),
            Json::Dict(d) => d.values().cloned().collect(),
            _ => vec![],
        }
    }
    /// The strings of a list (other elements are skipped).
    pub fn strs(&self) -> Vec<String> {
        match self {
            Json::List(v) => v.iter().filter_map(|x| if let Json::Str(s) = x { Some(s.clone()) } else { None }).collect(),
            _ => vec![],
        }
    }
    /// `len(v)` of a list, dict or string; 0 otherwise.
    pub fn size(&self) -> i64 {
        (match self {
            Json::List(v) => v.len(),
            Json::Dict(d) => d.len(),
            Json::Str(s) => s.chars().count(),
            _ => 0,
        }) as i64
    }
    /// `d[key] = v` (a non-dict becomes an empty dict first).
    pub fn set(&mut self, key: &str, v: &Json) {
        if !matches!(self, Json::Dict(_)) {
            *self = Json::new_dict();
        }
        if let Json::Dict(d) = self {
            d.insert(key.to_string(), v.clone());
        }
    }
    /// `d.pop(key, None)`
    pub fn remove(&mut self, key: &str) {
        if let Json::Dict(d) = self {
            d.shift_remove(key);
        }
    }
    /// `xs.append(v)` (a non-list becomes an empty list first).
    pub fn append(&mut self, v: &Json) {
        if !matches!(self, Json::List(_)) {
            *self = Json::new_list();
        }
        if let Json::List(xs) = self {
            xs.push(v.clone());
        }
    }
    /// A copy.
    pub fn copy(&self) -> Json {
        self.clone()
    }
    /// `json.dumps(v, indent=indent, ensure_ascii=ensure_ascii)`.
    pub fn dumps(&self, indent: Option<i64>, ensure_ascii: bool) -> String {
        let mut out = String::new();
        self.write(&mut out, indent.map(|n| n.max(0) as usize), 0, ensure_ascii);
        out
    }

    fn write(&self, out: &mut String, indent: Option<usize>, level: usize, ascii: bool) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(i) => out.push_str(&i.to_string()),
            Json::Float(x) => crate::pyrt::stdlib::Json::json(x, out),
            Json::Str(s) => write_str(out, s, ascii),
            Json::List(v) if v.is_empty() => out.push_str("[]"),
            Json::Dict(d) if d.is_empty() => out.push_str("{}"),
            Json::List(v) => {
                out.push('[');
                for (k, x) in v.iter().enumerate() {
                    sep(out, indent, level + 1, k > 0);
                    x.write(out, indent, level + 1, ascii);
                }
                close(out, indent, level);
                out.push(']');
            }
            Json::Dict(d) => {
                out.push('{');
                for (k, (key, x)) in d.iter().enumerate() {
                    sep(out, indent, level + 1, k > 0);
                    write_str(out, key, ascii);
                    out.push_str(": ");
                    x.write(out, indent, level + 1, ascii);
                }
                close(out, indent, level);
                out.push('}');
            }
        }
    }
}

/// Before an element: `, ` without indent; `,\n` and the indentation with it.
fn sep(out: &mut String, indent: Option<usize>, level: usize, comma: bool) {
    match indent {
        None => {
            if comma {
                out.push_str(", ")
            }
        }
        Some(n) => {
            if comma {
                out.push(',');
            }
            out.push('\n');
            out.push_str(&" ".repeat(n * level));
        }
    }
}

fn close(out: &mut String, indent: Option<usize>, level: usize) {
    if let Some(n) = indent {
        out.push('\n');
        out.push_str(&" ".repeat(n * level));
    }
}

fn write_str(out: &mut String, s: &str, ascii: bool) {
    if ascii {
        crate::pyrt::stdlib::Json::json(s, out);
        return;
    }
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

impl crate::pyrt::Py for Json {
    fn fmt_repr(&self, out: &mut String) {
        out.push_str("Json(");
        out.push_str(&self.dumps(None, false));
        out.push(')');
    }
    fn fmt_str(&self, out: &mut String) {
        out.push_str(&self.dumps(None, false));
    }
}

struct Parser {
    s: Vec<char>,
    i: usize,
}

impl Parser {
    /// CPython's message: `Expecting value: line 1 column 5 (char 4)`.
    fn fail(&self, what: &str, at: usize) -> PyErr {
        let before = &self.s[..at.min(self.s.len())];
        let line = before.iter().filter(|&&c| c == '\n').count() + 1;
        let col = at - before.iter().rposition(|&c| c == '\n').map(|p| p + 1).unwrap_or(0) + 1;
        PyErr::new(VALUE_ERROR, format!("{}: line {} column {} (char {})", what, line, col, at))
    }
    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], ' ' | '\t' | '\n' | '\r') {
            self.i += 1;
        }
    }
    fn peek(&self) -> Option<char> {
        self.s.get(self.i).copied()
    }
    fn word(&mut self, w: &str) -> bool {
        let n = w.chars().count();
        if self.i + n <= self.s.len() && self.s[self.i..self.i + n].iter().copied().eq(w.chars()) {
            self.i += n;
            true
        } else {
            false
        }
    }
    fn value(&mut self) -> Result<Json, PyErr> {
        let start = self.i;
        match self.peek() {
            Some('{') => self.dict(),
            Some('[') => self.list(),
            Some('"') => Ok(Json::Str(self.string()?)),
            Some('n') if self.word("null") => Ok(Json::Null),
            Some('t') if self.word("true") => Ok(Json::Bool(true)),
            Some('f') if self.word("false") => Ok(Json::Bool(false)),
            Some('N') if self.word("NaN") => Ok(Json::Float(f64::NAN)),
            Some('I') if self.word("Infinity") => Ok(Json::Float(f64::INFINITY)),
            Some('-') if self.word("-Infinity") => Ok(Json::Float(f64::NEG_INFINITY)),
            Some(c) if c == '-' || c.is_ascii_digit() => self.number(),
            _ => Err(self.fail("Expecting value", start)),
        }
    }
    fn number(&mut self) -> Result<Json, PyErr> {
        let start = self.i;
        if self.peek() == Some('-') {
            self.i += 1;
        }
        match self.peek() {
            Some('0') => self.i += 1,
            Some(c) if c.is_ascii_digit() => {
                while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    self.i += 1;
                }
            }
            _ => return Err(self.fail("Expecting value", start)),
        }
        let mut float = false;
        if self.peek() == Some('.') && self.s.get(self.i + 1).is_some_and(|c| c.is_ascii_digit()) {
            float = true;
            self.i += 1;
            while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                self.i += 1;
            }
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            let save = self.i;
            self.i += 1;
            if matches!(self.peek(), Some('+' | '-')) {
                self.i += 1;
            }
            if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                float = true;
                while self.peek().is_some_and(|c| c.is_ascii_digit()) {
                    self.i += 1;
                }
            } else {
                self.i = save;
            }
        }
        let text: String = self.s[start..self.i].iter().collect();
        if !float {
            if let Ok(i) = text.parse::<i64>() {
                return Ok(Json::Int(i));
            }
        }
        Ok(Json::Float(text.parse::<f64>().unwrap_or(f64::NAN)))
    }
    fn string(&mut self) -> Result<String, PyErr> {
        let start = self.i;
        self.i += 1;
        let mut out = String::new();
        loop {
            let Some(c) = self.peek() else {
                return Err(self.fail("Unterminated string starting at", start));
            };
            self.i += 1;
            match c {
                '"' => return Ok(out),
                '\\' => {
                    let Some(e) = self.peek() else {
                        return Err(self.fail("Unterminated string starting at", start));
                    };
                    self.i += 1;
                    match e {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'b' => out.push('\u{8}'),
                        'f' => out.push('\u{c}'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'u' => {
                            let u = self.hex4().ok_or_else(|| self.fail("Invalid \\uXXXX escape", self.i - 1))?;
                            if (0xD800..0xDC00).contains(&u) && self.s.get(self.i) == Some(&'\\') && self.s.get(self.i + 1) == Some(&'u') {
                                let save = self.i;
                                self.i += 2;
                                match self.hex4() {
                                    Some(l) if (0xDC00..0xE000).contains(&l) => {
                                        let c = 0x10000 + ((u - 0xD800) << 10) + (l - 0xDC00);
                                        out.push(char::from_u32(c).unwrap_or('\u{FFFD}'));
                                        continue;
                                    }
                                    _ => self.i = save,
                                }
                            }
                            out.push(char::from_u32(u).unwrap_or('\u{FFFD}'));
                        }
                        _ => return Err(self.fail("Invalid \\escape", self.i - 2)),
                    }
                }
                c if (c as u32) < 0x20 => return Err(self.fail("Invalid control character at", self.i - 1)),
                c => out.push(c),
            }
        }
    }
    fn hex4(&mut self) -> Option<u32> {
        if self.i + 4 > self.s.len() {
            return None;
        }
        let h: String = self.s[self.i..self.i + 4].iter().collect();
        let v = u32::from_str_radix(&h, 16).ok()?;
        self.i += 4;
        Some(v)
    }
    fn list(&mut self) -> Result<Json, PyErr> {
        self.i += 1;
        let mut v = vec![];
        self.ws();
        if self.peek() == Some(']') {
            self.i += 1;
            return Ok(Json::List(v));
        }
        loop {
            self.ws();
            v.push(self.value()?);
            self.ws();
            match self.peek() {
                Some(',') => self.i += 1,
                Some(']') => {
                    self.i += 1;
                    return Ok(Json::List(v));
                }
                _ => return Err(self.fail("Expecting ',' delimiter", self.i)),
            }
        }
    }
    fn dict(&mut self) -> Result<Json, PyErr> {
        self.i += 1;
        let mut d = IndexMap::new();
        self.ws();
        if self.peek() == Some('}') {
            self.i += 1;
            return Ok(Json::Dict(d));
        }
        loop {
            self.ws();
            if self.peek() != Some('"') {
                return Err(self.fail("Expecting property name enclosed in double quotes", self.i));
            }
            let k = self.string()?;
            self.ws();
            if self.peek() != Some(':') {
                return Err(self.fail("Expecting ':' delimiter", self.i));
            }
            self.i += 1;
            self.ws();
            let v = self.value()?;
            d.insert(k, v);
            self.ws();
            match self.peek() {
                Some(',') => self.i += 1,
                Some('}') => {
                    self.i += 1;
                    return Ok(Json::Dict(d));
                }
                _ => return Err(self.fail("Expecting ',' delimiter", self.i)),
            }
        }
    }
}

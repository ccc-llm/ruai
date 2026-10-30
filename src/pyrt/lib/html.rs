//! `html`: escaping and unescaping HTML text, like CPython's `html` module.

use super::html_tables::{HTML5, INVALID_CHARREFS, INVALID_CODEPOINTS};

/// `html.escape(s, quote=True)`.
pub fn escape(s: &str, quote: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' if quote => out.push_str("&quot;"),
            '\'' if quote => out.push_str("&#x27;"),
            c => out.push(c),
        }
    }
    out
}

fn html5(name: &str) -> Option<&'static str> {
    HTML5.binary_search_by(|(k, _)| (*k).cmp(name)).ok().map(|i| HTML5[i].1)
}

/// The replacement of one character reference (`#123;`, `#x7b`, `amp;` ...).
fn replace_charref(s: &[char]) -> String {
    if s[0] == '#' {
        let (digits, radix) = if s[1] == 'x' || s[1] == 'X' { (&s[2..], 16) } else { (&s[1..], 10) };
        let digits: String = digits.iter().filter(|&&c| c != ';').collect();
        // Python ints are unbounded: anything too large is out of range anyway
        let num = u64::from_str_radix(&digits, radix).unwrap_or(u64::MAX);
        if let Some(&(_, r)) = INVALID_CHARREFS.iter().find(|(k, _)| *k as u64 == num) {
            return r.to_string();
        }
        if (0xD800..=0xDFFF).contains(&num) || num > 0x10FFFF {
            return "\u{FFFD}".to_string();
        }
        if INVALID_CODEPOINTS.contains(&(num as u32)) {
            return String::new();
        }
        return char::from_u32(num as u32).map(String::from).unwrap_or_default();
    }
    let name: String = s.iter().collect();
    if let Some(v) = html5(&name) {
        return v.to_string();
    }
    // the longest prefix that is a name
    for x in (2..s.len()).rev() {
        let prefix: String = s[..x].iter().collect();
        if let Some(v) = html5(&prefix) {
            return format!("{}{}", v, s[x..].iter().collect::<String>());
        }
    }
    format!("&{}", name)
}

/// Length of CPython's charref match `#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\f <&#;]{1,32};?` at the start of `s`.
fn charref_len(s: &[char]) -> Option<usize> {
    let semi = |k: usize| if s.get(k) == Some(&';') { k + 1 } else { k };
    if s.first() == Some(&'#') {
        let d = s[1..].iter().take_while(|c| c.is_ascii_digit()).count();
        if d > 0 {
            return Some(semi(1 + d));
        }
        if matches!(s.get(1), Some('x' | 'X')) {
            let h = s[2..].iter().take_while(|c| c.is_ascii_hexdigit()).count();
            if h > 0 {
                return Some(semi(2 + h));
            }
        }
        return None;
    }
    let n = s.iter().take(32).take_while(|c| !matches!(c, '\t' | '\n' | '\x0c' | ' ' | '<' | '&' | '#' | ';')).count();
    (n > 0).then(|| semi(n))
}

/// `html.unescape(s)`: named and numeric character references to characters,
/// following the HTML5 rules (also for invalid references).
pub fn unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let c: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < c.len() {
        if c[i] == '&' {
            if let Some(n) = charref_len(&c[i + 1..]) {
                out.push_str(&replace_charref(&c[i + 1..i + 1 + n]));
                i += 1 + n;
                continue;
            }
        }
        out.push(c[i]);
        i += 1;
    }
    out
}

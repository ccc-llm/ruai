//! `shlex`: shell-like syntax, like CPython's `shlex` module.

/// `shlex.split(s)`: words of a POSIX shell command line (quotes and
/// backslash escapes, no comments). Errors like CPython: "No closing
/// quotation", "No escaped character".
pub fn split(s: &str) -> Result<Vec<String>, String> {
    // CPython's shlex.read_token in POSIX mode with whitespace_split = True
    const WS: &[char] = &[' ', '\t', '\r', '\n'];
    let chars: Vec<char> = s.chars().collect();
    let mut pos = 0;
    let mut out = Vec::new();
    // state: ' ' (between tokens), 'a' (in a word), a quote, or '\\'; None at EOF
    let mut state = Some(' ');
    loop {
        let mut token = String::new();
        let mut quoted = false;
        let mut escaped_state = ' ';
        loop {
            let next = chars.get(pos).copied();
            pos += 1;
            match state {
                None => break,
                Some(' ') => match next {
                    None => {
                        state = None;
                        break;
                    }
                    Some(c) if WS.contains(&c) => {
                        if !token.is_empty() || quoted {
                            break;
                        }
                    }
                    Some('\\') => {
                        escaped_state = 'a';
                        state = Some('\\');
                    }
                    Some(c @ ('\'' | '"')) => state = Some(c),
                    Some(c) => {
                        token.push(c);
                        state = Some('a');
                    }
                },
                Some(q @ ('\'' | '"')) => {
                    quoted = true;
                    match next {
                        None => return Err("No closing quotation".into()),
                        Some(c) if c == q => state = Some('a'),
                        Some('\\') if q == '"' => {
                            escaped_state = q;
                            state = Some('\\');
                        }
                        Some(c) => token.push(c),
                    }
                }
                Some('\\') => match next {
                    None => return Err("No escaped character".into()),
                    Some(c) => {
                        // inside quotes, only the quote and the escape itself are escapable
                        if escaped_state == '"' && c != '\\' && c != escaped_state {
                            token.push('\\');
                        }
                        token.push(c);
                        state = Some(escaped_state);
                    }
                },
                Some(_) => match next {
                    None => {
                        state = None;
                        break;
                    }
                    Some(c) if WS.contains(&c) => {
                        state = Some(' ');
                        if !token.is_empty() || quoted {
                            break;
                        }
                    }
                    Some(c @ ('\'' | '"')) => state = Some(c),
                    Some('\\') => {
                        escaped_state = 'a';
                        state = Some('\\');
                    }
                    Some(c) => token.push(c),
                },
            }
        }
        if token.is_empty() && !quoted {
            return Ok(out);
        }
        out.push(token);
    }
}

/// `shlex.quote(s)`: `s` as one shell word.
pub fn quote(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    let safe = |c: char| c.is_ascii_alphanumeric() || "%+,-./:=@_".contains(c);
    if s.chars().all(safe) {
        return s.to_string();
    }
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

/// `shlex.join(words)`: the words quoted and joined by spaces.
pub fn join(words: &[String]) -> String {
    words.iter().map(|w| quote(w)).collect::<Vec<_>>().join(" ")
}

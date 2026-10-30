//! `fnmatch`: shell-style wildcards (`*`, `?`, `[seq]`, `[!seq]`), with
//! CPython's semantics (its `translate`, including how invalid ranges and
//! unclosed brackets behave).

#[derive(Debug)]
enum Tok {
    Star,
    Any,
    Lit(char),
    Class { neg: bool, chars: Vec<char>, ranges: Vec<(char, char)> },
    Never,
}

/// CPython's `fnmatch._translate`, producing tokens instead of a regex.
fn compile(pat: &str) -> Vec<Tok> {
    let p: Vec<char> = pat.chars().collect();
    let n = p.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = p[i];
        i += 1;
        match c {
            '*' => {
                out.push(Tok::Star);
                while i < n && p[i] == '*' {
                    i += 1;
                }
            }
            '?' => out.push(Tok::Any),
            '[' => {
                let mut j = i;
                if j < n && p[j] == '!' {
                    j += 1;
                }
                if j < n && p[j] == ']' {
                    j += 1;
                }
                while j < n && p[j] != ']' {
                    j += 1;
                }
                if j >= n {
                    out.push(Tok::Lit('['));
                    continue;
                }
                // chunks separated by range hyphens
                let stuff: Vec<char> = p[i..j].to_vec();
                let chunks: Vec<Vec<char>> = if !stuff.contains(&'-') {
                    vec![stuff]
                } else {
                    let mut chunks: Vec<Vec<char>> = Vec::new();
                    let mut start = i;
                    let mut k = if p[i] == '!' { i + 2 } else { i + 1 };
                    while let Some(off) = (k < j).then(|| p[k..j].iter().position(|&x| x == '-')).flatten() {
                        let kk = k + off;
                        chunks.push(p[start..kk].to_vec());
                        start = kk + 1;
                        k = kk + 3;
                    }
                    let last: Vec<char> = p[start..j].to_vec();
                    if last.is_empty() {
                        chunks.last_mut().unwrap().push('-');
                    } else {
                        chunks.push(last);
                    }
                    // remove empty ranges (invalid in a regex)
                    let mut k = chunks.len() - 1;
                    while k > 0 {
                        if chunks[k - 1].last().is_some_and(|&a| a > chunks[k][0]) {
                            let mut merged = chunks[k - 1][..chunks[k - 1].len() - 1].to_vec();
                            merged.extend_from_slice(&chunks[k][1..]);
                            chunks[k - 1] = merged;
                            chunks.remove(k);
                        }
                        k -= 1;
                    }
                    chunks
                };
                i = j + 1;
                let flat: Vec<char> = chunks.iter().flatten().copied().collect();
                // the regex `stuff`: chunks joined by range hyphens
                let total = flat.len() + chunks.len() - 1;
                if total == 0 {
                    out.push(Tok::Never);
                } else if total == 1 && flat == ['!'] {
                    out.push(Tok::Any);
                } else {
                    let neg = chunks[0].first() == Some(&'!');
                    let mut chars = Vec::new();
                    let mut ranges = Vec::new();
                    for (ci, ch) in chunks.iter().enumerate() {
                        let skip_first = ci == 0 && neg;
                        for (k, &x) in ch.iter().enumerate() {
                            let starts_range = k + 1 == ch.len() && ci + 1 < chunks.len();
                            let ends_range = k == 0 && ci > 0;
                            if (skip_first && k == 0) || starts_range || ends_range {
                                continue;
                            }
                            chars.push(x);
                        }
                        if ci + 1 < chunks.len() {
                            let a = *ch.last().unwrap();
                            ranges.push((a, chunks[ci + 1][0]));
                        }
                    }
                    out.push(Tok::Class { neg, chars, ranges });
                }
            }
            c => out.push(Tok::Lit(c)),
        }
    }
    out
}

fn tok_matches(t: &Tok, c: char) -> bool {
    match t {
        Tok::Any => true,
        Tok::Lit(x) => *x == c,
        Tok::Class { neg, chars, ranges } => (chars.contains(&c) || ranges.iter().any(|&(a, b)| a <= c && c <= b)) != *neg,
        Tok::Never => false,
        Tok::Star => unreachable!(),
    }
}

/// `fnmatch.fnmatchcase(name, pat)`: case-sensitive match of the whole name.
pub fn fnmatchcase(name: &str, pat: &str) -> bool {
    let toks = compile(pat);
    let s: Vec<char> = name.chars().collect();
    // ok[j]: the tokens so far can match s[..j]
    let mut ok = vec![false; s.len() + 1];
    ok[0] = true;
    for t in &toks {
        let mut next = vec![false; s.len() + 1];
        match t {
            Tok::Star => {
                let mut seen = false;
                for j in 0..=s.len() {
                    seen |= ok[j];
                    next[j] = seen;
                }
            }
            t => {
                for j in 0..s.len() {
                    if ok[j] && tok_matches(t, s[j]) {
                        next[j + 1] = true;
                    }
                }
            }
        }
        ok = next;
    }
    ok[s.len()]
}

/// `fnmatch.fnmatch(name, pat)` (POSIX: case-sensitive, like `fnmatchcase`).
pub fn fnmatch(name: &str, pat: &str) -> bool {
    fnmatchcase(name, pat)
}

/// `fnmatch.filter(names, pat)`: the names that match.
pub fn filter(names: &[String], pat: &str) -> Vec<String> {
    names.iter().filter(|n| fnmatchcase(n, pat)).cloned().collect()
}

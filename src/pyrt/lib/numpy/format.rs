//! Printing arrays as NumPy's `arrayprint` does (default options: precision 8,
//! line width 75, threshold 1000, edge items 3).

use super::{shape_str, Kind, NdArray, Scalar};

const LINEWIDTH: usize = 75;
const THRESHOLD: usize = 1000;
const EDGE: usize = 3;
const PRECISION: usize = 8;

/// Digits of `|x|` in positional notation: the shortest repr, rounded to at
/// most `PRECISION` fractional digits (Dragon4 `unique=True`), trailing zeros
/// trimmed but the point kept: `1.`, `2.5`, `0.33333333`.
fn positional(x: f64, f32: bool) -> (String, String) {
    let a = x.abs();
    let mut s = if f32 { format!("{}", a as f32) } else { format!("{}", a) };
    if let Some(p) = s.find('.') {
        if s.len() - p - 1 > PRECISION {
            s = format!("{:.*}", PRECISION, a);
        }
    }
    let (int, frac) = match s.split_once('.') {
        Some((i, f)) => (i.to_string(), f.trim_end_matches('0').to_string()),
        None => (s, String::new()),
    };
    let int = if x.is_sign_negative() { format!("-{}", int) } else { int };
    (int, frac)
}

/// Scientific digits of `|x|`: mantissa integer digit, fraction digits, exponent.
fn scientific(x: f64, f32: bool) -> (String, String, i32) {
    let a = x.abs();
    let mut s = if f32 { format!("{:e}", a as f32) } else { format!("{:e}", a) };
    if s.split_once('e').unwrap().0.split_once('.').is_some_and(|(_, f)| f.len() > PRECISION) {
        s = format!("{:.*e}", PRECISION, a);
    }
    let (m, e) = s.split_once('e').unwrap();
    let (int, frac) = match m.split_once('.') {
        Some((i, f)) => (i.to_string(), f.trim_end_matches('0').to_string()),
        None => (m.to_string(), String::new()),
    };
    let int = if x.is_sign_negative() { format!("-{}", int) } else { int };
    (int, frac, e.parse().unwrap())
}

/// The format of every element of one array (NumPy's `FloatingFormat`,
/// `IntegerFormat`, `BoolFormat`).
enum Fmt {
    Bool { scalar: bool },
    Int { width: usize },
    Float { exp: bool, pad_left: usize, pad_right: usize, precision: usize, exp_size: usize, f32: bool },
}

fn float_fmt(vals: &[f64], f32: bool) -> Fmt {
    let finite: Vec<f64> = vals.iter().copied().filter(|x| x.is_finite()).collect();
    let nz: Vec<f64> = finite.iter().filter(|&&x| x != 0.0).map(|x| x.abs()).collect();
    let mut exp = false;
    if !nz.is_empty() {
        let mx = nz.iter().cloned().fold(f64::MIN, f64::max);
        let mn = nz.iter().cloned().fold(f64::MAX, f64::min);
        exp = mx >= 1e8 || mn < 0.0001 || mx / mn > 1000.0;
    }
    let (mut pad_left, mut pad_right, mut precision, mut exp_size) = (0, 0, PRECISION, 0);
    if finite.is_empty() {
    } else if exp {
        let parts: Vec<(String, String, i32)> = finite.iter().map(|&x| scientific(x, f32)).collect();
        precision = parts.iter().map(|p| p.1.len()).max().unwrap();
        exp_size = parts.iter().map(|p| p.2.unsigned_abs().to_string().len().max(2)).max().unwrap();
        pad_left = parts.iter().map(|p| p.0.len()).max().unwrap();
        pad_right = exp_size + 2 + precision;
    } else {
        let parts: Vec<(String, String)> = finite.iter().map(|&x| positional(x, f32)).collect();
        pad_left = parts.iter().map(|p| p.0.len()).max().unwrap();
        pad_right = parts.iter().map(|p| p.1.len()).max().unwrap();
    }
    if finite.len() != vals.len() {
        let neginf = vals.iter().any(|&x| x.is_infinite() && x < 0.0) as usize;
        let offset = pad_right + 1;
        pad_left = pad_left.max(3usize.saturating_sub(offset)).max((3 + neginf).saturating_sub(offset));
    }
    Fmt::Float { exp, pad_left, pad_right, precision, exp_size, f32 }
}

fn make_fmt(a: &NdArray, items: &[Scalar]) -> Fmt {
    match a.kind() {
        Kind::Bool => Fmt::Bool { scalar: a.shape.is_empty() },
        Kind::Int => {
            let width = items.iter().map(|x| x.i().to_string().len()).max().unwrap_or(0);
            Fmt::Int { width }
        }
        k => float_fmt(&items.iter().map(|x| x.f()).collect::<Vec<_>>(), k == Kind::F32),
    }
}

fn fmt_one(f: &Fmt, x: Scalar) -> String {
    match *f {
        Fmt::Bool { scalar } => match (x.b(), scalar) {
            (true, false) => " True".into(),
            (true, true) => "True".into(),
            (false, _) => "False".into(),
        },
        Fmt::Int { width } => format!("{:>w$}", x.i(), w = width),
        Fmt::Float { exp, pad_left, pad_right, precision, exp_size, f32 } => {
            let v = x.f();
            if !v.is_finite() {
                let s = if v.is_nan() { "nan".to_string() } else if v < 0.0 { "-inf".into() } else { "inf".into() };
                return format!("{:>w$}", s, w = pad_left + pad_right + 1);
            }
            if exp {
                let (int, frac, e) = scientific(v, f32);
                let frac = format!("{:0<p$}", frac, p = precision);
                let sign = if e < 0 { '-' } else { '+' };
                format!("{:>l$}.{}e{}{:0>es$}", int, frac, sign, e.unsigned_abs(), l = pad_left, es = exp_size)
            } else {
                let (int, frac) = positional(v, f32);
                format!("{:>l$}.{:<r$}", int, frac, l = pad_left, r = pad_right)
            }
        }
    }
}

/// The elements used to choose the format: all, or the edges of a summarized array.
fn edge_items(a: &NdArray, summarize: bool) -> Vec<Scalar> {
    if !summarize {
        return a.scalars();
    }
    // the index along each axis that survives `_leading_trailing`
    let keep = |d: usize, i: usize| d <= 2 * EDGE || i < EDGE || i >= d - EDGE;
    let items = a.scalars();
    let nd = a.shape.len();
    let mut out = vec![];
    let mut idx = vec![0usize; nd];
    for x in items {
        if (0..nd).all(|k| keep(a.shape[k], idx[k])) {
            out.push(x);
        }
        for k in (0..nd).rev() {
            idx[k] += 1;
            if idx[k] < a.shape[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    out
}

fn extend_line(s: &mut String, line: &mut String, word: &str, width: usize, prefix: &str) {
    let mut wrap = line.chars().count() + word.chars().count() > width;
    if line.chars().count() <= prefix.chars().count() {
        wrap = false;
    }
    if wrap {
        s.push_str(line.trim_end());
        s.push('\n');
        *line = prefix.to_string();
    }
    line.push_str(word);
}

struct Ctx<'a> {
    items: Vec<Scalar>,
    shape: &'a [usize],
    fmt: Fmt,
    sep: &'a str,
    summarize: bool,
}

impl Ctx<'_> {
    /// The flat index of `index` (negative parts count from the end).
    fn flat(&self, index: &[i64]) -> usize {
        let mut f = 0usize;
        for (k, &i) in index.iter().enumerate() {
            let d = self.shape[k];
            let i = if i < 0 { (d as i64 + i) as usize } else { i as usize };
            f = f * d + i;
        }
        f
    }

    fn rec(&self, index: &mut Vec<i64>, hanging: &str, width: usize) -> String {
        let axis = index.len();
        let left = self.shape.len() - axis;
        if left == 0 {
            return fmt_one(&self.fmt, self.items[self.flat(index)]);
        }
        let next_hanging = format!("{} ", hanging);
        let next_width = width - 1;
        let n = self.shape[axis];
        let show = self.summarize && 2 * EDGE < n;
        let (lead, trail) = if show { (EDGE, EDGE) } else { (0, n) };
        let mut s = String::new();
        if left == 1 {
            let sep_r = self.sep.trim_end();
            let elem_width = width - sep_r.len().max(1);
            let mut line = hanging.to_string();
            for i in 0..lead {
                index.push(i as i64);
                let w = self.rec(index, &next_hanging, next_width);
                index.pop();
                extend_line(&mut s, &mut line, &w, elem_width, hanging);
                line.push_str(self.sep);
            }
            if show {
                extend_line(&mut s, &mut line, "...", elem_width, hanging);
                line.push_str(self.sep);
            }
            for i in (2..=trail).rev() {
                index.push(-(i as i64));
                let w = self.rec(index, &next_hanging, next_width);
                index.pop();
                extend_line(&mut s, &mut line, &w, elem_width, hanging);
                line.push_str(self.sep);
            }
            index.push(-1);
            let w = self.rec(index, &next_hanging, next_width);
            index.pop();
            extend_line(&mut s, &mut line, &w, elem_width, hanging);
            s.push_str(&line);
        } else {
            let line_sep = format!("{}{}", self.sep.trim_end(), "\n".repeat(left - 1));
            for i in 0..lead {
                index.push(i as i64);
                let nested = self.rec(index, &next_hanging, next_width);
                index.pop();
                s.push_str(hanging);
                s.push_str(&nested);
                s.push_str(&line_sep);
            }
            if show {
                s.push_str(hanging);
                s.push_str("...");
                s.push_str(&line_sep);
            }
            for i in (2..=trail).rev() {
                index.push(-(i as i64));
                let nested = self.rec(index, &next_hanging, next_width);
                index.pop();
                s.push_str(hanging);
                s.push_str(&nested);
                s.push_str(&line_sep);
            }
            index.push(-1);
            let nested = self.rec(index, &next_hanging, next_width);
            index.pop();
            s.push_str(hanging);
            s.push_str(&nested);
        }
        format!("[{}]", &s[hanging.len()..])
    }
}

/// `np.array2string(a, separator=sep, prefix=prefix)` with a suffix of `suffix_len` characters.
fn array2string(a: &NdArray, sep: &str, prefix: &str, suffix_len: usize) -> String {
    if a.len() == 0 {
        return "[]".into();
    }
    let summarize = a.len() > THRESHOLD;
    let fmt_items = edge_items(a, summarize);
    let ctx = Ctx { items: a.scalars(), shape: &a.shape, fmt: make_fmt(a, &fmt_items), sep, summarize };
    let hanging = format!(" {}", " ".repeat(prefix.len()));
    ctx.rec(&mut vec![], &hanging, LINEWIDTH - suffix_len)
}

/// `str(x)` of a NumPy scalar (like the Python value).
fn scalar_str(x: Scalar, k: Kind) -> String {
    match x {
        Scalar::B(b) => if b { "True" } else { "False" }.into(),
        Scalar::I(i) => i.to_string(),
        Scalar::F(f) if k == Kind::F32 => crate::pyrt::float32_repr(f as f32),
        Scalar::F(f) => crate::pyrt::float_repr(f),
    }
}

/// `str(a)`
pub fn array_str(a: &NdArray) -> String {
    if a.shape.is_empty() {
        return scalar_str(a.first(), a.kind());
    }
    array2string(a, " ", "", 0)
}

/// `repr(a)`
pub fn array_repr(a: &NdArray) -> String {
    if a.scalar {
        return match a.first() {
            Scalar::B(b) => if b { "np.True_" } else { "np.False_" }.into(),
            Scalar::I(i) => format!("np.int64({})", i),
            Scalar::F(f) if a.kind() == Kind::F32 => format!("np.float32({})", crate::pyrt::float32_repr(f as f32)),
            Scalar::F(f) => format!("np.float64({})", crate::pyrt::float_repr(f)),
        };
    }
    let prefix = "array(";
    let lst = if a.len() > 0 || a.shape == [0] {
        array2string(a, ", ", prefix, 1)
    } else {
        format!("[], shape={}", shape_str(&a.shape))
    };
    let mut extras = vec![];
    if (a.len() == 0 && a.shape != [0]) || a.len() > THRESHOLD {
        if a.len() > THRESHOLD || !lst.contains("shape=") {
            extras.push(format!("shape={}", shape_str(&a.shape)));
        }
    }
    if a.len() == 0 || a.kind() == Kind::F32 {
        extras.push(format!("dtype={}", a.kind().name()));
    }
    if a.len() == 0 && a.shape != [0] {
        // array2string printed "[]"; NumPy's repr carries shape and dtype
        return format!("array([], shape={}, dtype={})", shape_str(&a.shape), a.kind().name());
    }
    if extras.is_empty() {
        return format!("{}{})", prefix, lst);
    }
    let arr = format!("{}{},", prefix, lst);
    let extra = format!("{})", extras.join(", "));
    let last = arr.len() - arr.rfind('\n').map(|i| i + 1).unwrap_or(0);
    let spacer = if last + extra.len() + 1 > LINEWIDTH { format!("\n{}", " ".repeat(prefix.len())) } else { " ".into() };
    format!("{}{}{}", arr, spacer, extra)
}

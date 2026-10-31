//! Printing tensors as PyTorch's `_tensor_str` does (precision 4, line width
//! 80, threshold 1000, edge items 3).

use super::{DType, Tensor};

const PRECISION: usize = 4;
const LINEWIDTH: usize = 80;
const THRESHOLD: usize = 1000;
const EDGE: usize = 3;

/// Python's `"{:.4e}"`: two exponent digits at least.
fn sci(x: f64, width: usize) -> String {
    let s = format!("{:.*e}", PRECISION, x);
    let (m, e) = s.split_once('e').unwrap();
    let e: i32 = e.parse().unwrap();
    let s = format!("{}e{}{:02}", m, if e < 0 { '-' } else { '+' }, e.abs());
    format!("{:>w$}", s, w = width)
}

/// Python's `str(float)` of inf / nan.
fn special(x: f64) -> String {
    if x.is_nan() {
        "nan".into()
    } else if x > 0.0 {
        "inf".into()
    } else {
        "-inf".into()
    }
}

struct Formatter {
    float: bool,
    int_mode: bool,
    sci_mode: bool,
    width: usize,
}

impl Formatter {
    fn new(dt: DType, vals: &[f64]) -> Formatter {
        let mut f = Formatter { float: dt == DType::Float32, int_mode: true, sci_mode: false, width: 1 };
        if !f.float {
            for &v in vals {
                f.width = f.width.max(Formatter::plain(dt, v).len());
            }
            return f;
        }
        let nz: Vec<f64> = vals.iter().copied().filter(|v| v.is_finite() && *v != 0.0).collect();
        if nz.is_empty() {
            return f;
        }
        let abs: Vec<f64> = nz.iter().map(|v| v.abs()).collect();
        let mn = abs.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = abs.iter().cloned().fold(0.0, f64::max);
        f.int_mode = nz.iter().all(|v| *v == v.ceil());
        if f.int_mode {
            if mx / mn > 1000.0 || mx > 1.0e8 {
                f.sci_mode = true;
                for &v in &nz {
                    f.width = f.width.max(sci(v, 0).len());
                }
            } else {
                for &v in &nz {
                    f.width = f.width.max(format!("{:.0}", v).len() + 1);
                }
            }
        } else if mx / mn > 1000.0 || mx > 1.0e8 || mn < 1.0e-4 {
            f.sci_mode = true;
            for &v in &nz {
                f.width = f.width.max(sci(v, 0).len());
            }
        } else {
            for &v in &nz {
                f.width = f.width.max(format!("{:.*}", PRECISION, v).len());
            }
        }
        f
    }
    fn plain(dt: DType, v: f64) -> String {
        match dt {
            DType::Bool => if v != 0.0 { "True" } else { "False" }.into(),
            _ => (v as i64).to_string(),
        }
    }
    fn format(&self, dt: DType, v: f64) -> String {
        let s = if self.float {
            if self.sci_mode {
                if v.is_finite() {
                    sci(v, self.width)
                } else {
                    special(v)
                }
            } else if self.int_mode {
                if v.is_finite() {
                    format!("{:.0}.", v)
                } else {
                    special(v)
                }
            } else if v.is_finite() {
                format!("{:.*}", PRECISION, v)
            } else {
                special(v)
            }
        } else {
            Formatter::plain(dt, v)
        };
        format!("{:>w$}", s, w = self.width)
    }
}

struct View<'a> {
    vals: &'a [f64],
    shape: &'a [usize],
}

impl View<'_> {
    fn sub(&self, i: usize) -> View<'_> {
        let inner: usize = self.shape[1..].iter().product();
        View { vals: &self.vals[i * inner..(i + 1) * inner], shape: &self.shape[1..] }
    }
}

fn vector(v: &View, indent: usize, summarize: bool, f: &Formatter, dt: DType) -> String {
    let elem = f.width + 2;
    let per_line = ((LINEWIDTH - indent) / elem).max(1);
    let n = v.shape[0];
    let data: Vec<String> = if summarize && n > 2 * EDGE {
        let mut d: Vec<String> = v.vals[..EDGE].iter().map(|&x| f.format(dt, x)).collect();
        d.push(" ...".into());
        d.extend(v.vals[n - EDGE..].iter().map(|&x| f.format(dt, x)));
        d
    } else {
        v.vals.iter().map(|&x| f.format(dt, x)).collect()
    };
    let lines: Vec<String> = data.chunks(per_line).map(|c| c.join(", ")).collect();
    format!("[{}]", lines.join(&format!(",\n{}", " ".repeat(indent + 1))))
}

fn tensor_str(v: &View, indent: usize, summarize: bool, f: &Formatter, dt: DType) -> String {
    let dim = v.shape.len();
    if dim == 0 {
        return f.format(dt, v.vals[0]);
    }
    if dim == 1 {
        return vector(v, indent, summarize, f, dt);
    }
    let n = v.shape[0];
    let slices: Vec<String> = if summarize && n > 2 * EDGE {
        let mut s: Vec<String> = (0..EDGE).map(|i| tensor_str(&v.sub(i), indent + 1, summarize, f, dt)).collect();
        s.push("...".into());
        s.extend((n - EDGE..n).map(|i| tensor_str(&v.sub(i), indent + 1, summarize, f, dt)));
        s
    } else {
        (0..n).map(|i| tensor_str(&v.sub(i), indent + 1, summarize, f, dt)).collect()
    };
    let sep = format!(",{}{}", "\n".repeat(dim - 1), " ".repeat(indent + 1));
    format!("[{}]", slices.join(&sep))
}

/// The values a summarized tensor shows (for the formatter's widths).
fn summarized(vals: &[f64], shape: &[usize]) -> Vec<f64> {
    if shape.is_empty() {
        return vals.to_vec();
    }
    let n = shape[0];
    let inner: usize = shape[1..].iter().product();
    let rows: Vec<usize> = if n > 2 * EDGE { (0..EDGE).chain(n - EDGE..n).collect() } else { (0..n).collect() };
    rows.into_iter().flat_map(|i| summarized(&vals[i * inner..(i + 1) * inner], &shape[1..])).collect()
}

/// `repr(tensor)`
pub fn tensor_repr(t: &Tensor) -> String {
    let prefix = "tensor(";
    let indent = prefix.len();
    let shape = t.dims();
    let vals: Vec<f64> = t.t.to_vec().into_iter().map(|x| x as f64).collect();
    let mut suffixes = vec![];
    let body = if vals.is_empty() {
        if shape.len() != 1 {
            suffixes.push(format!("size=({})", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ") + if shape.len() == 1 { "," } else { "" }));
        }
        if t.dt != DType::Float32 {
            suffixes.push(format!("dtype={}", t.dt.name()));
        }
        "[]".to_string()
    } else {
        let summarize = vals.len() > THRESHOLD;
        let f = Formatter::new(t.dt, &if summarize { summarized(&vals, &shape) } else { vals.clone() });
        tensor_str(&View { vals: &vals, shape: &shape }, indent, summarize, &f, t.dt)
    };
    if let Some(g) = t.grad_fn {
        suffixes.push(format!("grad_fn=<{}>", g));
    } else if t.t.requires_grad() {
        suffixes.push("requires_grad=True".into());
    }
    let mut out = format!("{}{}", prefix, body);
    // PyTorch's `len(s) - s.rfind("\n") + 1`: two more than the last line's length
    let mut last = out.len() - out.rfind('\n').map(|i| i + 1).unwrap_or(0) + 2;
    for s in suffixes {
        if last + s.len() + 2 > LINEWIDTH {
            out += &format!(",\n{}{}", " ".repeat(indent), s);
            last = indent + s.len();
        } else {
            out += &format!(", {}", s);
            last += s.len() + 2;
        }
    }
    out.push(')');
    if t.param {
        format!("Parameter containing:\n{}", out)
    } else {
        out
    }
}

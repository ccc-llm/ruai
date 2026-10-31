//! Reductions (`sum`, `mean`, `max`, ...), cumulative sums, sorting.

use super::ops::axis_index;
use super::{verr, Data, Kind, NdArray, Scalar};
use crate::pyrt::err::PyErr;

/// NumPy's pairwise summation of a contiguous run (`pairwise_sum_DOUBLE`).
pub(crate) fn pairwise(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut r = 0.0;
        for &x in a {
            r += x;
        }
        r
    } else if n <= 128 {
        let mut r = [0.0; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise(&a[..n2]) + pairwise(&a[n2..])
    }
}

/// The lanes of a reduction over `axis`: for each output position, the
/// elements along the axis (in order), and the output shape.
fn lanes(a: &NdArray, axis: usize) -> (Vec<Vec<Scalar>>, Vec<usize>) {
    let items = a.scalars();
    let shape = &a.shape;
    let inner: usize = shape[axis + 1..].iter().product();
    let n = shape[axis];
    let outer: usize = shape[..axis].iter().product();
    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            out.push((0..n).map(|k| items[(o * n + k) * inner + i]).collect());
        }
    }
    let mut oshape = shape.clone();
    oshape.remove(axis);
    (out, oshape)
}

/// Is `axis` the innermost axis in memory (NumPy then sums pairwise)?
fn innermost(a: &NdArray, axis: usize) -> bool {
    let s = a.strides[axis].unsigned_abs();
    a.strides.iter().enumerate().all(|(k, t)| k == axis || a.shape[k] <= 1 || t.unsigned_abs() >= s)
}

/// The elements in memory order (how NumPy iterates a full reduction).
fn memory_order(a: &NdArray) -> Vec<Scalar> {
    let d = a.data.borrow();
    let mut offs = a.offsets();
    offs.sort_unstable();
    offs.into_iter().map(|o| d.get(o)).collect()
}

fn result(items: Vec<Scalar>, k: Kind, shape: Vec<usize>) -> NdArray {
    let mut r = NdArray::from_data(Data::from_scalars(k, items.into_iter()), shape);
    r.scalar = r.shape.is_empty();
    r
}

/// A reduction over `axis` (None: all elements) with `f(lane, pairwise_ok)`.
fn reduce(
    a: &NdArray,
    axis: Option<i64>,
    k: Kind,
    f: &dyn Fn(&[Scalar], bool) -> Result<Scalar, PyErr>,
) -> Result<NdArray, PyErr> {
    match axis {
        None => {
            let items = memory_order(a);
            Ok(result(vec![f(&items, true)?], k, vec![]))
        }
        Some(ax) => {
            let ax = axis_index(ax, a.shape.len())?;
            let pw = innermost(a, ax);
            let (ls, shape) = lanes(a, ax);
            let items = ls.iter().map(|l| f(l, pw)).collect::<Result<Vec<_>, _>>()?;
            Ok(result(items, k, shape))
        }
    }
}

/// A float32 sum in NumPy's order (pairwise, float32 accumulators).
fn pairwise32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        let mut r = 0.0f32;
        for &x in a {
            r += x;
        }
        r
    } else if n <= 128 {
        let mut r = [0.0f32; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise32(&a[..n2]) + pairwise32(&a[n2..])
    }
}

fn fsum32(l: &[Scalar], pw: bool) -> f32 {
    let v: Vec<f32> = l.iter().map(|x| x.f() as f32).collect();
    if pw {
        0.0f32 + pairwise32(&v)
    } else {
        v.iter().fold(0.0f32, |s, x| s + x)
    }
}

fn fsum(l: &[Scalar], pw: bool) -> f64 {
    let v: Vec<f64> = l.iter().map(|x| x.f()).collect();
    if pw {
        0.0 + pairwise(&v)
    } else {
        v.iter().fold(0.0, |s, x| s + x)
    }
}

fn sum_kind(k: Kind) -> Kind {
    if k == Kind::Bool {
        Kind::Int
    } else {
        k
    }
}

fn empty_err(name: &str) -> PyErr {
    verr(format!("zero-size array to reduction operation {} which has no identity", name))
}

fn extreme(l: &[Scalar], max: bool, name: &str) -> Result<Scalar, PyErr> {
    let Some(&first) = l.first() else { return Err(empty_err(name)) };
    let mut best = first;
    for &x in &l[1..] {
        match (best, x) {
            (Scalar::F(b), _) if b.is_nan() => return Ok(best),
            (_, Scalar::F(v)) if v.is_nan() => return Ok(x),
            _ => {
                let better = match (best, x) {
                    (Scalar::F(_), _) | (_, Scalar::F(_)) => {
                        if max {
                            x.f() > best.f()
                        } else {
                            x.f() < best.f()
                        }
                    }
                    _ => {
                        if max {
                            x.i() > best.i()
                        } else {
                            x.i() < best.i()
                        }
                    }
                };
                if better {
                    best = x;
                }
            }
        }
    }
    Ok(best)
}

fn arg_extreme(l: &[Scalar], max: bool, name: &str) -> Result<Scalar, PyErr> {
    if l.is_empty() {
        return Err(verr(format!("attempt to get {} of an empty sequence", name)));
    }
    let mut bi = 0;
    for i in 1..l.len() {
        let (b, x) = (l[bi], l[i]);
        if let Scalar::F(v) = b {
            if v.is_nan() {
                break;
            }
        }
        if let Scalar::F(v) = x {
            if v.is_nan() {
                bi = i;
                break;
            }
        }
        let better = if matches!(b, Scalar::F(_)) || matches!(x, Scalar::F(_)) {
            if max {
                x.f() > b.f()
            } else {
                x.f() < b.f()
            }
        } else if max {
            x.i() > b.i()
        } else {
            x.i() < b.i()
        };
        if better {
            bi = i;
        }
    }
    Ok(Scalar::I(bi as i64))
}

impl NdArray {
    /// `a.sum(axis=None)`
    pub fn sum(&self, axis: Option<i64>) -> Result<NdArray, PyErr> {
        let k = sum_kind(self.kind());
        reduce(self, axis, k, &|l, pw| {
            Ok(match k {
                Kind::Float => Scalar::F(fsum(l, pw)),
                Kind::F32 => Scalar::F(fsum32(l, pw) as f64),
                _ => Scalar::I(l.iter().fold(0i64, |s, x| s.wrapping_add(x.i()))),
            })
        })
    }
    /// `a.prod(axis=None)`
    pub fn prod(&self, axis: Option<i64>) -> Result<NdArray, PyErr> {
        let k = sum_kind(self.kind());
        reduce(self, axis, k, &|l, _| {
            Ok(match k {
                Kind::Float => Scalar::F(l.iter().fold(1.0, |s, x| s * x.f())),
                Kind::F32 => Scalar::F(l.iter().fold(1.0f32, |s, x| s * x.f() as f32) as f64),
                _ => Scalar::I(l.iter().fold(1i64, |s, x| s.wrapping_mul(x.i()))),
            })
        })
    }
    /// `a.mean(axis=None)`
    pub fn mean(&self, axis: Option<i64>) -> Result<NdArray, PyErr> {
        if self.kind() == Kind::F32 {
            return reduce(self, axis, Kind::F32, &|l, pw| Ok(Scalar::F((fsum32(l, pw) / l.len() as f32) as f64)));
        }
        reduce(self, axis, Kind::Float, &|l, pw| Ok(Scalar::F(fsum(l, pw) / l.len() as f64)))
    }
    /// `a.var(axis=None, ddof=0)`
    pub fn var(&self, axis: Option<i64>, ddof: i64) -> Result<NdArray, PyErr> {
        if self.kind() == Kind::F32 {
            return reduce(self, axis, Kind::F32, &|l, pw| {
                let mean = fsum32(l, pw) / l.len() as f32;
                let sq: Vec<Scalar> = l
                    .iter()
                    .map(|x| {
                        let d = x.f() as f32 - mean;
                        Scalar::F((d * d) as f64)
                    })
                    .collect();
                let n = l.len() as i64 - ddof;
                Ok(Scalar::F((fsum32(&sq, pw) / n.max(0) as f32) as f64))
            });
        }
        reduce(self, axis, Kind::Float, &|l, pw| {
            let mean = fsum(l, pw) / l.len() as f64;
            let sq: Vec<Scalar> = l
                .iter()
                .map(|x| {
                    let d = x.f() - mean;
                    Scalar::F(d * d)
                })
                .collect();
            let n = l.len() as i64 - ddof;
            Ok(Scalar::F(fsum(&sq, pw) / n.max(0) as f64))
        })
    }
    /// `a.std(axis=None, ddof=0)`
    pub fn std(&self, axis: Option<i64>, ddof: i64) -> Result<NdArray, PyErr> {
        let v = self.var(axis, ddof)?;
        Ok(super::ops::map_f(&v, f64::sqrt))
    }
    /// `a.max(axis=None)`
    pub fn max(&self, axis: Option<i64>) -> Result<NdArray, PyErr> {
        reduce(self, axis, self.kind(), &|l, _| extreme(l, true, "maximum"))
    }
    /// `a.min(axis=None)`
    pub fn min(&self, axis: Option<i64>) -> Result<NdArray, PyErr> {
        reduce(self, axis, self.kind(), &|l, _| extreme(l, false, "minimum"))
    }
    /// `a.argmax(axis=None)` (the flat index when axis is None)
    pub fn argmax(&self, axis: Option<i64>) -> Result<NdArray, PyErr> {
        match axis {
            None => Ok(result(vec![arg_extreme(&self.scalars(), true, "argmax")?], Kind::Int, vec![])),
            Some(_) => reduce(self, axis, Kind::Int, &|l, _| arg_extreme(l, true, "argmax")),
        }
    }
    /// `a.argmin(axis=None)`
    pub fn argmin(&self, axis: Option<i64>) -> Result<NdArray, PyErr> {
        match axis {
            None => Ok(result(vec![arg_extreme(&self.scalars(), false, "argmin")?], Kind::Int, vec![])),
            Some(_) => reduce(self, axis, Kind::Int, &|l, _| arg_extreme(l, false, "argmin")),
        }
    }
    /// `a.all(axis=None)`
    pub fn all(&self, axis: Option<i64>) -> Result<NdArray, PyErr> {
        reduce(self, axis, Kind::Bool, &|l, _| Ok(Scalar::B(l.iter().all(|x| x.b()))))
    }
    /// `a.any(axis=None)`
    pub fn any(&self, axis: Option<i64>) -> Result<NdArray, PyErr> {
        reduce(self, axis, Kind::Bool, &|l, _| Ok(Scalar::B(l.iter().any(|x| x.b()))))
    }
    /// `a.cumsum(axis=None)` (flattened when axis is None)
    pub fn cumsum(&self, axis: Option<i64>) -> Result<NdArray, PyErr> {
        self.cumulative(axis, false)
    }
    /// `a.cumprod(axis=None)`
    pub fn cumprod(&self, axis: Option<i64>) -> Result<NdArray, PyErr> {
        self.cumulative(axis, true)
    }
    fn cumulative(&self, axis: Option<i64>, prod: bool) -> Result<NdArray, PyErr> {
        let a = match axis {
            None => self.flatten(),
            Some(_) => self.contiguous(),
        };
        let ax = axis_index(axis.unwrap_or(0), a.shape.len())?;
        let k = sum_kind(a.kind());
        let mut items = a.scalars();
        let inner: usize = a.shape[ax + 1..].iter().product();
        let n = a.shape[ax];
        let outer: usize = a.shape[..ax].iter().product();
        for o in 0..outer {
            for i in 0..inner {
                let mut acc = if prod { Scalar::I(1).cast(k) } else { Scalar::I(0).cast(k) };
                for j in 0..n {
                    let p = (o * n + j) * inner + i;
                    let x = items[p];
                    acc = match (k, prod) {
                        (Kind::Float, false) => Scalar::F(acc.f() + x.f()),
                        (Kind::Float, true) => Scalar::F(acc.f() * x.f()),
                        (Kind::F32, false) => Scalar::F((acc.f() as f32 + x.f() as f32) as f64),
                        (Kind::F32, true) => Scalar::F((acc.f() as f32 * x.f() as f32) as f64),
                        (_, false) => Scalar::I(acc.i().wrapping_add(x.i())),
                        (_, true) => Scalar::I(acc.i().wrapping_mul(x.i())),
                    };
                    items[p] = acc;
                }
            }
        }
        Ok(NdArray::from_data(Data::from_scalars(k, items.into_iter()), a.shape.clone()))
    }
    /// `np.sort(a, axis=-1)`: a sorted copy (NaN last).
    pub fn sorted(&self, axis: i64) -> Result<NdArray, PyErr> {
        self.sort_lanes(axis, false)
    }
    /// `a.sort(axis=-1)`: in place.
    pub fn sort(&self, axis: i64) -> Result<(), PyErr> {
        let s = self.sort_lanes(axis, false)?;
        let offs = self.offsets();
        let mut d = self.data.borrow_mut();
        for (o, x) in offs.into_iter().zip(s.scalars()) {
            d.set(o, x);
        }
        Ok(())
    }
    /// `a.argsort(axis=-1)` (a stable sort: ties keep their order)
    pub fn argsort(&self, axis: i64) -> Result<NdArray, PyErr> {
        self.sort_lanes(axis, true)
    }
    fn sort_lanes(&self, axis: i64, args: bool) -> Result<NdArray, PyErr> {
        if self.shape.is_empty() {
            return Err(PyErr::new(
                &["AxisError", "ValueError", "IndexError", "Exception", "BaseException"],
                format!("axis {} is out of bounds for array of dimension 0", axis),
            ));
        }
        let a = self.contiguous();
        let ax = axis_index(axis, a.shape.len())?;
        let mut items = a.scalars();
        let inner: usize = a.shape[ax + 1..].iter().product();
        let n = a.shape[ax];
        let outer: usize = a.shape[..ax].iter().product();
        let key = |x: &Scalar| -> (bool, f64, i64) {
            match x {
                Scalar::F(f) => (f.is_nan(), *f, 0),
                s => (false, 0.0, s.i()),
            }
        };
        for o in 0..outer {
            for i in 0..inner {
                let pos: Vec<usize> = (0..n).map(|j| (o * n + j) * inner + i).collect();
                let mut order: Vec<usize> = (0..n).collect();
                let lane: Vec<Scalar> = pos.iter().map(|&p| items[p]).collect();
                order.sort_by(|&x, &y| {
                    let (kx, ky) = (key(&lane[x]), key(&lane[y]));
                    kx.0.cmp(&ky.0).then(kx.1.partial_cmp(&ky.1).unwrap_or(std::cmp::Ordering::Equal)).then(kx.2.cmp(&ky.2))
                });
                for (j, &p) in pos.iter().enumerate() {
                    items[p] = if args { Scalar::I(order[j] as i64) } else { lane[order[j]] };
                }
            }
        }
        let k = if args { Kind::Int } else { a.kind() };
        Ok(NdArray::from_data(Data::from_scalars(k, items.into_iter()), a.shape.clone()))
    }
}

/// `np.unique(a)`: the sorted distinct values (flattened).
pub fn unique(a: &NdArray) -> NdArray {
    let s = a.flatten().sorted(0).unwrap();
    let items = s.scalars();
    let mut out: Vec<Scalar> = vec![];
    for x in items {
        let dup = out.last().is_some_and(|y| match (x, *y) {
            (Scalar::F(p), Scalar::F(q)) => p == q || (p.is_nan() && q.is_nan()),
            (p, q) => p.i() == q.i() && p.kind() == q.kind(),
        });
        if !dup {
            out.push(x);
        }
    }
    let n = out.len();
    NdArray::from_data(Data::from_scalars(a.kind(), out.into_iter()), vec![n])
}

/// `np.dot`-style sum of products, in order (NumPy's own loop for integers).
pub(crate) fn dot_lane(x: &[Scalar], y: &[Scalar], k: Kind) -> Scalar {
    match k {
        Kind::Float => {
            let mut s = 0.0;
            for (a, b) in x.iter().zip(y) {
                s += a.f() * b.f();
            }
            Scalar::F(s)
        }
        Kind::F32 => {
            let mut s = 0.0f32;
            for (a, b) in x.iter().zip(y) {
                s += a.f() as f32 * b.f() as f32;
            }
            Scalar::F(s as f64)
        }
        Kind::Bool => Scalar::B(x.iter().zip(y).any(|(a, b)| a.b() && b.b())),
        Kind::Int => Scalar::I(x.iter().zip(y).fold(0i64, |s, (a, b)| s.wrapping_add(a.i().wrapping_mul(b.i())))),
    }
}

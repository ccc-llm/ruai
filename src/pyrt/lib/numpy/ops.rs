//! Element-wise operations with broadcasting, and indexing.

use super::{c_strides, ierr, shape_err, terr, verr, Data, Kind, NdArray, Scalar};
use crate::pyrt::err::PyErr;
use crate::pyrt::Ix;

/// An operand: an array, or a Python scalar (a "weak" type in NumPy 2: it
/// takes the array's type unless it needs a wider kind).
pub enum Arg {
    A(NdArray),
    S(Scalar),
}

/// Values an operator or function accepts: arrays and Python bool / int / float.
pub trait Operand {
    fn arg(&self) -> Arg;
    /// The value of a Python scalar or of a 0-d array.
    fn scalar_value(&self) -> Option<Scalar> {
        match self.arg() {
            Arg::S(s) => Some(s),
            Arg::A(a) if a.shape.is_empty() => Some(a.first()),
            Arg::A(_) => None,
        }
    }
}
impl Operand for i64 {
    fn arg(&self) -> Arg {
        Arg::S(Scalar::I(*self))
    }
}
impl Operand for f64 {
    fn arg(&self) -> Arg {
        Arg::S(Scalar::F(*self))
    }
}
impl Operand for bool {
    fn arg(&self) -> Arg {
        Arg::S(Scalar::B(*self))
    }
}
impl Operand for NdArray {
    fn arg(&self) -> Arg {
        Arg::A(self.clone())
    }
}
impl<T: Operand + ?Sized> Operand for &T {
    fn arg(&self) -> Arg {
        (**self).arg()
    }
}

impl Arg {
    fn shape(&self) -> &[usize] {
        match self {
            Arg::A(a) => &a.shape,
            Arg::S(_) => &[],
        }
    }
}

/// The result kind of combining two operands (NEP 50 promotion).
fn promote(a: &Arg, b: &Arg) -> Kind {
    match (a, b) {
        (Arg::A(x), Arg::A(y)) => x.kind().max(y.kind()),
        (Arg::A(x), Arg::S(s)) | (Arg::S(s), Arg::A(x)) => {
            let k = x.kind();
            match s.kind() {
                Kind::Bool => k,
                Kind::Int => k.max(Kind::Int),
                Kind::Float => Kind::Float,
            }
        }
        (Arg::S(x), Arg::S(y)) => x.kind().max(y.kind()),
    }
}

pub(crate) fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, PyErr> {
    let n = a.len().max(b.len());
    let mut out = vec![0; n];
    for i in 0..n {
        let x = if i + a.len() >= n { a[i + a.len() - n] } else { 1 };
        let y = if i + b.len() >= n { b[i + b.len() - n] } else { 1 };
        out[i] = if x == y || y == 1 {
            x
        } else if x == 1 {
            y
        } else {
            return Err(verr(format!(
                "operands could not be broadcast together with shapes {} {} ",
                shape_err(a),
                shape_err(b)
            )));
        };
    }
    Ok(out)
}

/// Strides of `a` read as an array of `shape` (0 where broadcast).
fn bstrides(a: &NdArray, shape: &[usize]) -> Vec<isize> {
    let off = shape.len() - a.shape.len();
    (0..shape.len())
        .map(|i| if i < off || a.shape[i - off] == 1 { 0 } else { a.strides[i - off] })
        .collect()
}

/// The elements of `x` broadcast to `shape`, in C order.
pub(crate) fn broadcast_items(x: &Arg, shape: &[usize]) -> Vec<Scalar> {
    let n: usize = shape.iter().product();
    match x {
        Arg::S(s) => vec![*s; n],
        Arg::A(a) => {
            let view = NdArray {
                data: a.data.clone(),
                strides: bstrides(a, shape),
                shape: shape.to_vec(),
                offset: a.offset,
                scalar: false,
            };
            view.scalars()
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
    Xor,
    Max,
    Min,
    Atan2,
}

impl Op {
    fn ufunc(self) -> &'static str {
        match self {
            Op::And => "bitwise_and",
            Op::Or => "bitwise_or",
            Op::Xor => "bitwise_xor",
            _ => "",
        }
    }
}

/// CPython's / NumPy's float divmod.
pub(crate) fn divmod_f(a: f64, b: f64) -> (f64, f64) {
    let m = a % b;
    if b == 0.0 {
        return (a / b, m);
    }
    let mut div = (a - m) / b;
    let mut m = m;
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m += b;
            div -= 1.0;
        }
    } else {
        m = 0f64.copysign(b);
    }
    let fd = if div != 0.0 {
        let f = div.floor();
        if div - f > 0.5 {
            f + 1.0
        } else {
            f
        }
    } else {
        0f64.copysign(a / b)
    };
    (fd, m)
}

fn pow_i(mut b: i64, mut e: i64) -> i64 {
    let mut r: i64 = 1;
    while e > 0 {
        if e & 1 == 1 {
            r = r.wrapping_mul(b);
        }
        b = b.wrapping_mul(b);
        e >>= 1;
    }
    r
}

fn apply(op: Op, k: Kind, x: Scalar, y: Scalar) -> Result<Scalar, PyErr> {
    use Scalar::*;
    let cmp = |o: std::cmp::Ordering| -> Option<bool> {
        Some(match op {
            Op::Lt => o.is_lt(),
            Op::Le => o.is_le(),
            Op::Gt => o.is_gt(),
            Op::Ge => o.is_ge(),
            Op::Eq => o.is_eq(),
            Op::Ne => o.is_ne(),
            _ => return None,
        })
    };
    if matches!(op, Op::Lt | Op::Le | Op::Gt | Op::Ge | Op::Eq | Op::Ne) {
        return Ok(B(if k == Kind::Float {
            let (a, b) = (x.f(), y.f());
            match a.partial_cmp(&b) {
                Some(o) => cmp(o).unwrap(),
                None => op == Op::Ne,
            }
        } else {
            cmp(x.i().cmp(&y.i())).unwrap()
        }));
    }
    match k {
        Kind::Bool => match op {
            Op::Add | Op::Or | Op::Max => Ok(B(x.b() || y.b())),
            Op::Mul | Op::And | Op::Min => Ok(B(x.b() && y.b())),
            Op::Xor => Ok(B(x.b() != y.b())),
            Op::Sub => Err(terr(
                "numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead.",
            )),
            Op::Div | Op::Atan2 => apply(op, Kind::Float, x, y),
            _ => apply(op, Kind::Int, x, y),
        },
        Kind::Int => {
            let (a, b) = (x.i(), y.i());
            Ok(match op {
                Op::Add => I(a.wrapping_add(b)),
                Op::Sub => I(a.wrapping_sub(b)),
                Op::Mul => I(a.wrapping_mul(b)),
                Op::Div | Op::Atan2 => return apply(op, Kind::Float, x, y),
                Op::FloorDiv => I(if b == 0 { 0 } else { a.wrapping_div(b) - (((a % b) != 0) && ((a < 0) != (b < 0))) as i64 }),
                Op::Mod => I(if b == 0 { 0 } else { ((a % b) + b) % b }),
                Op::Pow => {
                    if b < 0 {
                        return Err(verr("Integers to negative integer powers are not allowed."));
                    }
                    I(pow_i(a, b))
                }
                Op::And => I(a & b),
                Op::Or => I(a | b),
                Op::Xor => I(a ^ b),
                Op::Max => I(a.max(b)),
                Op::Min => I(a.min(b)),
                _ => unreachable!(),
            })
        }
        Kind::Float => {
            let (a, b) = (x.f(), y.f());
            Ok(F(match op {
                Op::Add => a + b,
                Op::Sub => a - b,
                Op::Mul => a * b,
                Op::Div => a / b,
                Op::FloorDiv => divmod_f(a, b).0,
                Op::Mod => {
                    if b == 0.0 {
                        a % b
                    } else {
                        divmod_f(a, b).1
                    }
                }
                Op::Pow => a.powf(b),
                Op::Max => {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.max(b)
                    }
                }
                Op::Min => {
                    if a.is_nan() || b.is_nan() {
                        f64::NAN
                    } else {
                        a.min(b)
                    }
                }
                Op::Atan2 => a.atan2(b),
                Op::And | Op::Or | Op::Xor => {
                    return Err(terr(format!(
                        "ufunc '{}' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''",
                        op.ufunc()
                    )))
                }
                _ => unreachable!(),
            }))
        }
    }
}

/// `a op b` with broadcasting.
pub(crate) fn binary(op: Op, a: &Arg, b: &Arg) -> Result<NdArray, PyErr> {
    let shape = broadcast_shapes(a.shape(), b.shape())?;
    let k = promote(a, b);
    let xs = broadcast_items(a, &shape);
    let ys = broadcast_items(b, &shape);
    // `x ** 2` on a float array squares (NumPy's fast path)
    let square = op == Op::Pow && k == Kind::Float && matches!(b, Arg::S(s) if s.f() == 2.0);
    let mut out = Vec::with_capacity(xs.len());
    for (x, y) in xs.into_iter().zip(ys) {
        out.push(if square {
            Scalar::F(x.f() * x.f())
        } else {
            apply(op, k, x, y)?
        });
    }
    let rk = match op {
        Op::Lt | Op::Le | Op::Gt | Op::Ge | Op::Eq | Op::Ne => Kind::Bool,
        Op::Div | Op::Atan2 => Kind::Float,
        _ => out.first().map(|s| s.kind()).unwrap_or(if k == Kind::Bool && !matches!(op, Op::Add | Op::Mul | Op::And | Op::Or | Op::Xor | Op::Max | Op::Min) { Kind::Int } else { k }),
    };
    let mut r = NdArray::from_data(Data::from_scalars(rk, out.into_iter()), shape);
    r.scalar = r.shape.is_empty();
    Ok(r)
}

/// An element function of one array.
pub(crate) fn map(a: &NdArray, out: Kind, f: impl Fn(Scalar) -> Scalar) -> NdArray {
    let mut r = NdArray::from_data(Data::from_scalars(out, a.scalars().into_iter().map(f)), a.shape.clone());
    r.scalar = r.shape.is_empty();
    r
}

/// A float function (`np.sqrt`, `np.exp`, ...): ints and bools become floats.
pub(crate) fn map_f(a: &NdArray, f: impl Fn(f64) -> f64) -> NdArray {
    map(a, Kind::Float, |x| Scalar::F(f(x.f())))
}

pub(crate) fn axis_index(axis: i64, nd: usize) -> Result<usize, PyErr> {
    let n = nd as i64;
    let a = if axis < 0 { axis + n } else { axis };
    if a < 0 || a >= n {
        return Err(PyErr::new(
            &["AxisError", "ValueError", "IndexError", "Exception", "BaseException"],
            format!("axis {} is out of bounds for array of dimension {}", axis, nd),
        ));
    }
    Ok(a as usize)
}

pub(crate) fn expand_dims(a: &NdArray, ax: usize) -> NdArray {
    let mut r = a.clone();
    r.shape.insert(ax, 1);
    r.strides.insert(ax, 0);
    r.scalar = false;
    r
}

pub(crate) fn at_least_2d(a: &NdArray) -> NdArray {
    let mut r = a.clone();
    while r.shape.len() < 2 {
        r = expand_dims(&r, 0);
    }
    r
}

pub(crate) fn where_(cond: &NdArray, x: impl Operand, y: impl Operand) -> Result<NdArray, PyErr> {
    let (x, y) = (x.arg(), y.arg());
    let shape = broadcast_shapes(&broadcast_shapes(&cond.shape, x.shape())?, y.shape())?;
    let k = promote(&x, &y);
    let c = broadcast_items(&Arg::A(cond.clone()), &shape);
    let xs = broadcast_items(&x, &shape);
    let ys = broadcast_items(&y, &shape);
    let items = c.into_iter().zip(xs.into_iter().zip(ys)).map(|(c, (a, b))| if c.b() { a } else { b }.cast(k));
    Ok(NdArray::from_data(Data::from_scalars(k, items), shape))
}

// ------------------------------------------------------------ indexing

/// What an index selects: a view, or (with index arrays) some elements.
enum Target {
    View(NdArray),
    Gather { shape: Vec<usize>, offsets: Vec<usize>, data: NdArray },
}

fn norm_index(i: i64, d: usize, axis: usize) -> Result<usize, PyErr> {
    let j = if i < 0 { i + d as i64 } else { i };
    if j < 0 || j >= d as i64 {
        return Err(ierr(format!("index {} is out of bounds for axis {} with size {}", i, axis, d)));
    }
    Ok(j as usize)
}

/// Python's `slice.indices(d)`: (start, step, count).
fn slice_range(a: Option<i64>, b: Option<i64>, c: Option<i64>, d: usize) -> Result<(i64, i64, usize), PyErr> {
    let step = c.unwrap_or(1);
    if step == 0 {
        return Err(verr("slice step cannot be zero"));
    }
    let n = d as i64;
    let (lo, hi) = if step > 0 { (0, n) } else { (-1, n - 1) };
    let fix = |x: Option<i64>, dflt: i64| -> i64 {
        match x {
            None => dflt,
            Some(x) if x < 0 => (x + n).max(lo),
            Some(x) => x.min(hi),
        }
    };
    let start = fix(a, if step > 0 { 0 } else { n - 1 });
    let stop = fix(b, if step > 0 { n } else { -1 });
    let count = if step > 0 {
        if stop > start {
            (stop - start + step - 1) / step
        } else {
            0
        }
    } else if start > stop {
        (start - stop - step - 1) / (-step)
    } else {
        0
    };
    Ok((start, step, count as usize))
}

/// An index array part: integer indices with their shape.
struct IdxArr {
    shape: Vec<usize>,
    vals: Vec<i64>,
}

fn resolve(a: &NdArray, parts: &[Ix]) -> Result<Target, PyErr> {
    // expand masks to integer arrays (one per mask dimension), find the axes used
    enum P {
        Basic(Ix),
        Arr(IdxArr),
    }
    // a list of bools is a 1-d mask array
    let parts: Vec<Ix> = parts
        .iter()
        .map(|p| match p {
            Ix::Bools(m) => Ix::obj(NdArray::from_data(Data::Bool(m.clone()), vec![m.len()])),
            p => p.clone(),
        })
        .collect();
    let mut ps: Vec<P> = vec![];
    for p in &parts {
        match p {
            Ix::Ints(v) => ps.push(P::Arr(IdxArr { shape: vec![v.len()], vals: v.clone() })),
            Ix::Obj(o) => {
                let Some(x) = o.downcast_ref::<NdArray>() else {
                    return Err(ierr("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices"));
                };
                match x.kind() {
                    Kind::Int if x.shape.is_empty() => ps.push(P::Basic(Ix::Int(x.first().i()))),
                    Kind::Int => ps.push(P::Arr(IdxArr { shape: x.shape.clone(), vals: x.to_i64() })),
                    Kind::Bool if x.shape.is_empty() => {
                        return Err(ierr("ruai: a 0-d boolean index is not supported"));
                    }
                    Kind::Bool => {
                        // a k-d mask: k index arrays of its true positions
                        let items = x.scalars();
                        let st = c_strides(&x.shape);
                        let trues: Vec<usize> = (0..items.len()).filter(|&i| items[i].b()).collect();
                        let mask_axis = ps.iter().map(|p| if let P::Basic(Ix::NewAxis) = p { 0 } else { 1 }).sum::<usize>();
                        for (k, &stride) in st.iter().enumerate() {
                            let want = x.shape[k];
                            let have = a.shape.get(mask_axis + k).copied();
                            if have != Some(want) {
                                return Err(ierr(format!(
                                    "boolean index did not match indexed array along axis {}; size of axis is {} but size of corresponding boolean axis is {}",
                                    mask_axis + k,
                                    have.unwrap_or(0),
                                    want
                                )));
                            }
                            ps.push(P::Arr(IdxArr {
                                shape: vec![trues.len()],
                                vals: trues.iter().map(|&t| ((t as isize / stride) as usize % x.shape[k]) as i64).collect(),
                            }));
                        }
                    }
                    Kind::Float => {
                        return Err(ierr("arrays used as indices must be of integer (or boolean) type"));
                    }
                }
            }
            p => ps.push(P::Basic(p.clone())),
        }
    }
    let used = ps.iter().filter(|p| !matches!(p, P::Basic(Ix::NewAxis) | P::Basic(Ix::Ellipsis))).count();
    let nd = a.shape.len();
    if used > nd {
        return Err(ierr(format!(
            "too many indices for array: array is {}-dimensional, but {} were indexed",
            nd, used
        )));
    }
    if ps.iter().filter(|p| matches!(p, P::Basic(Ix::Ellipsis))).count() > 1 {
        return Err(ierr("an index can only have a single ellipsis ('...')"));
    }
    // the basic view; array parts keep their axis (a full slice for now)
    let mut shape = vec![];
    let mut strides = vec![];
    let mut offset = a.offset;
    let mut arrays: Vec<(usize, IdxArr)> = vec![];
    let mut axis = 0usize;
    for p in ps {
        match p {
            P::Basic(Ix::Ellipsis) => {
                for _ in 0..nd - used {
                    shape.push(a.shape[axis]);
                    strides.push(a.strides[axis]);
                    axis += 1;
                }
            }
            P::Basic(Ix::NewAxis) => {
                shape.push(1);
                strides.push(0);
            }
            P::Basic(Ix::Int(i)) => {
                let j = norm_index(i, a.shape[axis], axis)?;
                offset += j as isize * a.strides[axis];
                axis += 1;
            }
            P::Basic(Ix::Slice(s0, s1, s2)) => {
                let (start, step, count) = slice_range(s0, s1, s2, a.shape[axis])?;
                if count > 0 {
                    offset += start as isize * a.strides[axis];
                }
                shape.push(count);
                strides.push(a.strides[axis] * step as isize);
                axis += 1;
            }
            P::Arr(ix) => {
                let d = a.shape[axis];
                for &v in &ix.vals {
                    norm_index(v, d, axis)?;
                }
                arrays.push((shape.len(), ix));
                shape.push(d);
                strides.push(a.strides[axis]);
                axis += 1;
            }
            P::Basic(_) => unreachable!(),
        }
    }
    while axis < nd {
        shape.push(a.shape[axis]);
        strides.push(a.strides[axis]);
        axis += 1;
    }
    let view = NdArray { data: a.data.clone(), shape, strides, offset, scalar: false };
    if arrays.is_empty() {
        return Ok(Target::View(view));
    }
    // index arrays: broadcast together, placed where the first one is when
    // they are adjacent, else first
    let mut bshape: Vec<usize> = vec![];
    for (_, ix) in &arrays {
        bshape = broadcast_shapes(&bshape, &ix.shape).map_err(|_| {
            ierr(format!(
                "shape mismatch: indexing arrays could not be broadcast together with shapes {}",
                arrays.iter().map(|(_, x)| shape_err(&x.shape)).collect::<Vec<_>>().join(" ") + " "
            ))
        })?;
    }
    let axes: Vec<usize> = arrays.iter().map(|(ax, _)| *ax).collect();
    let adjacent = axes.windows(2).all(|w| w[1] == w[0] + 1);
    let rest: Vec<usize> = (0..view.shape.len()).filter(|ax| !axes.contains(ax)).collect();
    let insert_at = if adjacent { rest.iter().filter(|&&r| r < axes[0]).count() } else { 0 };
    let mut out_shape: Vec<usize> = rest.iter().map(|&r| view.shape[r]).collect();
    for (i, &d) in bshape.iter().enumerate() {
        out_shape.insert(insert_at + i, d);
    }
    // each index array broadcast to bshape
    let idx_items: Vec<Vec<i64>> = arrays
        .iter()
        .map(|(ax, ix)| {
            let arr = NdArray::from_data(Data::Int(ix.vals.clone()), ix.shape.clone());
            let d = view.shape[*ax] as i64;
            broadcast_items(&Arg::A(arr), &bshape).into_iter().map(|s| if s.i() < 0 { s.i() + d } else { s.i() }).collect()
        })
        .collect();
    let n: usize = out_shape.iter().product();
    let mut offsets = Vec::with_capacity(n);
    let nb = bshape.len();
    let bst = c_strides(&bshape);
    let mut idx = vec![0usize; out_shape.len()];
    for _ in 0..n {
        let mut off = view.offset;
        // the position in the broadcast index arrays
        let mut flat = 0usize;
        for i in 0..nb {
            flat += idx[insert_at + i] * bst[i] as usize;
        }
        for (k, (ax, _)) in arrays.iter().enumerate() {
            off += idx_items[k][flat] as isize * view.strides[*ax];
        }
        let mut ri = 0;
        for (pos, _) in out_shape.iter().enumerate() {
            if pos >= insert_at && pos < insert_at + nb {
                continue;
            }
            off += idx[pos] as isize * view.strides[rest[ri]];
            ri += 1;
        }
        offsets.push(off as usize);
        for k in (0..out_shape.len()).rev() {
            idx[k] += 1;
            if idx[k] < out_shape[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    Ok(Target::Gather { shape: out_shape, offsets, data: view })
}

pub(crate) fn get(a: &NdArray, parts: &[Ix]) -> Result<NdArray, PyErr> {
    match resolve(a, parts)? {
        Target::View(v) if v.shape.is_empty() => Ok(NdArray::from_scalar(v.first())),
        Target::View(v) => Ok(v),
        Target::Gather { shape, offsets, data } => {
            let d = data.data.borrow();
            let items: Vec<Scalar> = offsets.iter().map(|&o| d.get(o)).collect();
            Ok(NdArray::from_data(Data::from_scalars(d.kind(), items.into_iter()), shape))
        }
    }
}

pub(crate) fn set(a: &NdArray, parts: &[Ix], value: impl Operand) -> Result<(), PyErr> {
    let (shape, offsets) = match resolve(a, parts)? {
        Target::View(v) => (v.shape.clone(), v.offsets()),
        Target::Gather { shape, offsets, .. } => (shape, offsets),
    };
    let v = value.arg();
    let vs = v.shape().to_vec();
    // the value broadcasts to the target (leading 1s of the value may drop)
    let vs_trim: Vec<usize> = {
        let mut s = vs.clone();
        while s.len() > shape.len() && s[0] == 1 {
            s.remove(0);
        }
        s
    };
    let ok = vs_trim.len() <= shape.len()
        && vs_trim.iter().rev().zip(shape.iter().rev()).all(|(&x, &y)| x == y || x == 1);
    if !ok {
        return Err(verr(format!(
            "could not broadcast input array from shape {} into shape {}",
            shape_err(&vs),
            shape_err(&shape)
        )));
    }
    let items = match &v {
        Arg::A(x) => {
            let mut x = x.clone();
            x.shape = vs_trim.clone();
            x.strides = x.strides[x.strides.len() - vs_trim.len()..].to_vec();
            broadcast_items(&Arg::A(x), &shape)
        }
        s => broadcast_items(s, &shape),
    };
    let mut d = a.data.borrow_mut();
    for (o, x) in offsets.into_iter().zip(items) {
        if d.kind() == Kind::Int {
            if let Scalar::F(f) = x {
                if !f.is_finite() {
                    return Err(verr(format!("cannot convert float {} to integer", if f.is_nan() { "NaN" } else { "infinity" })));
                }
            }
        }
        d.set(o, x);
    }
    Ok(())
}

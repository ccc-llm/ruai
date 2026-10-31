//! The operators of `ndarray` and the element functions of `numpy`.

use super::ops::{binary, map, map_f, Arg, Op, Operand};
use super::{terr, Kind, NdArray, Scalar};
use crate::pyrt::err::PyErr;

macro_rules! binops {
    ($($name:ident $rname:ident $op:ident),* $(,)?) => {
        impl NdArray {
            $(
                #[doc = concat!("`a ", stringify!($op), " b`")]
                pub fn $name(&self, o: impl Operand) -> Result<NdArray, PyErr> {
                    binary(Op::$op, &Arg::A(self.clone()), &o.arg())
                }
                #[doc = concat!("`b ", stringify!($op), " a` (reflected)")]
                pub fn $rname(&self, o: impl Operand) -> Result<NdArray, PyErr> {
                    binary(Op::$op, &o.arg(), &Arg::A(self.clone()))
                }
            )*
        }
    };
}

binops! {
    __add__ __radd__ Add,
    __sub__ __rsub__ Sub,
    __mul__ __rmul__ Mul,
    __truediv__ __rtruediv__ Div,
    __floordiv__ __rfloordiv__ FloorDiv,
    __mod__ __rmod__ Mod,
    __pow__ __rpow__ Pow,
    __and__ __rand__ And,
    __or__ __ror__ Or,
    __xor__ __rxor__ Xor,
}

macro_rules! cmpops {
    ($($name:ident $op:ident),* $(,)?) => {
        impl NdArray {
            $(
                #[doc = concat!("`a ", stringify!($op), " b`: a bool array")]
                pub fn $name(&self, o: impl Operand) -> Result<NdArray, PyErr> {
                    binary(Op::$op, &Arg::A(self.clone()), &o.arg())
                }
            )*
        }
    };
}

cmpops! { __lt__ Lt, __le__ Le, __gt__ Gt, __ge__ Ge, __eq__ Eq, __ne__ Ne }

impl NdArray {
    /// `-a`
    pub fn __neg__(&self) -> Result<NdArray, PyErr> {
        match self.kind() {
            Kind::Bool => Err(terr(
                "The numpy boolean negative, the `-` operator, is not supported, use the `~` operator or the logical_not function instead.",
            )),
            Kind::Int => Ok(map(self, Kind::Int, |x| Scalar::I(x.i().wrapping_neg()))),
            Kind::Float | Kind::F32 => Ok(map(self, self.kind(), |x| Scalar::F(-x.f()))),
        }
    }
    /// `+a`
    pub fn __pos__(&self) -> NdArray {
        map(self, self.kind(), |x| x)
    }
    /// `abs(a)`
    pub fn __abs__(&self) -> NdArray {
        absolute(self)
    }
    /// `~a`
    pub fn __invert__(&self) -> Result<NdArray, PyErr> {
        match self.kind() {
            Kind::Bool => Ok(map(self, Kind::Bool, |x| Scalar::B(!x.b()))),
            Kind::Int => Ok(map(self, Kind::Int, |x| Scalar::I(!x.i()))),
            Kind::Float | Kind::F32 => Err(terr(
                "ufunc 'invert' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''",
            )),
        }
    }
    /// `a @ b`
    pub fn __matmul__(&self, o: &NdArray) -> Result<NdArray, PyErr> {
        super::linalg::matmul(self, o)
    }
    /// `a.dot(b)`
    pub fn dot(&self, o: &NdArray) -> Result<NdArray, PyErr> {
        super::linalg::dot(self, o)
    }
    /// `a.clip(lo, hi)`
    pub fn clip(&self, lo: impl Operand, hi: impl Operand) -> Result<NdArray, PyErr> {
        clip(self, lo, hi)
    }
    /// `a.round(decimals=0)`
    pub fn round(&self, decimals: i64) -> NdArray {
        round(self, decimals)
    }
}

/// `np.abs(a)`, `np.absolute(a)`
pub fn absolute(a: &NdArray) -> NdArray {
    match a.kind() {
        Kind::Bool => map(a, Kind::Bool, |x| x),
        Kind::Int => map(a, Kind::Int, |x| Scalar::I(x.i().wrapping_abs())),
        Kind::Float | Kind::F32 => map_f(a, f64::abs),
    }
}

/// `np.sign(a)`
pub fn sign(a: &NdArray) -> NdArray {
    match a.kind() {
        Kind::Float | Kind::F32 => map_f(a, |x| if x.is_nan() { x } else if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }),
        k => map(a, k.max(Kind::Int), |x| Scalar::I(x.i().signum())),
    }
}

/// `np.floor`, `np.ceil`, `np.trunc`, `np.rint`: integers stay integers.
fn int_keeping(a: &NdArray, f: fn(f64) -> f64) -> NdArray {
    match a.kind() {
        Kind::Float | Kind::F32 => map_f(a, f),
        k => map(a, k, |x| x),
    }
}

/// `np.floor(a)`
pub fn floor(a: &NdArray) -> NdArray {
    int_keeping(a, f64::floor)
}
/// `np.ceil(a)`
pub fn ceil(a: &NdArray) -> NdArray {
    int_keeping(a, f64::ceil)
}
/// `np.trunc(a)`
pub fn trunc(a: &NdArray) -> NdArray {
    int_keeping(a, f64::trunc)
}
/// `np.rint(a)`
pub fn rint(a: &NdArray) -> NdArray {
    int_keeping(a, f64::round_ties_even)
}

/// `np.round(a, decimals=0)`: half to even, as `rint(a * 10**d) / 10**d`.
pub fn round(a: &NdArray, decimals: i64) -> NdArray {
    let d = decimals;
    match a.kind() {
        Kind::Float | Kind::F32 => map_f(a, move |x| {
            if d >= 0 {
                let p = 10f64.powi(d as i32);
                (x * p).round_ties_even() / p
            } else {
                let p = 10f64.powi(-d as i32);
                (x / p).round_ties_even() * p
            }
        }),
        k if d >= 0 => map(a, k, |x| x),
        _ => {
            let p = 10f64.powi(-d as i32);
            map(a, Kind::Int, move |x| Scalar::I(((x.i() as f64 / p).round_ties_even() * p) as i64))
        }
    }
}

macro_rules! float_fns {
    ($($name:ident $f:expr),* $(,)?) => {
        $(
            #[doc = concat!("`np.", stringify!($name), "(a)`")]
            pub fn $name(a: &NdArray) -> NdArray {
                map_f(a, $f)
            }
        )*
    };
}

float_fns! {
    sqrt f64::sqrt,
    exp f64::exp,
    log f64::ln,
    log2 f64::log2,
    log10 f64::log10,
    log1p f64::ln_1p,
    expm1 f64::exp_m1,
    sin f64::sin,
    cos f64::cos,
    tan f64::tan,
    arcsin f64::asin,
    arccos f64::acos,
    arctan f64::atan,
    sinh f64::sinh,
    cosh f64::cosh,
    tanh f64::tanh,
}

/// `np.square(a)`
pub fn square(a: &NdArray) -> NdArray {
    match a.kind() {
        Kind::Float | Kind::F32 => map_f(a, |x| x * x),
        k => map(a, k.max(Kind::Int), |x| Scalar::I(x.i().wrapping_mul(x.i()))),
    }
}

/// `np.isnan(a)`
pub fn isnan(a: &NdArray) -> NdArray {
    map(a, Kind::Bool, |x| Scalar::B(matches!(x, Scalar::F(f) if f.is_nan())))
}
/// `np.isinf(a)`
pub fn isinf(a: &NdArray) -> NdArray {
    map(a, Kind::Bool, |x| Scalar::B(matches!(x, Scalar::F(f) if f.is_infinite())))
}
/// `np.isfinite(a)`
pub fn isfinite(a: &NdArray) -> NdArray {
    map(a, Kind::Bool, |x| Scalar::B(!matches!(x, Scalar::F(f) if !f.is_finite())))
}
/// `np.logical_not(a)`
pub fn logical_not(a: &NdArray) -> NdArray {
    map(a, Kind::Bool, |x| Scalar::B(!x.b()))
}

/// `np.maximum(a, b)`
pub fn maximum(a: impl Operand, b: impl Operand) -> Result<NdArray, PyErr> {
    binary(Op::Max, &a.arg(), &b.arg())
}
/// `np.minimum(a, b)`
pub fn minimum(a: impl Operand, b: impl Operand) -> Result<NdArray, PyErr> {
    binary(Op::Min, &a.arg(), &b.arg())
}
/// `np.power(a, b)`
pub fn power(a: impl Operand, b: impl Operand) -> Result<NdArray, PyErr> {
    binary(Op::Pow, &a.arg(), &b.arg())
}
/// `np.arctan2(y, x)`
pub fn arctan2(y: impl Operand, x: impl Operand) -> Result<NdArray, PyErr> {
    binary(Op::Atan2, &y.arg(), &x.arg())
}
/// `np.clip(a, lo, hi)`
pub fn clip(a: &NdArray, lo: impl Operand, hi: impl Operand) -> Result<NdArray, PyErr> {
    let m = binary(Op::Max, &Arg::A(a.clone()), &lo.arg())?;
    binary(Op::Min, &Arg::A(m), &hi.arg())
}
/// `np.count_nonzero(a)`
pub fn count_nonzero(a: &NdArray) -> i64 {
    a.scalars().into_iter().filter(|x| x.b()).count() as i64
}
/// `np.allclose(a, b, rtol=1e-05, atol=1e-08)`
pub fn allclose(a: &NdArray, b: impl Operand, rtol: f64, atol: f64) -> Result<bool, PyErr> {
    let b = b.arg();
    let shape = super::ops::broadcast_shapes(&a.shape, match &b {
        Arg::A(x) => &x.shape,
        Arg::S(_) => &[],
    })?;
    let xs = super::ops::broadcast_items(&Arg::A(a.clone()), &shape);
    let ys = super::ops::broadcast_items(&b, &shape);
    Ok(xs.iter().zip(&ys).all(|(x, y)| {
        let (x, y) = (x.f(), y.f());
        if x.is_infinite() || y.is_infinite() {
            return x == y;
        }
        (x - y).abs() <= atol + rtol * y.abs()
    }))
}
/// `np.array_equal(a, b)`
pub fn array_equal(a: &NdArray, b: &NdArray) -> bool {
    a.shape == b.shape && a.scalars().iter().zip(b.scalars()).all(|(x, y)| x.f() == y.f())
}

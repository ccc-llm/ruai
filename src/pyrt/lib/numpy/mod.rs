//! `numpy` for py0: a subset of NumPy's `ndarray` with float64, int64 and bool
//! elements, declared by `src/py/lib/numpy.pyi`. Results match NumPy: values,
//! dtypes (NumPy 2 promotion), printing (`str` / `repr` as `arrayprint`),
//! errors, and the order of floating-point sums (NumPy's pairwise summation).
//!
//! An array is a view of a shared buffer (shape, strides, offset), so slices
//! share memory with their base as in NumPy, and so do copies of a py0 name
//! (`b = a` is the same array). Reductions and integer indexing to a single
//! element give a 0-d array marked as a scalar (`np.float64(2.5)`).

mod api;
mod format;
pub mod linalg;
mod ops;
pub mod random;
mod reduce;

use crate::pyrt::err::{PyErr, INDEX_ERROR, TYPE_ERROR, VALUE_ERROR};
use std::cell::RefCell;
use std::rc::Rc;

pub use api::*;
pub use ops::Operand;
pub use reduce::unique;

/// Element type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Kind {
    Bool,
    Int,
    F32,
    #[default]
    Float,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Bool => "bool",
            Kind::Int => "int64",
            Kind::F32 => "float32",
            Kind::Float => "float64",
        }
    }
    fn parse(s: &str) -> Result<Kind, PyErr> {
        Ok(match s {
            "bool" | "bool_" | "?" => Kind::Bool,
            "int" | "int64" | "i8" | "int_" | "long" => Kind::Int,
            "float" | "float64" | "f8" | "double" => Kind::Float,
            "float32" | "f4" | "single" => Kind::F32,
            _ => return Err(PyErr::new(TYPE_ERROR, format!("data type '{}' not understood", s))),
        })
    }
}

/// One element.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    B(bool),
    I(i64),
    F(f64),
}

impl Scalar {
    fn kind(self) -> Kind {
        match self {
            Scalar::B(_) => Kind::Bool,
            Scalar::I(_) => Kind::Int,
            Scalar::F(_) => Kind::Float,
        }
    }
    fn f(self) -> f64 {
        match self {
            Scalar::B(b) => b as i64 as f64,
            Scalar::I(i) => i as f64,
            Scalar::F(x) => x,
        }
    }
    fn i(self) -> i64 {
        match self {
            Scalar::B(b) => b as i64,
            Scalar::I(i) => i,
            // NumPy's float -> int64 cast: truncation (out of range: i64::MIN)
            Scalar::F(x) => {
                if (-9.223372036854776e18..9.223372036854776e18).contains(&x) {
                    x as i64
                } else {
                    i64::MIN
                }
            }
        }
    }
    fn b(self) -> bool {
        match self {
            Scalar::B(b) => b,
            Scalar::I(i) => i != 0,
            Scalar::F(x) => x != 0.0,
        }
    }
    fn cast(self, k: Kind) -> Scalar {
        match k {
            Kind::Bool => Scalar::B(self.b()),
            Kind::Int => Scalar::I(self.i()),
            Kind::F32 => Scalar::F(self.f() as f32 as f64),
            Kind::Float => Scalar::F(self.f()),
        }
    }
}

/// The elements of a buffer.
#[derive(Clone, Debug)]
pub enum Data {
    Bool(Vec<bool>),
    Int(Vec<i64>),
    F32(Vec<f32>),
    Float(Vec<f64>),
}

impl Data {
    fn kind(&self) -> Kind {
        match self {
            Data::Bool(_) => Kind::Bool,
            Data::Int(_) => Kind::Int,
            Data::F32(_) => Kind::F32,
            Data::Float(_) => Kind::Float,
        }
    }
    fn len(&self) -> usize {
        match self {
            Data::Bool(v) => v.len(),
            Data::Int(v) => v.len(),
            Data::F32(v) => v.len(),
            Data::Float(v) => v.len(),
        }
    }
    fn get(&self, i: usize) -> Scalar {
        match self {
            Data::Bool(v) => Scalar::B(v[i]),
            Data::Int(v) => Scalar::I(v[i]),
            Data::F32(v) => Scalar::F(v[i] as f64),
            Data::Float(v) => Scalar::F(v[i]),
        }
    }
    /// Store `x` (cast to this buffer's type).
    fn set(&mut self, i: usize, x: Scalar) {
        match self {
            Data::Bool(v) => v[i] = x.b(),
            Data::Int(v) => v[i] = x.i(),
            Data::F32(v) => v[i] = x.f() as f32,
            Data::Float(v) => v[i] = x.f(),
        }
    }
    fn from_scalars(k: Kind, xs: impl Iterator<Item = Scalar>) -> Data {
        match k {
            Kind::Bool => Data::Bool(xs.map(|x| x.b()).collect()),
            Kind::Int => Data::Int(xs.map(|x| x.i()).collect()),
            Kind::F32 => Data::F32(xs.map(|x| x.f() as f32).collect()),
            Kind::Float => Data::Float(xs.map(|x| x.f()).collect()),
        }
    }
}

/// `numpy.ndarray`
#[derive(Clone)]
pub struct NdArray {
    data: Rc<RefCell<Data>>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
    /// a NumPy scalar (`np.float64`), the result of a reduction or of indexing
    /// one element: printed by repr as `np.float64(1.5)`
    scalar: bool,
}

impl std::fmt::Debug for NdArray {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&self.__repr__())
    }
}

impl PartialEq for NdArray {
    fn eq(&self, o: &NdArray) -> bool {
        Rc::ptr_eq(&self.data, &o.data) && self.shape == o.shape && self.strides == o.strides && self.offset == o.offset
    }
}

impl Default for NdArray {
    fn default() -> NdArray {
        NdArray::from_data(Data::Float(vec![]), vec![0])
    }
}

fn c_strides(shape: &[usize]) -> Vec<isize> {
    let mut s = vec![0isize; shape.len()];
    let mut acc = 1isize;
    for i in (0..shape.len()).rev() {
        s[i] = acc;
        acc *= shape[i].max(1) as isize;
    }
    s
}

/// A shape as NumPy writes it in error messages: `(2,3)`, `(3,)`.
pub(crate) fn shape_err(shape: &[usize]) -> String {
    shape_str(shape).replace(", ", ",")
}

pub(crate) fn shape_str(shape: &[usize]) -> String {
    match shape.len() {
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    }
}

fn verr(msg: impl Into<String>) -> PyErr {
    PyErr::new(VALUE_ERROR, msg.into())
}

fn ierr(msg: impl Into<String>) -> PyErr {
    PyErr::new(INDEX_ERROR, msg.into())
}

fn terr(msg: impl Into<String>) -> PyErr {
    PyErr::new(TYPE_ERROR, msg.into())
}

impl NdArray {
    /// A new C-contiguous array.
    pub(crate) fn from_data(data: Data, shape: Vec<usize>) -> NdArray {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>());
        let strides = c_strides(&shape);
        NdArray { data: Rc::new(RefCell::new(data)), shape, strides, offset: 0, scalar: false }
    }
    pub(crate) fn from_scalar(x: Scalar) -> NdArray {
        NdArray::scalar_of(x, x.kind())
    }
    /// A NumPy scalar of kind `k`.
    pub(crate) fn scalar_of(x: Scalar, k: Kind) -> NdArray {
        let mut a = NdArray::from_data(Data::from_scalars(k, std::iter::once(x)), vec![]);
        a.scalar = true;
        a
    }
    pub(crate) fn floats(v: Vec<f64>, shape: Vec<usize>) -> NdArray {
        NdArray::from_data(Data::Float(v), shape)
    }
    /// The dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.shape
    }
    /// A new int64 array.
    pub fn ints(v: Vec<i64>, shape: Vec<usize>) -> NdArray {
        NdArray::from_data(Data::Int(v), shape)
    }
    pub(crate) fn kind(&self) -> Kind {
        self.data.borrow().kind()
    }
    pub(crate) fn len(&self) -> usize {
        self.shape.iter().product()
    }
    /// Buffer offsets of the elements in C order.
    pub(crate) fn offsets(&self) -> Vec<usize> {
        let n = self.len();
        let mut out = Vec::with_capacity(n);
        if n == 0 {
            return out;
        }
        let nd = self.shape.len();
        let mut idx = vec![0usize; nd];
        let mut off = self.offset;
        loop {
            out.push(off as usize);
            let mut k = nd;
            loop {
                if k == 0 {
                    return out;
                }
                k -= 1;
                idx[k] += 1;
                off += self.strides[k];
                if idx[k] < self.shape[k] {
                    break;
                }
                off -= self.strides[k] * self.shape[k] as isize;
                idx[k] = 0;
            }
        }
    }
    /// The elements in C order.
    pub(crate) fn scalars(&self) -> Vec<Scalar> {
        let d = self.data.borrow();
        self.offsets().into_iter().map(|o| d.get(o)).collect()
    }
    pub(crate) fn to_f64(&self) -> Vec<f64> {
        self.scalars().into_iter().map(Scalar::f).collect()
    }
    pub(crate) fn to_i64(&self) -> Vec<i64> {
        self.scalars().into_iter().map(Scalar::i).collect()
    }
    /// A C-contiguous copy (not a scalar).
    pub(crate) fn contiguous(&self) -> NdArray {
        NdArray::from_data(Data::from_scalars(self.kind(), self.scalars().into_iter()), self.shape.clone())
    }
    fn first(&self) -> Scalar {
        self.data.borrow().get(self.offset as usize)
    }

    // ------------------------------------------------------------ attributes

    /// `a.shape`
    pub fn shape(&self) -> Shape {
        Shape(self.shape.iter().map(|&d| d as i64).collect())
    }
    /// `a.ndim`
    pub fn ndim(&self) -> i64 {
        self.shape.len() as i64
    }
    /// `a.size`
    pub fn size(&self) -> i64 {
        self.len() as i64
    }
    /// `a.dtype`
    pub fn dtype(&self) -> DType {
        DType(self.kind())
    }
    /// `a.T`
    #[allow(non_snake_case)]
    pub fn T(&self) -> NdArray {
        self.transpose()
    }
    /// `len(a)`
    pub fn __len__(&self) -> Result<i64, PyErr> {
        match self.shape.first() {
            Some(&d) => Ok(d as i64),
            None => Err(terr("len() of unsized object")),
        }
    }
    /// `for x in a`: the items along the first axis (views; NumPy scalars for 1-d)
    pub fn __iter__(&self) -> Result<Vec<NdArray>, PyErr> {
        let Some(&n) = self.shape.first() else {
            return Err(terr("iteration over a 0-d array"));
        };
        (0..n as i64).map(|i| ops::get(self, &[crate::pyrt::Ix::Int(i)])).collect()
    }
    /// `str(a)`
    pub fn __str__(&self) -> String {
        format::array_str(self)
    }
    /// `repr(a)`
    pub fn __repr__(&self) -> String {
        format::array_repr(self)
    }
    fn one(&self, what: &str) -> Result<Scalar, PyErr> {
        if self.len() != 1 {
            return Err(terr(format!("only length-1 arrays can be converted to Python {}", what)));
        }
        Ok(self.scalars()[0])
    }
    /// `float(a)`
    pub fn __float__(&self) -> Result<f64, PyErr> {
        Ok(self.one("scalars")?.f())
    }
    /// `int(a)`
    pub fn __int__(&self) -> Result<i64, PyErr> {
        let x = self.one("scalars")?;
        if let Scalar::F(f) = x {
            if f.is_nan() {
                return Err(verr("cannot convert float NaN to integer"));
            }
            if f.is_infinite() {
                return Err(PyErr::new(crate::pyrt::err::OVERFLOW_ERROR, "cannot convert float infinity to integer"));
            }
        }
        Ok(x.i())
    }
    /// `bool(a)`, `if a:`
    pub fn __bool__(&self) -> Result<bool, PyErr> {
        match self.len() {
            1 => Ok(self.scalars()[0].b()),
            0 => Err(verr("The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.")),
            _ => Err(verr("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()")),
        }
    }
    /// `a.item()`: the one element as a float (int and bool arrays convert).
    pub fn item(&self) -> Result<f64, PyErr> {
        if self.len() != 1 {
            return Err(verr("can only convert an array of size 1 to a Python scalar"));
        }
        Ok(self.scalars()[0].f())
    }
    /// `a.tolist()` of a 1-d array, as floats.
    pub fn tolist(&self) -> Result<Vec<f64>, PyErr> {
        if self.shape.len() != 1 {
            return Err(verr("ruai: tolist() is for 1-d arrays (a list of floats)"));
        }
        Ok(self.to_f64())
    }

    // ------------------------------------------------------------ shape

    /// `a.copy()`
    pub fn copy(&self) -> NdArray {
        let mut c = self.contiguous();
        c.scalar = self.scalar;
        c
    }
    /// `a.astype(dtype)`
    pub fn astype(&self, dtype: &str) -> Result<NdArray, PyErr> {
        let k = Kind::parse(dtype)?;
        Ok(NdArray::from_data(Data::from_scalars(k, self.scalars().into_iter().map(|x| x.cast(k))), self.shape.clone()))
    }
    /// `a.reshape(shape)`
    pub fn reshape(&self, shape: impl ShapeArg) -> Result<NdArray, PyErr> {
        let want = shape.dims();
        let n = self.len();
        let unknown: Vec<usize> = (0..want.len()).filter(|&i| want[i] < 0).collect();
        if unknown.len() > 1 {
            return Err(verr("can only specify one unknown dimension"));
        }
        let known: i64 = want.iter().filter(|&&d| d >= 0).product();
        let mut dims: Vec<usize> = want.iter().map(|&d| d.max(0) as usize).collect();
        if let Some(&u) = unknown.first() {
            if known == 0 || n as i64 % known != 0 {
                return Err(verr(format!("cannot reshape array of size {} into shape {}", n, shape_text_sep(&want, ","))));
            }
            dims[u] = (n as i64 / known) as usize;
        }
        if dims.iter().product::<usize>() != n {
            return Err(verr(format!("cannot reshape array of size {} into shape {}", n, shape_text_sep(&want, ","))));
        }
        let c = if self.is_contiguous() { self.clone() } else { self.contiguous() };
        Ok(NdArray { data: c.data, strides: c_strides(&dims), shape: dims, offset: c.offset, scalar: false })
    }
    fn is_contiguous(&self) -> bool {
        self.strides == c_strides(&self.shape)
    }
    /// `a.flatten()`: a copy.
    pub fn flatten(&self) -> NdArray {
        let c = self.contiguous();
        let n = c.len();
        NdArray { strides: vec![1], shape: vec![n], ..c }
    }
    /// `a.ravel()`: a view when possible.
    pub fn ravel(&self) -> NdArray {
        self.reshape(-1i64).unwrap_or_else(|_| self.flatten())
    }
    /// `a.transpose()`: the axes reversed (a view).
    pub fn transpose(&self) -> NdArray {
        let mut t = self.clone();
        t.shape.reverse();
        t.strides.reverse();
        t.scalar = false;
        t
    }
    /// `a.fill(x)`
    pub fn fill(&self, x: impl Operand) {
        let x = x.scalar_value().unwrap_or(Scalar::F(0.0));
        let offs = self.offsets();
        let mut d = self.data.borrow_mut();
        for o in offs {
            d.set(o, x);
        }
    }

    // ------------------------------------------------------------ indexing

    /// `a[index]`
    pub fn __getitem__(&self, index: &crate::pyrt::Index) -> Result<NdArray, PyErr> {
        ops::get(self, &index.0)
    }
    /// `a[index] = value`
    pub fn __setitem__(&self, index: &crate::pyrt::Index, value: impl Operand) -> Result<(), PyErr> {
        ops::set(self, &index.0, value)
    }
}

fn shape_text(d: &[i64]) -> String {
    shape_text_sep(d, ", ")
}

fn shape_text_sep(d: &[i64], sep: &str) -> String {
    match d.len() {
        1 => format!("({},)", d[0]),
        _ => format!("({})", d.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(sep)),
    }
}

/// `a.shape`: printed like a tuple.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Shape(pub Vec<i64>);

impl Shape {
    /// `shape[i]`
    pub fn __getitem__(&self, i: i64) -> Result<i64, PyErr> {
        let n = self.0.len() as i64;
        let j = if i < 0 { i + n } else { i };
        if j < 0 || j >= n {
            return Err(ierr("tuple index out of range"));
        }
        Ok(self.0[j as usize])
    }
    /// `len(shape)`
    pub fn __len__(&self) -> i64 {
        self.0.len() as i64
    }
    /// `str(shape)`
    pub fn __str__(&self) -> String {
        shape_text(&self.0)
    }
    /// `repr(shape)`
    pub fn __repr__(&self) -> String {
        shape_text(&self.0)
    }
    /// `shape == other`
    pub fn __eq__(&self, o: &Shape) -> bool {
        self.0 == o.0
    }
    /// The dimensions as a list.
    pub fn tolist(&self) -> Vec<i64> {
        self.0.clone()
    }
}

/// `a.dtype`
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct DType(Kind);

impl DType {
    /// `str(dtype)`: `float64`, `int64`, `bool`
    pub fn __str__(&self) -> String {
        self.0.name().to_string()
    }
    /// `repr(dtype)`
    pub fn __repr__(&self) -> String {
        format!("dtype('{}')", self.0.name())
    }
    /// `dtype.name`
    pub fn name(&self) -> String {
        self.0.name().to_string()
    }
    /// `dtype == other`
    pub fn __eq__(&self, o: &DType) -> bool {
        self.0 == o.0
    }
}

/// A shape argument: `3`, `(2, 3)`, `(2, 3, 4)` or a list.
pub trait ShapeArg {
    fn dims(&self) -> Vec<i64>;
}

impl ShapeArg for i64 {
    fn dims(&self) -> Vec<i64> {
        vec![*self]
    }
}
impl ShapeArg for (i64,) {
    fn dims(&self) -> Vec<i64> {
        vec![self.0]
    }
}
impl ShapeArg for (i64, i64) {
    fn dims(&self) -> Vec<i64> {
        vec![self.0, self.1]
    }
}
impl ShapeArg for (i64, i64, i64) {
    fn dims(&self) -> Vec<i64> {
        vec![self.0, self.1, self.2]
    }
}
impl<T: ShapeArg + ?Sized> ShapeArg for &T {
    fn dims(&self) -> Vec<i64> {
        (**self).dims()
    }
}
impl ShapeArg for [i64] {
    fn dims(&self) -> Vec<i64> {
        self.to_vec()
    }
}
impl ShapeArg for Vec<i64> {
    fn dims(&self) -> Vec<i64> {
        self.clone()
    }
}

fn shape_of(s: &impl ShapeArg) -> Result<Vec<usize>, PyErr> {
    s.dims()
        .into_iter()
        .map(|d| if d < 0 { Err(verr("negative dimensions are not allowed")) } else { Ok(d as usize) })
        .collect()
}

// ------------------------------------------------------------ creation

/// Python values that make an array: (nested) lists of bool, int or float, or an array.
pub trait IntoArray {
    fn shape_and_items(&self, shape: &mut Vec<usize>, items: &mut Vec<Scalar>, depth: usize) -> Result<(), PyErr>;
    fn as_array(&self) -> Option<NdArray> {
        None
    }
}

/// An element of a list given to `array`.
pub trait Elem: Copy {
    fn scalar(self) -> Scalar;
}
impl Elem for bool {
    fn scalar(self) -> Scalar {
        Scalar::B(self)
    }
}
impl Elem for i64 {
    fn scalar(self) -> Scalar {
        Scalar::I(self)
    }
}
impl Elem for f64 {
    fn scalar(self) -> Scalar {
        Scalar::F(self)
    }
}

fn record(shape: &mut Vec<usize>, depth: usize, n: usize) -> Result<(), PyErr> {
    if shape.len() == depth {
        shape.push(n);
    } else if shape[depth] != n {
        return Err(verr(format!(
            "setting an array element with a sequence. The requested array has an inhomogeneous shape after {} dimensions. The detected shape was {} + inhomogeneous part.",
            depth,
            shape_str(&shape[..depth])
        )));
    }
    Ok(())
}

impl<T: Elem> IntoArray for [T] {
    fn shape_and_items(&self, shape: &mut Vec<usize>, items: &mut Vec<Scalar>, depth: usize) -> Result<(), PyErr> {
        record(shape, depth, self.len())?;
        items.extend(self.iter().map(|x| x.scalar()));
        Ok(())
    }
}
impl<T: Elem> IntoArray for [Vec<T>] {
    fn shape_and_items(&self, shape: &mut Vec<usize>, items: &mut Vec<Scalar>, depth: usize) -> Result<(), PyErr> {
        record(shape, depth, self.len())?;
        for r in self {
            r.as_slice().shape_and_items(shape, items, depth + 1)?;
        }
        Ok(())
    }
}
impl<T: Elem> IntoArray for [Vec<Vec<T>>] {
    fn shape_and_items(&self, shape: &mut Vec<usize>, items: &mut Vec<Scalar>, depth: usize) -> Result<(), PyErr> {
        record(shape, depth, self.len())?;
        for r in self {
            r.as_slice().shape_and_items(shape, items, depth + 1)?;
        }
        Ok(())
    }
}
impl<T: IntoArray + ?Sized> IntoArray for &T {
    fn shape_and_items(&self, shape: &mut Vec<usize>, items: &mut Vec<Scalar>, depth: usize) -> Result<(), PyErr> {
        (**self).shape_and_items(shape, items, depth)
    }
    fn as_array(&self) -> Option<NdArray> {
        (**self).as_array()
    }
}
impl<T> IntoArray for Vec<T>
where
    [T]: IntoArray,
{
    fn shape_and_items(&self, shape: &mut Vec<usize>, items: &mut Vec<Scalar>, depth: usize) -> Result<(), PyErr> {
        self.as_slice().shape_and_items(shape, items, depth)
    }
}
macro_rules! scalar_into_array {
    ($($t:ty),*) => {
        $(
            impl IntoArray for $t {
                fn shape_and_items(&self, _shape: &mut Vec<usize>, items: &mut Vec<Scalar>, _depth: usize) -> Result<(), PyErr> {
                    items.push(self.scalar());
                    Ok(())
                }
            }
        )*
    };
}
scalar_into_array!(bool, i64, f64);

impl IntoArray for NdArray {
    fn shape_and_items(&self, shape: &mut Vec<usize>, items: &mut Vec<Scalar>, _depth: usize) -> Result<(), PyErr> {
        shape.extend(&self.shape);
        items.extend(self.scalars());
        Ok(())
    }
    fn as_array(&self) -> Option<NdArray> {
        Some(self.clone())
    }
}

fn kind_arg(dtype: &Option<String>) -> Result<Option<Kind>, PyErr> {
    dtype.as_deref().map(Kind::parse).transpose()
}

/// `np.array(x, dtype=None)`: a new array (a copy of an array).
pub fn array(x: impl IntoArray, dtype: &Option<String>) -> Result<NdArray, PyErr> {
    let mut shape = vec![];
    let mut items = vec![];
    x.shape_and_items(&mut shape, &mut items, 0)?;
    // the widest element type (an empty list is float64)
    let k = match kind_arg(dtype)? {
        Some(k) => k,
        None => match x.as_array() {
            Some(a) => a.kind(),
            None => items.iter().map(|s| s.kind()).max().unwrap_or(Kind::Float),
        },
    };
    Ok(NdArray::from_data(Data::from_scalars(k, items.into_iter()), shape))
}

/// `np.asarray(x)`: `x` itself when it is an array.
pub fn asarray(x: impl IntoArray, dtype: &Option<String>) -> Result<NdArray, PyErr> {
    if let Some(a) = x.as_array() {
        if dtype.is_none() || kind_arg(dtype)? == Some(a.kind()) {
            return Ok(a);
        }
    }
    array(x, dtype)
}

fn filled(shape: Vec<usize>, x: Scalar, k: Kind) -> NdArray {
    let n = shape.iter().product();
    NdArray::from_data(Data::from_scalars(k, std::iter::repeat_n(x, n)), shape)
}

/// `np.zeros(shape, dtype="float64")`
pub fn zeros(shape: impl ShapeArg, dtype: &Option<String>) -> Result<NdArray, PyErr> {
    let k = kind_arg(dtype)?.unwrap_or(Kind::Float);
    Ok(filled(shape_of(&shape)?, Scalar::I(0), k))
}
/// `np.ones(shape, dtype="float64")`
pub fn ones(shape: impl ShapeArg, dtype: &Option<String>) -> Result<NdArray, PyErr> {
    let k = kind_arg(dtype)?.unwrap_or(Kind::Float);
    Ok(filled(shape_of(&shape)?, Scalar::I(1), k))
}
/// `np.full(shape, fill_value, dtype=None)`
pub fn full(shape: impl ShapeArg, fill: impl Operand, dtype: &Option<String>) -> Result<NdArray, PyErr> {
    let x = fill.scalar_value().ok_or_else(|| verr("fill_value must be a scalar"))?;
    let k = kind_arg(dtype)?.unwrap_or(x.kind());
    Ok(filled(shape_of(&shape)?, x, k))
}
/// `np.zeros_like(a)`
pub fn zeros_like(a: &NdArray) -> NdArray {
    filled(a.shape.clone(), Scalar::I(0), a.kind())
}
/// `np.ones_like(a)`
pub fn ones_like(a: &NdArray) -> NdArray {
    filled(a.shape.clone(), Scalar::I(1), a.kind())
}
/// `np.full_like(a, x)`
pub fn full_like(a: &NdArray, fill: impl Operand) -> NdArray {
    let x = fill.scalar_value().unwrap_or(Scalar::F(0.0));
    filled(a.shape.clone(), x, a.kind())
}
/// `np.eye(n, m=None, k=0)`
pub fn eye(n: i64, m: Option<i64>, k: i64) -> Result<NdArray, PyErr> {
    if n < 0 || m.is_some_and(|m| m < 0) {
        return Err(verr("negative dimensions are not allowed"));
    }
    let (n, m) = (n as usize, m.unwrap_or(n) as usize);
    let mut v = vec![0.0; n * m];
    for i in 0..n {
        let j = i as i64 + k;
        if j >= 0 && (j as usize) < m {
            v[i * m + j as usize] = 1.0;
        }
    }
    Ok(NdArray::floats(v, vec![n, m]))
}
/// `np.identity(n)`
pub fn identity(n: i64) -> Result<NdArray, PyErr> {
    eye(n, None, 0)
}

/// `np.arange(start, stop, step, dtype=None)` (int when all are ints)
pub fn arange(start: impl Operand, stop: impl Operand, step: impl Operand, dtype: &Option<String>) -> Result<NdArray, PyErr> {
    let (a, b, s) = (start.scalar_value().unwrap(), stop.scalar_value().unwrap(), step.scalar_value().unwrap());
    let all_int = [a, b, s].iter().all(|x| x.kind() != Kind::Float);
    if s.f() == 0.0 {
        return Err(PyErr::new(crate::pyrt::err::ZERO_DIVISION_ERROR, "division by zero"));
    }
    let r = if all_int {
        let (a, b, s) = (a.i(), b.i(), s.i());
        let n = if (s > 0 && b > a) || (s < 0 && b < a) { ((b - a).abs() + s.abs() - 1) / s.abs() } else { 0 };
        NdArray::from_data(Data::Int((0..n).map(|i| a + i * s).collect()), vec![n as usize])
    } else {
        let (a, b, s) = (a.f(), b.f(), s.f());
        let n = ((b - a) / s).ceil().max(0.0) as usize;
        // NumPy fills from the first two values: start + i * (next - start)
        let delta = (a + s) - a;
        NdArray::floats((0..n).map(|i| if i == 0 { a } else { a + i as f64 * delta }).collect(), vec![n])
    };
    match kind_arg(dtype)? {
        Some(k) if k != r.kind() => r.astype(k.name()),
        _ => Ok(r),
    }
}
/// `np.arange(stop)`
pub fn arange1(stop: impl Operand, dtype: &Option<String>) -> Result<NdArray, PyErr> {
    arange(0i64, stop, 1i64, dtype)
}
/// `np.arange(start, stop)`
pub fn arange2(start: impl Operand, stop: impl Operand, dtype: &Option<String>) -> Result<NdArray, PyErr> {
    arange(start, stop, 1i64, dtype)
}

/// `np.linspace(start, stop, num=50, endpoint=True)`
pub fn linspace(start: impl Operand, stop: impl Operand, num: i64, endpoint: bool) -> Result<NdArray, PyErr> {
    if num < 0 {
        return Err(verr(format!("Number of samples, {}, must be non-negative.", num)));
    }
    let (a, b) = (start.scalar_value().unwrap().f(), stop.scalar_value().unwrap().f());
    let n = num as usize;
    let div = if endpoint { n.saturating_sub(1) } else { n } as f64;
    let delta = b - a;
    let mut v: Vec<f64> = (0..n).map(|i| i as f64).collect();
    if div > 0.0 {
        let step = delta / div;
        if step == 0.0 {
            // NumPy: y = y / div * delta
            for y in v.iter_mut() {
                *y = *y / div * delta + a;
            }
        } else {
            for y in v.iter_mut() {
                *y = *y * step + a;
            }
        }
    } else {
        for y in v.iter_mut() {
            *y = *y * delta + a;
        }
    }
    if endpoint && n > 1 {
        v[n - 1] = b;
    }
    Ok(NdArray::floats(v, vec![n]))
}

/// `np.concatenate(arrays, axis=0)`
pub fn concatenate(arrays: &[NdArray], axis: i64) -> Result<NdArray, PyErr> {
    let Some(first) = arrays.first() else {
        return Err(verr("need at least one array to concatenate"));
    };
    let nd = first.shape.len();
    if nd == 0 {
        return Err(verr("zero-dimensional arrays cannot be concatenated"));
    }
    let ax = ops::axis_index(axis, nd)?;
    for (i, a) in arrays.iter().enumerate() {
        if a.shape.len() != nd {
            return Err(verr(format!(
                "all the input arrays must have same number of dimensions, but the array at index 0 has {} dimension(s) and the array at index {} has {} dimension(s)",
                nd, i, a.shape.len()
            )));
        }
        for d in 0..nd {
            if d != ax && a.shape[d] != first.shape[d] {
                return Err(verr(format!(
                    "all the input array dimensions except for the concatenation axis must match exactly, but along dimension {}, the array at index 0 has size {} and the array at index {} has size {}",
                    d, first.shape[d], i, a.shape[d]
                )));
            }
        }
    }
    let k = arrays.iter().map(|a| a.kind()).max().unwrap();
    let mut shape = first.shape.clone();
    shape[ax] = arrays.iter().map(|a| a.shape[ax]).sum();
    let outer: usize = shape[..ax].iter().product();
    let parts: Vec<(Vec<Scalar>, usize)> = arrays.iter().map(|a| (a.scalars(), a.shape[ax..].iter().product())).collect();
    let mut items = Vec::with_capacity(shape.iter().product());
    for o in 0..outer {
        for (v, block) in &parts {
            items.extend_from_slice(&v[o * block..(o + 1) * block]);
        }
    }
    Ok(NdArray::from_data(Data::from_scalars(k, items.into_iter()), shape))
}

/// `np.stack(arrays, axis=0)`
pub fn stack(arrays: &[NdArray], axis: i64) -> Result<NdArray, PyErr> {
    let Some(first) = arrays.first() else {
        return Err(verr("need at least one array to stack"));
    };
    if arrays.iter().any(|a| a.shape != first.shape) {
        return Err(verr("all input arrays must have the same shape"));
    }
    let ax = ops::axis_index(axis, first.shape.len() + 1)?;
    let expanded: Vec<NdArray> = arrays.iter().map(|a| ops::expand_dims(a, ax)).collect();
    concatenate(&expanded, ax as i64)
}

/// `np.vstack(arrays)`
pub fn vstack(arrays: &[NdArray]) -> Result<NdArray, PyErr> {
    let a: Vec<NdArray> = arrays.iter().map(|a| if a.shape.len() < 2 { ops::at_least_2d(a) } else { a.clone() }).collect();
    concatenate(&a, 0)
}

/// `np.hstack(arrays)`
pub fn hstack(arrays: &[NdArray]) -> Result<NdArray, PyErr> {
    let one_d = arrays.first().is_some_and(|a| a.shape.len() == 1);
    concatenate(arrays, if one_d { 0 } else { 1 })
}

/// `np.where(cond, x, y)`
pub fn r#where(cond: &NdArray, x: impl Operand, y: impl Operand) -> Result<NdArray, PyErr> {
    ops::where_(cond, x, y)
}

/// `print(x)` / `repr(x)` of the package's objects.
macro_rules! py_text {
    ($($t:ty),*) => {
        $(
            impl std::fmt::Display for $t {
                fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    f.write_str(&self.__str__())
                }
            }
            impl crate::pyrt::Py for $t {
                fn fmt_repr(&self, out: &mut String) {
                    out.push_str(&self.__repr__())
                }
                fn fmt_str(&self, out: &mut String) {
                    out.push_str(&self.__str__())
                }
            }
        )*
    };
}

py_text!(NdArray, Shape, DType, random::Generator);

/// A float sum in NumPy's order (pairwise), as `np.sum` of a contiguous array.
pub fn pairwise_sum(v: &[f64]) -> f64 {
    0.0 + reduce::pairwise(v)
}

/// A float mean in NumPy's order, as `np.mean`.
pub fn pairwise_mean(v: &[f64]) -> f64 {
    pairwise_sum(v) / v.len() as f64
}

//! Calling CPython libraries from py0 programs (feature `pyrt-python`).
//!
//! A stub whose first lines say `# python: MODULE` declares, in typed Python,
//! the functions, classes and properties of a real Python module; the
//! translator turns each use into [`call`] / [`get`] on an embedded CPython
//! (PyO3), converting arguments with [`ToPy`] and results with [`FromPy`] by
//! the declared types. Python objects without a py0 type are [`Object`]s.
//!
//! The interpreter is the one the program was built with (`PYO3_PYTHON`), or
//! `PY0_PYTHON` at run time; its `sys.path` (with a virtual environment's
//! packages) is taken from that interpreter.

use crate::pyrt::err::PyErr;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::cell::RefCell;
use std::collections::HashMap;

/// A Python object.
pub struct Object(Py<PyAny>);

impl Clone for Object {
    fn clone(&self) -> Object {
        Python::attach(|py| Object(self.0.clone_ref(py)))
    }
}

impl Default for Object {
    fn default() -> Object {
        Python::attach(|py| Object(py.None()))
    }
}

impl PartialEq for Object {
    /// `a is b`
    fn eq(&self, o: &Object) -> bool {
        self.0.as_ptr() == o.0.as_ptr()
    }
}

impl std::fmt::Debug for Object {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&self.repr())
    }
}

impl std::fmt::Display for Object {
    /// `str(obj)`
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let s = Python::attach(|py| self.0.bind(py).str().map(|s| s.to_string()).unwrap_or_default());
        f.write_str(&s)
    }
}

impl crate::pyrt::Py for Object {
    fn fmt_repr(&self, out: &mut String) {
        out.push_str(&self.repr())
    }
    fn fmt_str(&self, out: &mut String) {
        out.push_str(&self.to_string())
    }
}

impl Object {
    /// `repr(obj)`
    pub fn repr(&self) -> String {
        Python::attach(|py| self.0.bind(py).repr().map(|s| s.to_string()).unwrap_or_default())
    }
}

/// An argument the caller left out: Python's own default applies.
pub struct Missing;

// ------------------------------------------------------------ the interpreter

thread_local! {
    static MODULES: RefCell<HashMap<String, Object>> = RefCell::new(HashMap::new());
    static READY: RefCell<bool> = const { RefCell::new(false) };
}

/// Use the `sys.path` of the configured interpreter (a venv's packages too).
fn setup(py: Python<'_>) {
    if READY.with(|r| r.replace(true)) {
        return;
    }
    let exe = std::env::var("PY0_PYTHON").ok().or_else(|| option_env!("PYO3_PYTHON").map(String::from)).unwrap_or_else(|| "python3".into());
    let out = std::process::Command::new(&exe).args(["-c", "import sys; print('\\n'.join(sys.path))"]).output();
    if let Ok(o) = out {
        if o.status.success() {
            let paths: Vec<String> = String::from_utf8_lossy(&o.stdout).lines().map(String::from).collect();
            if let Ok(sys) = py.import("sys") {
                if let Ok(list) = PyList::new(py, &paths) {
                    let _ = sys.setattr("path", list);
                }
            }
        }
    }
}

/// A Python exception as a py0 exception (its class name and message).
fn from_python(py: Python<'_>, e: pyo3::PyErr) -> PyErr {
    let name: String = e.get_type(py).getattr("__name__").and_then(|n| n.extract()).unwrap_or_else(|_| "Exception".into());
    let msg = e.value(py).str().map(|s| s.to_string()).unwrap_or_default();
    let chain = crate::pyrt::err::builtin_chain(&name);
    let chain: &'static [&'static str] = if chain[0] == name {
        chain
    } else {
        Box::leak(vec![Box::leak(name.into_boxed_str()) as &str, "Exception", "BaseException"].into_boxed_slice())
    };
    PyErr::new(chain, msg)
}

/// `import MODULE` (cached).
pub fn module(name: &str) -> Object {
    if let Some(m) = MODULES.with(|m| m.borrow().get(name).cloned()) {
        return m;
    }
    let m = Python::attach(|py| {
        setup(py);
        match py.import(name) {
            Ok(m) => Object(m.into_any().unbind()),
            Err(e) => {
                let err = from_python(py, e);
                crate::pyrt::raise(err.chain[0], &err.msg)
            }
        }
    });
    MODULES.with(|ms| ms.borrow_mut().insert(name.to_string(), m.clone()));
    m
}

/// `MODULE.NAME` (a class of a module, for its static methods).
pub fn module_attr(module_name: &str, name: &str) -> Object {
    let m = module(module_name);
    Python::attach(|py| match m.0.bind(py).getattr(name) {
        Ok(a) => Object(a.unbind()),
        Err(e) => {
            let err = from_python(py, e);
            crate::pyrt::raise(err.chain[0], &err.msg)
        }
    })
}

fn flush_python(py: Python<'_>) {
    if let Ok(sys) = py.import("sys") {
        for f in ["stdout", "stderr"] {
            if let Ok(s) = sys.getattr(f) {
                let _ = s.call_method0("flush");
            }
        }
    }
}

/// `target.name(*positional, **keywords)` converted to `T`. Each argument is
/// `("", value)` (positional), `("*", list)` (spread) or `("key", value)`;
/// a [`Missing`] one is left out.
pub fn call<T: FromPy>(target: &Object, name: &str, args: &[(&str, &dyn ToPy)]) -> Result<T, PyErr> {
    crate::pyrt::out::flush();
    Python::attach(|py| {
        setup(py);
        let r = (|| -> PyResult<T> {
            let f = if name.is_empty() { target.0.bind(py).clone() } else { target.0.bind(py).getattr(name)? };
            let mut pos: Vec<Bound<'_, PyAny>> = vec![];
            let kw = PyDict::new(py);
            for (k, v) in args {
                let Some(x) = v.to_py(py)? else { continue };
                match *k {
                    "" => pos.push(x),
                    "*" => {
                        for item in x.try_iter()? {
                            pos.push(item?);
                        }
                    }
                    key => kw.set_item(key, x)?,
                }
            }
            let tuple = PyTuple::new(py, pos)?;
            let out = f.call(tuple, if kw.is_empty() { None } else { Some(&kw) })?;
            T::from_py(&out)
        })();
        flush_python(py);
        r.map_err(|e| from_python(py, e))
    })
}

/// `target.name` converted to `T` (a property).
pub fn get<T: FromPy>(target: &Object, name: &str) -> Result<T, PyErr> {
    Python::attach(|py| {
        setup(py);
        target.0.bind(py).getattr(name).and_then(|a| T::from_py(&a)).map_err(|e| from_python(py, e))
    })
}

/// Any value as a Python object (a stub's catch-all `Object` type).
pub fn object<T: ToPy + ?Sized>(x: &T) -> Object {
    Python::attach(|py| match x.to_py(py) {
        Ok(Some(o)) => Object(o.unbind()),
        Ok(None) => Object(py.None()),
        Err(e) => {
            let err = from_python(py, e);
            crate::pyrt::raise(err.chain[0], &err.msg)
        }
    })
}

/// A converted argument (for templates: `pyrt::python::arg(&x)`).
pub fn arg<T: ToPy>(x: &T) -> &dyn ToPy {
    x
}

// ------------------------------------------------------------ conversions

/// py0 values as Python objects.
pub trait ToPy {
    /// The object; `None` for [`Missing`] (the argument is left out).
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>>;
}

macro_rules! to_py_scalar {
    ($($t:ty),*) => {
        $(
            impl ToPy for $t {
                fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
                    Ok(Some(self.clone().into_pyobject(py)?.to_owned().into_any()))
                }
            }
        )*
    };
}
to_py_scalar!(i64, f64, bool, String);

impl ToPy for str {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        Ok(Some(self.into_pyobject(py)?.into_any()))
    }
}
impl ToPy for Missing {
    fn to_py<'py>(&self, _py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        Ok(None)
    }
}
impl ToPy for () {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        Ok(Some(py.None().into_bound(py)))
    }
}
impl ToPy for Object {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        Ok(Some(self.0.bind(py).clone()))
    }
}
impl<T: ToPy + ?Sized> ToPy for &T {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        (**self).to_py(py)
    }
}
impl<T: ToPy> ToPy for Option<T> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match self {
            Some(x) => x.to_py(py),
            None => Ok(Some(py.None().into_bound(py))),
        }
    }
}
fn list<'py, T: ToPy>(py: Python<'py>, xs: &[T]) -> PyResult<Option<Bound<'py, PyAny>>> {
    let items: Vec<Bound<'py, PyAny>> = xs.iter().map(|x| x.to_py(py).map(|o| o.unwrap_or_else(|| py.None().into_bound(py)))).collect::<PyResult<_>>()?;
    Ok(Some(PyList::new(py, items)?.into_any()))
}
impl<T: ToPy> ToPy for [T] {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        list(py, self)
    }
}
impl<T: ToPy> ToPy for Vec<T> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        list(py, self)
    }
}
impl<V: ToPy> ToPy for indexmap::IndexMap<String, V> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let d = PyDict::new(py);
        for (k, v) in self {
            if let Some(x) = v.to_py(py)? {
                d.set_item(k, x)?;
            }
        }
        Ok(Some(d.into_any()))
    }
}
macro_rules! to_py_tuple {
    ($($n:ident $i:tt),*) => {
        impl<$($n: ToPy),*> ToPy for ($($n,)*) {
            fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
                let items = vec![$(self.$i.to_py(py)?.unwrap_or_else(|| py.None().into_bound(py))),*];
                Ok(Some(PyTuple::new(py, items)?.into_any()))
            }
        }
    };
}
to_py_tuple!(A 0, B 1);
to_py_tuple!(A 0, B 1, C 2);
to_py_tuple!(A 0, B 1, C 2, D 3);

/// Python objects as py0 values (by the type a stub declares).
pub trait FromPy: Sized {
    /// Convert, or a Python `TypeError`.
    fn from_py(o: &Bound<'_, PyAny>) -> PyResult<Self>;
}

impl FromPy for () {
    fn from_py(_o: &Bound<'_, PyAny>) -> PyResult<()> {
        Ok(())
    }
}
impl FromPy for i64 {
    fn from_py(o: &Bound<'_, PyAny>) -> PyResult<i64> {
        // numpy integers too (`__index__`)
        o.extract().or_else(|_| o.call_method0("__index__")?.extract())
    }
}
impl FromPy for f64 {
    fn from_py(o: &Bound<'_, PyAny>) -> PyResult<f64> {
        o.extract().or_else(|_| o.call_method0("__float__")?.extract())
    }
}
impl FromPy for bool {
    fn from_py(o: &Bound<'_, PyAny>) -> PyResult<bool> {
        o.extract().or_else(|_| o.is_truthy())
    }
}
impl FromPy for String {
    fn from_py(o: &Bound<'_, PyAny>) -> PyResult<String> {
        o.extract()
    }
}
impl FromPy for Object {
    fn from_py(o: &Bound<'_, PyAny>) -> PyResult<Object> {
        Ok(Object(o.clone().unbind()))
    }
}
impl<T: FromPy> FromPy for Option<T> {
    fn from_py(o: &Bound<'_, PyAny>) -> PyResult<Option<T>> {
        if o.is_none() {
            Ok(None)
        } else {
            T::from_py(o).map(Some)
        }
    }
}
impl<T: FromPy> FromPy for Vec<T> {
    fn from_py(o: &Bound<'_, PyAny>) -> PyResult<Vec<T>> {
        o.try_iter()?.map(|x| T::from_py(&x?)).collect()
    }
}
impl<V: FromPy> FromPy for indexmap::IndexMap<String, V> {
    fn from_py(o: &Bound<'_, PyAny>) -> PyResult<indexmap::IndexMap<String, V>> {
        let d = o.cast::<PyDict>()?;
        d.iter().map(|(k, v)| Ok((k.extract()?, V::from_py(&v)?))).collect()
    }
}
macro_rules! from_py_tuple {
    ($($n:ident $i:tt),*) => {
        impl<$($n: FromPy),*> FromPy for ($($n,)*) {
            fn from_py(o: &Bound<'_, PyAny>) -> PyResult<Self> {
                Ok(($($n::from_py(&o.get_item($i)?)?,)*))
            }
        }
    };
}
from_py_tuple!(A 0, B 1);
from_py_tuple!(A 0, B 1, C 2);
from_py_tuple!(A 0, B 1, C 2, D 3);

// numpy arrays both ways (py0's numpy and the real one)
#[cfg(feature = "pyrt-numpy")]
mod numpy_bridge {
    use super::*;
    use crate::pyrt::lib::numpy::NdArray;

    impl ToPy for NdArray {
        fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
            let np = py.import("numpy")?;
            let dtype = self.dtype().name();
            let flat: Vec<f64> = self.to_f64();
            let a = np.call_method1("array", (flat,))?.call_method1("astype", (dtype,))?;
            let shape: Vec<usize> = self.dims().to_vec();
            Ok(Some(a.call_method1("reshape", (PyTuple::new(py, shape)?,))?))
        }
    }

    impl FromPy for NdArray {
        fn from_py(o: &Bound<'_, PyAny>) -> PyResult<NdArray> {
            let py = o.py();
            let np = py.import("numpy")?;
            let a = np.call_method1("asarray", (o,))?;
            let shape: Vec<i64> = a.getattr("shape")?.extract()?;
            let kind: String = a.getattr("dtype")?.getattr("kind")?.extract()?;
            let size: String = a.getattr("dtype")?.getattr("name")?.extract()?;
            let dtype = match (kind.as_str(), size.as_str()) {
                ("b", _) => "bool",
                ("i" | "u", _) => "int64",
                ("f", "float32") => "float32",
                _ => "float64",
            };
            let flat: Vec<f64> = a.call_method0("ravel")?.call_method1("astype", ("float64",))?.call_method0("tolist")?.extract()?;
            let r = crate::pyrt::lib::numpy::array(flat, &Some("float64".into()))
                .and_then(|x| x.astype(dtype))
                .and_then(|x| x.reshape(shape));
            r.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.msg))
        }
    }
}

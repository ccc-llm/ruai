//! `np.random.default_rng(seed)`: NumPy's `Generator` on PCG64, bit for bit
//! (`ruai::rng`).

use super::{verr, Data, NdArray, ShapeArg};
use crate::pyrt::err::PyErr;
use std::cell::RefCell;
use std::rc::Rc;

/// `numpy.random.Generator` (copies share the state, like Python names).
#[derive(Clone)]
pub struct Generator(Rc<RefCell<crate::rng::Generator>>);

impl std::fmt::Debug for Generator {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("Generator(PCG64)")
    }
}

impl PartialEq for Generator {
    fn eq(&self, o: &Generator) -> bool {
        Rc::ptr_eq(&self.0, &o.0)
    }
}

impl Default for Generator {
    fn default() -> Generator {
        Generator(Rc::new(RefCell::new(crate::rng::Generator::from_entropy())))
    }
}

fn dims(size: &impl ShapeArg) -> Result<Vec<usize>, PyErr> {
    size.dims().into_iter().map(|d| if d < 0 { Err(verr("negative dimensions are not allowed")) } else { Ok(d as usize) }).collect()
}

impl Generator {
    /// `str(rng)`
    pub fn __str__(&self) -> String {
        "Generator(PCG64)".into()
    }
    /// `repr(rng)`
    pub fn __repr__(&self) -> String {
        "Generator(PCG64)".into()
    }
    /// `rng.random()`
    pub fn random1(&self) -> f64 {
        self.0.borrow_mut().random()
    }
    /// `rng.random(size)`
    pub fn random(&self, size: impl ShapeArg) -> Result<NdArray, PyErr> {
        let shape = dims(&size)?;
        let n = shape.iter().product();
        let mut g = self.0.borrow_mut();
        Ok(NdArray::floats((0..n).map(|_| g.random()).collect(), shape))
    }
    /// `rng.uniform(low, high)`
    pub fn uniform1(&self, low: f64, high: f64) -> f64 {
        self.0.borrow_mut().uniform(low, high)
    }
    /// `rng.uniform(low, high, size)`
    pub fn uniform(&self, low: f64, high: f64, size: impl ShapeArg) -> Result<NdArray, PyErr> {
        let shape = dims(&size)?;
        let n = shape.iter().product();
        let mut g = self.0.borrow_mut();
        Ok(NdArray::floats((0..n).map(|_| g.uniform(low, high)).collect(), shape))
    }
    fn check(low: i64, high: i64) -> Result<(), PyErr> {
        if high <= low {
            return Err(verr("low >= high"));
        }
        if (high - low - 1) as u64 > u32::MAX as u64 {
            return Err(verr("ruai: integers() ranges above 2**32 are not supported"));
        }
        Ok(())
    }
    /// `rng.integers(low, high)`
    pub fn integers1(&self, low: i64, high: i64) -> Result<i64, PyErr> {
        Self::check(low, high)?;
        Ok(self.0.borrow_mut().integers(low, high))
    }
    /// `rng.integers(low, high, size)`
    pub fn integers(&self, low: i64, high: i64, size: impl ShapeArg) -> Result<NdArray, PyErr> {
        Self::check(low, high)?;
        let shape = dims(&size)?;
        let n = shape.iter().product();
        let mut g = self.0.borrow_mut();
        Ok(NdArray::from_data(Data::Int((0..n).map(|_| g.integers(low, high)).collect()), shape))
    }
    /// A uniform integer in `[0, max]` (NumPy's `random_interval`).
    fn interval(&self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut g = self.0.borrow_mut();
        let bits = g.bit_generator();
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        mask |= mask >> 32;
        if max <= 0xffff_ffff {
            loop {
                let v = bits.next_u32() as u64 & mask;
                if v <= max {
                    return v;
                }
            }
        }
        loop {
            let v = bits.next_u64() & mask;
            if v <= max {
                return v;
            }
        }
    }
    /// `rng.shuffle(x)`: in place, along the first axis.
    pub fn shuffle(&self, x: &NdArray) -> Result<(), PyErr> {
        let Some(&n) = x.shape.first() else {
            return Err(PyErr::new(crate::pyrt::err::TYPE_ERROR, "array is 0-dimensional"));
        };
        let rows: usize = x.shape[1..].iter().product();
        let offs = x.offsets();
        let mut d = x.data.borrow_mut();
        // Fisher-Yates from the end, as NumPy
        for i in (1..n).rev() {
            drop(d);
            let j = self.interval(i as u64) as usize;
            d = x.data.borrow_mut();
            for k in 0..rows {
                let (a, b) = (offs[i * rows + k], offs[j * rows + k]);
                let (va, vb) = (d.get(a), d.get(b));
                d.set(a, vb);
                d.set(b, va);
            }
        }
        Ok(())
    }
    /// `rng.permutation(n)`
    pub fn permutation(&self, n: i64) -> Result<NdArray, PyErr> {
        let a = super::arange1(n, &None)?;
        self.shuffle(&a)?;
        Ok(a)
    }
    /// `rng.permutation(x)`: a shuffled copy.
    pub fn permuted_copy(&self, x: &NdArray) -> Result<NdArray, PyErr> {
        let a = x.copy();
        self.shuffle(&a)?;
        Ok(a)
    }
}

/// `np.random.default_rng(seed)`
pub fn default_rng(seed: i64) -> Result<Generator, PyErr> {
    if seed < 0 {
        return Err(verr("expected non-negative integer"));
    }
    Ok(Generator(Rc::new(RefCell::new(crate::rng::Generator::new(seed as u64)))))
}

/// `np.random.default_rng()` (seeded from the clock)
pub fn default_rng0() -> Generator {
    Generator::default()
}

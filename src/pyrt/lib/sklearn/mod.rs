//! `sklearn` for py0: a subset of scikit-learn on `numpy` arrays, declared by
//! the stubs `src/py/lib/sklearn.*.pyi` (`from sklearn.linear_model import
//! LinearRegression`). Estimators follow scikit-learn's algorithms (the same
//! centering, solvers' optimum, k-means iterations, PCA sign convention, the
//! legacy `RandomState` permutation of `train_test_split`), so results agree
//! with scikit-learn up to floating-point rounding (iterative solvers that
//! scikit-learn stops early, like lbfgs, agree to their tolerance).
#![allow(clippy::needless_range_loop)]

pub mod models;

use super::numpy::NdArray;
use crate::pyrt::err::{PyErr, VALUE_ERROR};

pub(crate) fn verr(msg: impl Into<String>) -> PyErr {
    PyErr::new(VALUE_ERROR, msg.into())
}

const NOT_FITTED: &[&str] = &["NotFittedError", "ValueError", "AttributeError", "Exception", "BaseException"];

pub(crate) fn not_fitted(name: &str) -> PyErr {
    PyErr::new(
        NOT_FITTED,
        format!(
            "This {} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.",
            name
        ),
    )
}

/// A dense matrix (row-major).
#[derive(Clone, Debug, Default)]
pub(crate) struct Mat {
    pub n: usize,
    pub d: usize,
    pub v: Vec<f64>,
}

impl Mat {
    pub fn zeros(n: usize, d: usize) -> Mat {
        Mat { n, d, v: vec![0.0; n * d] }
    }
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.v[i * self.d + j]
    }
    pub fn row(&self, i: usize) -> &[f64] {
        &self.v[i * self.d..(i + 1) * self.d]
    }
    pub fn to_array(&self) -> NdArray {
        NdArray::floats(self.v.clone(), vec![self.n, self.d])
    }
    pub fn t(&self) -> Mat {
        let mut m = Mat::zeros(self.d, self.n);
        for i in 0..self.n {
            for j in 0..self.d {
                m.v[j * self.n + i] = self.at(i, j);
            }
        }
        m
    }
    /// The mean of each column (NumPy: rows added in order).
    pub fn col_means(&self) -> Vec<f64> {
        let mut s = vec![0.0; self.d];
        for i in 0..self.n {
            for (j, x) in s.iter_mut().enumerate() {
                *x += self.at(i, j);
            }
        }
        s.iter().map(|x| x / self.n as f64).collect()
    }
    /// The (population) variance of each column, as `np.var(X, axis=0)`.
    pub fn col_vars(&self) -> Vec<f64> {
        let m = self.col_means();
        let mut s = vec![0.0; self.d];
        for i in 0..self.n {
            for (j, x) in s.iter_mut().enumerate() {
                let t = self.at(i, j) - m[j];
                *x += t * t;
            }
        }
        s.iter().map(|x| x / self.n as f64).collect()
    }
}

/// `X` as a 2-d float matrix (scikit-learn's `check_array`).
pub(crate) fn matrix(x: &NdArray) -> Result<Mat, PyErr> {
    match x.dims().len() {
        2 => {
            let (n, d) = (x.dims()[0], x.dims()[1]);
            if n == 0 {
                return Err(verr(format!(
                    "Found array with 0 sample(s) (shape=({}, {})) while a minimum of 1 is required.",
                    n, d
                )));
            }
            let v = x.to_f64();
            if v.iter().any(|f| !f.is_finite()) {
                return Err(verr("Input X contains NaN or infinity."));
            }
            Ok(Mat { n, d, v })
        }
        1 => Err(verr(format!(
            "Expected 2D array, got 1D array instead:\narray={}.\nReshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.",
            x
        ))),
        k => Err(verr(format!("Found array with dim {}, while dim <= 2 is required.", k))),
    }
}

/// `y` as a 1-d float vector of `n` values.
pub(crate) fn vector(y: &NdArray, n: usize) -> Result<Vec<f64>, PyErr> {
    if y.dims().len() != 1 {
        return Err(verr("ruai: y must be a 1-d array"));
    }
    if y.dims()[0] != n {
        return Err(verr(format!(
            "Found input variables with inconsistent numbers of samples: [{}, {}]",
            n,
            y.dims()[0]
        )));
    }
    Ok(y.to_f64())
}

// ------------------------------------------------------------ linear algebra

/// Solve a symmetric positive definite system (Cholesky).
pub(crate) fn cholesky_solve(a: &Mat, b: &[f64]) -> Result<Vec<f64>, PyErr> {
    let n = a.n;
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a.at(i, j);
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(PyErr::new(&["LinAlgError", "ValueError", "Exception", "BaseException"], "Matrix is not positive definite"));
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let mut y = b.to_vec();
    for i in 0..n {
        for k in 0..i {
            y[i] -= l[i * n + k] * y[k];
        }
        y[i] /= l[i * n + i];
    }
    for i in (0..n).rev() {
        for k in i + 1..n {
            y[i] -= l[k * n + i] * y[k];
        }
        y[i] /= l[i * n + i];
    }
    Ok(y)
}

/// Thin SVD by one-sided Jacobi: (U n×k, s, Vt k×d), k = min(n, d), singular
/// values in decreasing order.
pub(crate) fn svd(a: &Mat) -> (Mat, Vec<f64>, Mat) {
    if a.n < a.d {
        let (u, s, vt) = svd(&a.t());
        return (vt.t(), s, u.t());
    }
    let (n, d) = (a.n, a.d);
    let mut u = a.clone();
    let mut v = Mat::zeros(d, d);
    for i in 0..d {
        v.v[i * d + i] = 1.0;
    }
    for _sweep in 0..100 {
        let mut off = 0.0f64;
        for p in 0..d {
            for q in p + 1..d {
                let (mut alpha, mut beta, mut gamma) = (0.0, 0.0, 0.0);
                for i in 0..n {
                    let (x, y) = (u.v[i * d + p], u.v[i * d + q]);
                    alpha += x * x;
                    beta += y * y;
                    gamma += x * y;
                }
                if gamma == 0.0 {
                    continue;
                }
                off = off.max(gamma.abs() / (alpha * beta).sqrt());
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let t = if zeta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..n {
                    let (x, y) = (u.v[i * d + p], u.v[i * d + q]);
                    u.v[i * d + p] = c * x - s * y;
                    u.v[i * d + q] = s * x + c * y;
                }
                for i in 0..d {
                    let (x, y) = (v.v[i * d + p], v.v[i * d + q]);
                    v.v[i * d + p] = c * x - s * y;
                    v.v[i * d + q] = s * x + c * y;
                }
            }
        }
        if off < 1e-15 {
            break;
        }
    }
    let mut sv: Vec<(f64, usize)> = (0..d).map(|j| ((0..n).map(|i| u.v[i * d + j].powi(2)).sum::<f64>().sqrt(), j)).collect();
    sv.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    let mut uo = Mat::zeros(n, d);
    let mut vt = Mat::zeros(d, d);
    let mut s = vec![];
    for (k, &(sig, j)) in sv.iter().enumerate() {
        s.push(sig);
        for i in 0..n {
            uo.v[i * d + k] = if sig > 0.0 { u.v[i * d + j] / sig } else { 0.0 };
        }
        for i in 0..d {
            vt.v[k * d + i] = v.v[i * d + j];
        }
    }
    (uo, s, vt)
}

/// The minimum-norm least-squares solution of `a x = b` (as `scipy.linalg.lstsq`).
pub(crate) fn lstsq(a: &Mat, b: &[f64]) -> Vec<f64> {
    let (u, s, vt) = svd(a);
    let k = s.len();
    let cutoff = f64::EPSILON * a.n.max(a.d) as f64 * s.first().copied().unwrap_or(0.0);
    let mut x = vec![0.0; a.d];
    for c in 0..k {
        if s[c] <= cutoff {
            continue;
        }
        let ub: f64 = (0..a.n).map(|i| u.v[i * u.d + c] * b[i]).sum();
        for j in 0..a.d {
            x[j] += vt.v[c * vt.d + j] * ub / s[c];
        }
    }
    x
}

// ------------------------------------------------------------ RandomState

/// NumPy's legacy `RandomState` (MT19937 seeded by `init_genrand`), the
/// generator of scikit-learn's `random_state=int`.
pub(crate) struct RandomState {
    mt: [u32; 624],
    i: usize,
}

impl RandomState {
    pub fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1812433253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        RandomState { mt, i: 624 }
    }
    pub fn next_u32(&mut self) -> u32 {
        if self.i >= 624 {
            for k in 0..624 {
                let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % 624] & 0x7fff_ffff);
                let mut v = self.mt[(k + 397) % 624] ^ (y >> 1);
                if y & 1 == 1 {
                    v ^= 0x9908_b0df;
                }
                self.mt[k] = v;
            }
            self.i = 0;
        }
        let mut y = self.mt[self.i];
        self.i += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
    /// A uniform integer in `[0, max]` (`random_interval`).
    fn interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        for s in [1, 2, 4, 8, 16, 32] {
            mask |= mask >> s;
        }
        loop {
            let v = if max <= 0xffff_ffff { self.next_u32() as u64 & mask } else { ((self.next_u32() as u64) << 32 | self.next_u32() as u64) & mask };
            if v <= max {
                return v;
            }
        }
    }
    /// `permutation(n)`
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.interval(i as u64) as usize;
            p.swap(i, j);
        }
        p
    }
}

fn rows(a: &NdArray, idx: &[usize]) -> Result<NdArray, PyErr> {
    a.__getitem__(&crate::pyrt::Index(vec![crate::pyrt::Ix::Ints(idx.iter().map(|&i| i as i64).collect())]))
}

fn split_sizes(n: usize, test_size: f64) -> Result<(usize, usize), PyErr> {
    if test_size <= 0.0 || (test_size < 1.0 && (test_size * n as f64).ceil() as usize >= n) || test_size >= n as f64 {
        return Err(verr(format!(
            "With n_samples={}, test_size={} and train_size=None, the resulting train set will be empty. Adjust any of the aforementioned parameters.",
            n,
            crate::pyrt::float_repr(test_size)
        )));
    }
    let n_test = if test_size < 1.0 { (test_size * n as f64).ceil() as usize } else { test_size as usize };
    Ok((n - n_test, n_test))
}

/// `train_test_split(X, y, test_size=0.25, random_state=None)`
pub fn train_test_split(
    x: &NdArray,
    y: &NdArray,
    test_size: f64,
    random_state: Option<i64>,
) -> Result<(NdArray, NdArray, NdArray, NdArray), PyErr> {
    let n = x.dims().first().copied().unwrap_or(0);
    if y.dims().first().copied().unwrap_or(0) != n {
        return Err(verr(format!(
            "Found input variables with inconsistent numbers of samples: [{}, {}]",
            n,
            y.dims().first().copied().unwrap_or(0)
        )));
    }
    let (n_train, n_test) = split_sizes(n, test_size)?;
    let seed = random_state.unwrap_or_else(|| crate::rng::Generator::from_entropy().integers(0, 1 << 31));
    let mut rs = RandomState::new(seed as u32);
    let p = rs.permutation(n);
    let (test, train) = (&p[..n_test], &p[n_test..n_test + n_train]);
    Ok((rows(x, train)?, rows(x, test)?, rows(y, train)?, rows(y, test)?))
}

// ------------------------------------------------------------ metrics

fn pair(a: &NdArray, b: &NdArray) -> Result<(Vec<f64>, Vec<f64>), PyErr> {
    let (x, y) = (a.to_f64(), b.to_f64());
    if x.len() != y.len() {
        return Err(verr(format!("Found input variables with inconsistent numbers of samples: [{}, {}]", x.len(), y.len())));
    }
    Ok((x, y))
}

/// `accuracy_score(y_true, y_pred)`
pub fn accuracy_score(t: &NdArray, p: &NdArray) -> Result<f64, PyErr> {
    let (x, y) = pair(t, p)?;
    Ok(x.iter().zip(&y).filter(|(a, b)| a == b).count() as f64 / x.len() as f64)
}

/// `mean_squared_error(y_true, y_pred)`
pub fn mean_squared_error(t: &NdArray, p: &NdArray) -> Result<f64, PyErr> {
    let (x, y) = pair(t, p)?;
    let d: Vec<f64> = x.iter().zip(&y).map(|(a, b)| (a - b) * (a - b)).collect();
    Ok(super::numpy::pairwise_mean(&d))
}

/// `mean_absolute_error(y_true, y_pred)`
pub fn mean_absolute_error(t: &NdArray, p: &NdArray) -> Result<f64, PyErr> {
    let (x, y) = pair(t, p)?;
    let d: Vec<f64> = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).collect();
    Ok(super::numpy::pairwise_mean(&d))
}

/// `r2_score(y_true, y_pred)`
pub fn r2_score(t: &NdArray, p: &NdArray) -> Result<f64, PyErr> {
    let (x, y) = pair(t, p)?;
    let ss_res: Vec<f64> = x.iter().zip(&y).map(|(a, b)| (a - b) * (a - b)).collect();
    let mean = super::numpy::pairwise_mean(&x);
    let ss_tot: Vec<f64> = x.iter().map(|a| (a - mean) * (a - mean)).collect();
    let (num, den) = (super::numpy::pairwise_sum(&ss_res), super::numpy::pairwise_sum(&ss_tot));
    if den == 0.0 {
        return Ok(if num == 0.0 { 1.0 } else { 0.0 });
    }
    Ok(1.0 - num / den)
}

/// `confusion_matrix(y_true, y_pred)`: rows are true labels, columns predictions.
pub fn confusion_matrix(t: &NdArray, p: &NdArray) -> Result<NdArray, PyErr> {
    let (x, y) = pair(t, p)?;
    let mut labels: Vec<f64> = x.iter().chain(&y).copied().collect();
    labels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    labels.dedup();
    let k = labels.len();
    let pos = |v: f64| labels.iter().position(|&l| l == v).unwrap();
    let mut m = vec![0i64; k * k];
    for (a, b) in x.iter().zip(&y) {
        m[pos(*a) * k + pos(*b)] += 1;
    }
    Ok(NdArray::ints(m, vec![k, k]))
}

// ------------------------------------------------------------ datasets

/// `sklearn.datasets.load_iris()`
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bunch {
    data: Option<NdArray>,
    target: Option<NdArray>,
    feature_names: Vec<String>,
    target_names: Vec<String>,
}

impl Bunch {
    /// `bunch.data`
    pub fn data(&self) -> NdArray {
        self.data.clone().unwrap_or_default()
    }
    /// `bunch.target`
    pub fn target(&self) -> NdArray {
        self.target.clone().unwrap_or_default()
    }
    /// `bunch.feature_names`
    pub fn feature_names(&self) -> Vec<String> {
        self.feature_names.clone()
    }
    /// `bunch.target_names` (a list here; an array of str in scikit-learn)
    pub fn target_names(&self) -> Vec<String> {
        self.target_names.clone()
    }
}

/// The iris data set (Fisher, 1936), as shipped by scikit-learn.
pub fn load_iris() -> Bunch {
    let text = include_str!("iris.csv");
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().unwrap().split(',').collect();
    let mut data = vec![];
    let mut target = vec![];
    for l in lines.filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = l.split(',').collect();
        data.extend(f[..4].iter().map(|x| x.parse::<f64>().unwrap()));
        target.push(f[4].trim().parse::<i64>().unwrap());
    }
    let n = target.len();
    Bunch {
        data: Some(NdArray::floats(data, vec![n, 4])),
        target: Some(NdArray::ints(target, vec![n])),
        feature_names: ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"].map(String::from).to_vec(),
        target_names: head[2..].iter().map(|s| s.to_string()).collect(),
    }
}

//! The estimators. Each is a handle to shared state (fitting through one name
//! is seen through all, as in Python), printed like scikit-learn's repr.
#![allow(clippy::needless_range_loop)]

use super::{cholesky_solve, lstsq, matrix, not_fitted, svd, vector, verr, Mat};
use crate::pyrt::err::PyErr;
use crate::pyrt::lib::numpy::NdArray;
use std::cell::RefCell;
use std::rc::Rc;

/// `Name(param=value, ...)` with the parameters that differ from the defaults, sorted.
fn repr(name: &str, mut params: Vec<(&str, String)>) -> String {
    params.sort_by(|a, b| a.0.cmp(b.0));
    let p: Vec<String> = params.into_iter().map(|(k, v)| format!("{}={}", k, v)).collect();
    format!("{}({})", name, p.join(", "))
}

fn f(x: f64) -> String {
    crate::pyrt::float_repr(x)
}

/// A float vector as a 1-d array.
fn vec1(v: &[f64]) -> NdArray {
    NdArray::floats(v.to_vec(), vec![v.len()])
}

fn scalar(x: f64) -> NdArray {
    NdArray::from_scalar(crate::pyrt::lib::numpy::Scalar::F(x))
}

/// The distinct labels (sorted) and each sample's label index.
fn encode(y: &NdArray) -> Result<(NdArray, Vec<usize>), PyErr> {
    let classes = crate::pyrt::lib::numpy::unique(y);
    let cv = classes.to_f64();
    let yv = y.to_f64();
    let idx = yv.iter().map(|v| cv.iter().position(|c| c == v).unwrap()).collect();
    Ok((classes, idx))
}

/// Labels of `classes` at positions `idx`.
fn take(classes: &NdArray, idx: &[usize]) -> Result<NdArray, PyErr> {
    classes.__getitem__(&crate::pyrt::Index(vec![crate::pyrt::Ix::Ints(idx.iter().map(|&i| i as i64).collect())]))
}

fn check_features(x: &Mat, d: usize, name: &str) -> Result<(), PyErr> {
    if x.d != d {
        return Err(verr(format!(
            "X has {} features, but {} is expecting {} features as input.",
            x.d, name, d
        )));
    }
    Ok(())
}

macro_rules! handle {
    ($name:ident, $state:ident) => {
        #[doc = concat!("`sklearn.", stringify!($name), "`")]
        #[derive(Clone, Default)]
        pub struct $name(Rc<RefCell<$state>>);

        impl PartialEq for $name {
            fn eq(&self, o: &$name) -> bool {
                Rc::ptr_eq(&self.0, &o.0)
            }
        }
        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str(&self.__repr__())
            }
        }
        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str(&self.__repr__())
            }
        }
        impl crate::pyrt::Py for $name {
            fn fmt_repr(&self, out: &mut String) {
                out.push_str(&self.__repr__())
            }
        }
        impl $name {
            #[doc = "`str(model)`"]
            pub fn __str__(&self) -> String {
                self.__repr__()
            }
            fn st(&self) -> std::cell::Ref<'_, $state> {
                self.0.borrow()
            }
        }
    };
}

// ------------------------------------------------------------ preprocessing

#[derive(Default)]
pub struct ScalerState {
    with_mean: bool,
    with_std: bool,
    mean: Option<Vec<f64>>,
    var: Vec<f64>,
    scale: Vec<f64>,
}
handle!(StandardScaler, ScalerState);

impl StandardScaler {
    /// `StandardScaler(with_mean=True, with_std=True)`
    pub fn new(with_mean: bool, with_std: bool) -> StandardScaler {
        StandardScaler(Rc::new(RefCell::new(ScalerState { with_mean, with_std, ..Default::default() })))
    }
    /// `repr(scaler)`
    pub fn __repr__(&self) -> String {
        let s = self.st();
        let mut p = vec![];
        if !s.with_mean {
            p.push(("with_mean", "False".into()));
        }
        if !s.with_std {
            p.push(("with_std", "False".into()));
        }
        repr("StandardScaler", p)
    }
    /// `scaler.fit(X)`
    pub fn fit(&self, x: &NdArray) -> Result<StandardScaler, PyErr> {
        let m = matrix(x)?;
        // scikit-learn's _incremental_mean_and_var from zero
        let n = m.n as f64;
        let mut sum = vec![0.0; m.d];
        for i in 0..m.n {
            for (j, s) in sum.iter_mut().enumerate() {
                *s += m.at(i, j);
            }
        }
        let mean: Vec<f64> = sum.iter().map(|s| s / n).collect();
        let mut corr = vec![0.0; m.d];
        let mut sq = vec![0.0; m.d];
        for i in 0..m.n {
            for j in 0..m.d {
                let t = m.at(i, j) - mean[j];
                corr[j] += t;
                sq[j] += t * t;
            }
        }
        let var: Vec<f64> = (0..m.d).map(|j| (sq[j] - corr[j] * corr[j] / n) / n).collect();
        let mut st = self.0.borrow_mut();
        st.scale = var.iter().map(|&v| if v.sqrt() < 10.0 * f64::EPSILON { 1.0 } else { v.sqrt() }).collect();
        st.var = var;
        st.mean = Some(mean);
        drop(st);
        Ok(self.clone())
    }
    fn params(&self, d: usize) -> Result<(Vec<f64>, Vec<f64>), PyErr> {
        let s = self.st();
        let Some(mean) = &s.mean else { return Err(not_fitted("StandardScaler")) };
        if mean.len() != d {
            return Err(verr(format!("X has {} features, but StandardScaler is expecting {} features as input.", d, mean.len())));
        }
        let m = if s.with_mean { mean.clone() } else { vec![0.0; d] };
        let sc = if s.with_std { s.scale.clone() } else { vec![1.0; d] };
        Ok((m, sc))
    }
    /// `scaler.transform(X)`
    pub fn transform(&self, x: &NdArray) -> Result<NdArray, PyErr> {
        let mut m = matrix(x)?;
        let (mean, scale) = self.params(m.d)?;
        for i in 0..m.n {
            for j in 0..m.d {
                m.v[i * m.d + j] = (m.at(i, j) - mean[j]) / scale[j];
            }
        }
        Ok(m.to_array())
    }
    /// `scaler.fit_transform(X)`
    pub fn fit_transform(&self, x: &NdArray) -> Result<NdArray, PyErr> {
        self.fit(x)?;
        self.transform(x)
    }
    /// `scaler.inverse_transform(X)`
    pub fn inverse_transform(&self, x: &NdArray) -> Result<NdArray, PyErr> {
        let mut m = matrix(x)?;
        let (mean, scale) = self.params(m.d)?;
        for i in 0..m.n {
            for j in 0..m.d {
                m.v[i * m.d + j] = m.at(i, j) * scale[j] + mean[j];
            }
        }
        Ok(m.to_array())
    }
    /// `scaler.mean_`
    pub fn mean_(&self) -> Result<NdArray, PyErr> {
        self.st().mean.as_deref().map(vec1).ok_or_else(|| not_fitted("StandardScaler"))
    }
    /// `scaler.var_`
    pub fn var_(&self) -> Result<NdArray, PyErr> {
        self.mean_()?;
        Ok(vec1(&self.st().var))
    }
    /// `scaler.scale_`
    pub fn scale_(&self) -> Result<NdArray, PyErr> {
        self.mean_()?;
        Ok(vec1(&self.st().scale))
    }
}

#[derive(Default)]
pub struct MinMaxState {
    min: Option<Vec<f64>>,
    max: Vec<f64>,
    scale: Vec<f64>,
    offset: Vec<f64>,
}
handle!(MinMaxScaler, MinMaxState);

impl MinMaxScaler {
    /// `MinMaxScaler()`
    pub fn new() -> MinMaxScaler {
        MinMaxScaler::default()
    }
    /// `repr(scaler)`
    pub fn __repr__(&self) -> String {
        "MinMaxScaler()".into()
    }
    /// `scaler.fit(X)`
    pub fn fit(&self, x: &NdArray) -> Result<MinMaxScaler, PyErr> {
        let m = matrix(x)?;
        let mut lo = vec![f64::INFINITY; m.d];
        let mut hi = vec![f64::NEG_INFINITY; m.d];
        for i in 0..m.n {
            for j in 0..m.d {
                lo[j] = lo[j].min(m.at(i, j));
                hi[j] = hi[j].max(m.at(i, j));
            }
        }
        let mut st = self.0.borrow_mut();
        st.scale = (0..m.d)
            .map(|j| {
                let r = hi[j] - lo[j];
                1.0 / if r < 10.0 * f64::EPSILON { 1.0 } else { r }
            })
            .collect();
        st.offset = (0..m.d).map(|j| 0.0 - lo[j] * st.scale[j]).collect();
        st.min = Some(lo);
        st.max = hi;
        drop(st);
        Ok(self.clone())
    }
    /// `scaler.transform(X)`
    pub fn transform(&self, x: &NdArray) -> Result<NdArray, PyErr> {
        let mut m = matrix(x)?;
        let s = self.st();
        let Some(lo) = &s.min else { return Err(not_fitted("MinMaxScaler")) };
        check_features(&m, lo.len(), "MinMaxScaler")?;
        for i in 0..m.n {
            for j in 0..m.d {
                m.v[i * m.d + j] = m.at(i, j) * s.scale[j] + s.offset[j];
            }
        }
        Ok(m.to_array())
    }
    /// `scaler.fit_transform(X)`
    pub fn fit_transform(&self, x: &NdArray) -> Result<NdArray, PyErr> {
        self.fit(x)?;
        self.transform(x)
    }
    /// `scaler.data_min_`
    pub fn data_min_(&self) -> Result<NdArray, PyErr> {
        self.st().min.as_deref().map(vec1).ok_or_else(|| not_fitted("MinMaxScaler"))
    }
    /// `scaler.data_max_`
    pub fn data_max_(&self) -> Result<NdArray, PyErr> {
        self.data_min_()?;
        Ok(vec1(&self.st().max))
    }
}

// ------------------------------------------------------------ linear models

#[derive(Default)]
pub struct LinearState {
    fit_intercept: bool,
    alpha: f64,
    coef: Option<Vec<f64>>,
    intercept: f64,
}

/// Center X and y as scikit-learn's `_preprocess_data`.
fn center(m: &Mat, y: &[f64], fit_intercept: bool) -> (Mat, Vec<f64>, Vec<f64>, f64) {
    if !fit_intercept {
        return (m.clone(), y.to_vec(), vec![0.0; m.d], 0.0);
    }
    let xo = m.col_means();
    let yo = crate::pyrt::lib::numpy::pairwise_mean(y);
    let mut c = m.clone();
    for i in 0..m.n {
        for j in 0..m.d {
            c.v[i * m.d + j] -= xo[j];
        }
    }
    (c, y.iter().map(|v| v - yo).collect(), xo, yo)
}

fn predict_linear(x: &NdArray, coef: &[f64], intercept: f64, name: &str) -> Result<NdArray, PyErr> {
    let m = matrix(x)?;
    check_features(&m, coef.len(), name)?;
    Ok(vec1(&(0..m.n).map(|i| m.row(i).iter().zip(coef).fold(0.0, |s, (a, b)| s + a * b) + intercept).collect::<Vec<_>>()))
}

macro_rules! linear_common {
    ($name:ident) => {
        impl $name {
            /// `model.predict(X)`
            pub fn predict(&self, x: &NdArray) -> Result<NdArray, PyErr> {
                let s = self.st();
                let Some(c) = &s.coef else { return Err(not_fitted(stringify!($name))) };
                predict_linear(x, c, s.intercept, stringify!($name))
            }
            /// `model.score(X, y)`: R²
            pub fn score(&self, x: &NdArray, y: &NdArray) -> Result<f64, PyErr> {
                super::r2_score(y, &self.predict(x)?)
            }
            /// `model.coef_`
            pub fn coef_(&self) -> Result<NdArray, PyErr> {
                self.st().coef.as_deref().map(vec1).ok_or_else(|| not_fitted(stringify!($name)))
            }
            /// `model.intercept_`
            pub fn intercept_(&self) -> Result<NdArray, PyErr> {
                self.coef_()?;
                Ok(scalar(self.st().intercept))
            }
        }
    };
}

handle!(LinearRegression, LinearState);
linear_common!(LinearRegression);

impl LinearRegression {
    /// `LinearRegression(fit_intercept=True)`
    pub fn new(fit_intercept: bool) -> LinearRegression {
        LinearRegression(Rc::new(RefCell::new(LinearState { fit_intercept, ..Default::default() })))
    }
    /// `repr(model)`
    pub fn __repr__(&self) -> String {
        let s = self.st();
        repr("LinearRegression", if s.fit_intercept { vec![] } else { vec![("fit_intercept", "False".into())] })
    }
    /// `model.fit(X, y)`: least squares (the minimum-norm solution)
    pub fn fit(&self, x: &NdArray, y: &NdArray) -> Result<LinearRegression, PyErr> {
        let m = matrix(x)?;
        let yv = vector(y, m.n)?;
        let fi = self.st().fit_intercept;
        let (c, yc, xo, yo) = center(&m, &yv, fi);
        let coef = lstsq(&c, &yc);
        let intercept = if fi { yo - xo.iter().zip(&coef).fold(0.0, |s, (a, b)| s + a * b) } else { 0.0 };
        let mut st = self.0.borrow_mut();
        st.coef = Some(coef);
        st.intercept = intercept;
        drop(st);
        Ok(self.clone())
    }
}

handle!(Ridge, LinearState);
linear_common!(Ridge);

impl Ridge {
    /// `Ridge(alpha=1.0, fit_intercept=True)`
    pub fn new(alpha: f64, fit_intercept: bool) -> Ridge {
        Ridge(Rc::new(RefCell::new(LinearState { fit_intercept, alpha, ..Default::default() })))
    }
    /// `repr(model)`
    pub fn __repr__(&self) -> String {
        let s = self.st();
        let mut p = vec![];
        if s.alpha != 1.0 {
            p.push(("alpha", f(s.alpha)));
        }
        if !s.fit_intercept {
            p.push(("fit_intercept", "False".into()));
        }
        repr("Ridge", p)
    }
    /// `model.fit(X, y)`: (XᵀX + αI) w = Xᵀy on centered data (Cholesky)
    pub fn fit(&self, x: &NdArray, y: &NdArray) -> Result<Ridge, PyErr> {
        let m = matrix(x)?;
        let yv = vector(y, m.n)?;
        let (fi, alpha) = (self.st().fit_intercept, self.st().alpha);
        let (c, yc, xo, yo) = center(&m, &yv, fi);
        let mut a = Mat::zeros(c.d, c.d);
        let mut b = vec![0.0; c.d];
        for p in 0..c.d {
            for q in 0..c.d {
                a.v[p * c.d + q] = (0..c.n).fold(0.0, |s, i| s + c.at(i, p) * c.at(i, q));
            }
            a.v[p * c.d + p] += alpha;
            b[p] = (0..c.n).fold(0.0, |s, i| s + c.at(i, p) * yc[i]);
        }
        let coef = cholesky_solve(&a, &b)?;
        let intercept = if fi { yo - xo.iter().zip(&coef).fold(0.0, |s, (a, b)| s + a * b) } else { 0.0 };
        let mut st = self.0.borrow_mut();
        st.coef = Some(coef);
        st.intercept = intercept;
        drop(st);
        Ok(self.clone())
    }
}

// ------------------------------------------------------------ logistic regression

#[derive(Default)]
pub struct LogisticState {
    c: f64,
    max_iter: i64,
    classes: Option<NdArray>,
    /// one row per class (one row for two classes), intercept last
    w: Vec<Vec<f64>>,
}
handle!(LogisticRegression, LogisticState);

fn softmax_rows(z: &mut [f64]) {
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut s = 0.0;
    for v in z.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    for v in z.iter_mut() {
        *v /= s;
    }
}

/// Solve `a x = b` (Gaussian elimination with partial pivoting).
fn gauss(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Vec<f64> {
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i * n + c].abs().partial_cmp(&a[j * n + c].abs()).unwrap()).unwrap();
        for j in 0..n {
            a.swap(c * n + j, p * n + j);
        }
        b.swap(c, p);
        let d = a[c * n + c];
        if d == 0.0 {
            continue;
        }
        for r in c + 1..n {
            let f = a[r * n + c] / d;
            for j in c..n {
                a[r * n + j] -= f * a[c * n + j];
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut s = b[r];
        for j in r + 1..n {
            s -= a[r * n + j] * x[j];
        }
        x[r] = if a[r * n + r] == 0.0 { 0.0 } else { s / a[r * n + r] };
    }
    x
}

impl LogisticRegression {
    /// `LogisticRegression(C=1.0, max_iter=100)`
    pub fn new(c: f64, max_iter: i64) -> LogisticRegression {
        LogisticRegression(Rc::new(RefCell::new(LogisticState { c, max_iter, ..Default::default() })))
    }
    /// `repr(model)`
    pub fn __repr__(&self) -> String {
        let s = self.st();
        let mut p = vec![];
        if s.c != 1.0 {
            p.push(("C", f(s.c)));
        }
        if s.max_iter != 100 {
            p.push(("max_iter", s.max_iter.to_string()));
        }
        repr("LogisticRegression", p)
    }
    /// `model.fit(X, y)`: L2-penalized log loss (multinomial for more than two
    /// classes) minimized by Newton's method.
    pub fn fit(&self, x: &NdArray, y: &NdArray) -> Result<LogisticRegression, PyErr> {
        let m = matrix(x)?;
        vector(y, m.n)?;
        let (classes, yi) = encode(y)?;
        let k = classes.len();
        if k < 2 {
            return Err(verr(format!(
                "This solver needs samples of at least 2 classes in the data, but the data contains only one class: {:?}",
                classes.__getitem__(&crate::pyrt::Index(vec![crate::pyrt::Ix::Int(0)]))?
            )));
        }
        let c = self.st().c;
        let d = m.d + 1;
        // the rows that carry parameters: 1 for binary, k for multinomial
        let r = if k == 2 { 1 } else { k };
        let np = r * d;
        let mut w = vec![0.0; np];
        let lam = 1.0 / c;
        let feat = |i: usize, j: usize| if j < m.d { m.at(i, j) } else { 1.0 };
        for _ in 0..200 {
            let mut g = vec![0.0; np];
            let mut h = vec![0.0; np * np];
            for i in 0..m.n {
                let mut z: Vec<f64> = (0..r).map(|a| (0..d).map(|j| w[a * d + j] * feat(i, j)).sum()).collect();
                let p: Vec<f64> = if k == 2 {
                    vec![1.0 / (1.0 + (-z[0]).exp())]
                } else {
                    softmax_rows(&mut z);
                    z
                };
                for a in 0..r {
                    let t = if k == 2 { (yi[i] == 1) as i64 as f64 } else { (yi[i] == a) as i64 as f64 };
                    for j in 0..d {
                        g[a * d + j] += (p[a] - t) * feat(i, j);
                    }
                    for b2 in 0..r {
                        let cov = if k == 2 { p[0] * (1.0 - p[0]) } else if a == b2 { p[a] * (1.0 - p[a]) } else { -p[a] * p[b2] };
                        for j in 0..d {
                            for l in 0..d {
                                h[(a * d + j) * np + b2 * d + l] += cov * feat(i, j) * feat(i, l);
                            }
                        }
                    }
                }
            }
            for a in 0..r {
                for j in 0..m.d {
                    g[a * d + j] += lam * w[a * d + j];
                    h[(a * d + j) * np + a * d + j] += lam;
                }
                // the intercepts are not penalized; a tiny ridge keeps the
                // multinomial Hessian (singular along equal shifts) invertible
                h[(a * d + m.d) * np + a * d + m.d] += 1e-12;
            }
            let gn = g.iter().map(|v| v.abs()).fold(0.0, f64::max);
            if gn < 1e-10 {
                break;
            }
            let step = gauss(h, g, np);
            for (wi, s) in w.iter_mut().zip(step) {
                *wi -= s;
            }
        }
        if k > 2 {
            // equal shifts of the intercepts change nothing: center them (as lbfgs from zero)
            let mean: f64 = (0..r).map(|a| w[a * d + m.d]).sum::<f64>() / r as f64;
            for a in 0..r {
                w[a * d + m.d] -= mean;
            }
        }
        let mut st = self.0.borrow_mut();
        st.w = (0..r).map(|a| w[a * d..(a + 1) * d].to_vec()).collect();
        st.classes = Some(classes);
        drop(st);
        Ok(self.clone())
    }
    fn fitted(&self) -> Result<(), PyErr> {
        if self.st().classes.is_none() {
            return Err(not_fitted("LogisticRegression"));
        }
        Ok(())
    }
    /// `model.predict_proba(X)`
    pub fn predict_proba(&self, x: &NdArray) -> Result<NdArray, PyErr> {
        self.fitted()?;
        let m = matrix(x)?;
        let s = self.st();
        let d = s.w[0].len() - 1;
        check_features(&m, d, "LogisticRegression")?;
        let k = s.classes.as_ref().unwrap().len();
        let mut out = Vec::with_capacity(m.n * k);
        for i in 0..m.n {
            let mut z: Vec<f64> = s.w.iter().map(|w| m.row(i).iter().zip(w).fold(0.0, |a, (x, c)| a + x * c) + w[d]).collect();
            if k == 2 {
                let p = 1.0 / (1.0 + (-z[0]).exp());
                out.extend([1.0 - p, p]);
            } else {
                softmax_rows(&mut z);
                out.extend(z);
            }
        }
        Ok(NdArray::floats(out, vec![m.n, k]))
    }
    /// `model.predict(X)`
    pub fn predict(&self, x: &NdArray) -> Result<NdArray, PyErr> {
        let p = self.predict_proba(x)?;
        let idx: Vec<usize> = p.argmax(Some(1))?.to_i64().into_iter().map(|i| i as usize).collect();
        take(self.st().classes.as_ref().unwrap(), &idx)
    }
    /// `model.score(X, y)`: accuracy
    pub fn score(&self, x: &NdArray, y: &NdArray) -> Result<f64, PyErr> {
        super::accuracy_score(y, &self.predict(x)?)
    }
    /// `model.coef_`
    pub fn coef_(&self) -> Result<NdArray, PyErr> {
        self.fitted()?;
        let s = self.st();
        let d = s.w[0].len() - 1;
        Ok(NdArray::floats(s.w.iter().flat_map(|w| w[..d].to_vec()).collect(), vec![s.w.len(), d]))
    }
    /// `model.intercept_`
    pub fn intercept_(&self) -> Result<NdArray, PyErr> {
        self.fitted()?;
        let s = self.st();
        let d = s.w[0].len() - 1;
        Ok(vec1(&s.w.iter().map(|w| w[d]).collect::<Vec<_>>()))
    }
    /// `model.classes_`
    pub fn classes_(&self) -> Result<NdArray, PyErr> {
        self.fitted()?;
        Ok(self.st().classes.clone().unwrap())
    }
}

// ------------------------------------------------------------ neighbors

#[derive(Default)]
pub struct KnnState {
    k: i64,
    x: Option<Mat>,
    y: Option<NdArray>,
}

/// The `k` nearest training rows of each query row (stable on ties).
fn neighbors(train: &Mat, q: &Mat, k: usize) -> Vec<Vec<usize>> {
    (0..q.n)
        .map(|i| {
            let mut d: Vec<(f64, usize)> = (0..train.n)
                .map(|t| (q.row(i).iter().zip(train.row(t)).fold(0.0, |s, (a, b)| s + (a - b) * (a - b)), t))
                .collect();
            d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
            d.into_iter().take(k).map(|p| p.1).collect()
        })
        .collect()
}

macro_rules! knn {
    ($name:ident) => {
        handle!($name, KnnState);
        impl $name {
            #[doc = concat!("`", stringify!($name), "(n_neighbors=5)`")]
            pub fn new(n_neighbors: i64) -> $name {
                $name(Rc::new(RefCell::new(KnnState { k: n_neighbors, ..Default::default() })))
            }
            /// `repr(model)`
            pub fn __repr__(&self) -> String {
                let k = self.st().k;
                repr(stringify!($name), if k == 5 { vec![] } else { vec![("n_neighbors", k.to_string())] })
            }
            /// `model.fit(X, y)`
            pub fn fit(&self, x: &NdArray, y: &NdArray) -> Result<$name, PyErr> {
                let m = matrix(x)?;
                vector(y, m.n)?;
                let k = self.st().k;
                if k <= 0 {
                    return Err(verr(format!("Expected n_neighbors > 0. Got {}", k)));
                }
                let mut st = self.0.borrow_mut();
                st.x = Some(m);
                st.y = Some(y.copy());
                drop(st);
                Ok(self.clone())
            }
            fn near(&self, x: &NdArray) -> Result<(Vec<Vec<usize>>, NdArray), PyErr> {
                let s = self.st();
                let Some(train) = &s.x else { return Err(not_fitted(stringify!($name))) };
                let q = matrix(x)?;
                check_features(&q, train.d, stringify!($name))?;
                let k = s.k as usize;
                if k > train.n {
                    return Err(verr(format!(
                        "Expected n_neighbors <= n_samples_fit, but n_neighbors = {}, n_samples_fit = {}, n_samples = {}",
                        k, train.n, q.n
                    )));
                }
                Ok((neighbors(train, &q, k), s.y.clone().unwrap()))
            }
        }
    };
}

knn!(KNeighborsClassifier);
knn!(KNeighborsRegressor);

impl KNeighborsClassifier {
    /// `model.predict(X)`: the most common label of the neighbors (the smallest on ties)
    pub fn predict(&self, x: &NdArray) -> Result<NdArray, PyErr> {
        let (nb, y) = self.near(x)?;
        let (classes, yi) = encode(&y)?;
        let idx: Vec<usize> = nb
            .iter()
            .map(|ns| {
                let mut counts = vec![0usize; classes.len()];
                for &t in ns {
                    counts[yi[t]] += 1;
                }
                let best = *counts.iter().max().unwrap();
                counts.iter().position(|&c| c == best).unwrap()
            })
            .collect();
        take(&classes, &idx)
    }
    /// `model.score(X, y)`: accuracy
    pub fn score(&self, x: &NdArray, y: &NdArray) -> Result<f64, PyErr> {
        super::accuracy_score(y, &self.predict(x)?)
    }
}

impl KNeighborsRegressor {
    /// `model.predict(X)`: the mean target of the neighbors
    pub fn predict(&self, x: &NdArray) -> Result<NdArray, PyErr> {
        let (nb, y) = self.near(x)?;
        let yv = y.to_f64();
        Ok(vec1(&nb.iter().map(|ns| ns.iter().fold(0.0, |s, &t| s + yv[t]) / ns.len() as f64).collect::<Vec<_>>()))
    }
    /// `model.score(X, y)`: R²
    pub fn score(&self, x: &NdArray, y: &NdArray) -> Result<f64, PyErr> {
        super::r2_score(y, &self.predict(x)?)
    }
}

// ------------------------------------------------------------ naive Bayes

#[derive(Default)]
pub struct NbState {
    smoothing: f64,
    classes: Option<NdArray>,
    theta: Vec<Vec<f64>>,
    var: Vec<Vec<f64>>,
    prior: Vec<f64>,
}
handle!(GaussianNB, NbState);

impl GaussianNB {
    /// `GaussianNB(var_smoothing=1e-09)`
    pub fn new(var_smoothing: f64) -> GaussianNB {
        GaussianNB(Rc::new(RefCell::new(NbState { smoothing: var_smoothing, ..Default::default() })))
    }
    /// `repr(model)`
    pub fn __repr__(&self) -> String {
        let v = self.st().smoothing;
        repr("GaussianNB", if v == 1e-9 { vec![] } else { vec![("var_smoothing", f(v))] })
    }
    /// `model.fit(X, y)`
    pub fn fit(&self, x: &NdArray, y: &NdArray) -> Result<GaussianNB, PyErr> {
        let m = matrix(x)?;
        vector(y, m.n)?;
        let (classes, yi) = encode(y)?;
        let eps = self.st().smoothing * m.col_vars().iter().cloned().fold(0.0, f64::max);
        let mut theta = vec![];
        let mut var = vec![];
        let mut prior = vec![];
        for c in 0..classes.len() {
            let rows: Vec<usize> = (0..m.n).filter(|&i| yi[i] == c).collect();
            let sub = Mat { n: rows.len(), d: m.d, v: rows.iter().flat_map(|&i| m.row(i).to_vec()).collect() };
            theta.push(sub.col_means());
            var.push(sub.col_vars().into_iter().map(|v| v + eps).collect());
            prior.push(rows.len() as f64 / m.n as f64);
        }
        let mut st = self.0.borrow_mut();
        st.classes = Some(classes);
        st.theta = theta;
        st.var = var;
        st.prior = prior;
        drop(st);
        Ok(self.clone())
    }
    fn jll(&self, x: &NdArray) -> Result<Mat, PyErr> {
        let s = self.st();
        if s.classes.is_none() {
            return Err(not_fitted("GaussianNB"));
        }
        let m = matrix(x)?;
        check_features(&m, s.theta[0].len(), "GaussianNB")?;
        let k = s.theta.len();
        let mut out = Mat::zeros(m.n, k);
        for c in 0..k {
            let norm: f64 = -0.5 * s.var[c].iter().map(|v| (2.0 * std::f64::consts::PI * v).ln()).sum::<f64>();
            for i in 0..m.n {
                let q: f64 = (0..m.d).map(|j| (m.at(i, j) - s.theta[c][j]).powi(2) / s.var[c][j]).sum();
                out.v[i * k + c] = s.prior[c].ln() + norm - 0.5 * q;
            }
        }
        Ok(out)
    }
    /// `model.predict_proba(X)`
    pub fn predict_proba(&self, x: &NdArray) -> Result<NdArray, PyErr> {
        let mut j = self.jll(x)?;
        for i in 0..j.n {
            let r = &mut j.v[i * j.d..(i + 1) * j.d];
            softmax_rows(r);
        }
        Ok(j.to_array())
    }
    /// `model.predict(X)`
    pub fn predict(&self, x: &NdArray) -> Result<NdArray, PyErr> {
        let j = self.jll(x)?;
        let idx: Vec<usize> = (0..j.n)
            .map(|i| {
                let r = j.row(i);
                (0..r.len()).fold(0, |b, c| if r[c] > r[b] { c } else { b })
            })
            .collect();
        take(self.st().classes.as_ref().unwrap(), &idx)
    }
    /// `model.score(X, y)`: accuracy
    pub fn score(&self, x: &NdArray, y: &NdArray) -> Result<f64, PyErr> {
        super::accuracy_score(y, &self.predict(x)?)
    }
    /// `model.theta_`: the class means
    pub fn theta_(&self) -> Result<NdArray, PyErr> {
        let s = self.st();
        if s.classes.is_none() {
            return Err(not_fitted("GaussianNB"));
        }
        Ok(NdArray::floats(s.theta.concat(), vec![s.theta.len(), s.theta[0].len()]))
    }
    /// `model.class_prior_`
    pub fn class_prior_(&self) -> Result<NdArray, PyErr> {
        self.theta_()?;
        Ok(vec1(&self.st().prior))
    }
}

// ------------------------------------------------------------ k-means

#[derive(Default)]
pub struct KMeansState {
    k: i64,
    init: Option<Mat>,
    max_iter: i64,
    tol: f64,
    centers: Option<Mat>,
    labels: Vec<usize>,
    inertia: f64,
    n_iter: i64,
}
handle!(KMeans, KMeansState);

fn nearest(x: &[f64], c: &Mat) -> (usize, f64) {
    let mut best = (0, f64::INFINITY);
    for j in 0..c.n {
        let d = x.iter().zip(c.row(j)).fold(0.0, |s, (a, b)| s + (a - b) * (a - b));
        if d < best.1 {
            best = (j, d);
        }
    }
    best
}

impl KMeans {
    /// `KMeans(n_clusters, init=array, n_init=1, max_iter=300, tol=1e-4)`
    pub fn new(n_clusters: i64, init: &NdArray, max_iter: i64, tol: f64) -> Result<KMeans, PyErr> {
        let i = matrix(init)?;
        if i.n as i64 != n_clusters {
            return Err(verr(format!(
                "The shape of the initial centers {} does not match the number of clusters {}.",
                super::super::numpy::shape_err(init.dims()),
                n_clusters
            )));
        }
        Ok(KMeans(Rc::new(RefCell::new(KMeansState { k: n_clusters, init: Some(i), max_iter, tol, ..Default::default() }))))
    }
    /// `repr(model)`
    pub fn __repr__(&self) -> String {
        let s = self.st();
        let mut p = vec![("n_clusters", s.k.to_string()), ("n_init", "1".to_string())];
        if let Some(i) = &s.init {
            p.push(("init", format!("{:?}", i.to_array())));
        }
        if s.max_iter != 300 {
            p.push(("max_iter", s.max_iter.to_string()));
        }
        if s.tol != 1e-4 {
            p.push(("tol", f(s.tol)));
        }
        repr("KMeans", p)
    }
    /// `model.fit(X)`: Lloyd's algorithm from the given centers
    pub fn fit(&self, x: &NdArray) -> Result<KMeans, PyErr> {
        let m = matrix(x)?;
        let (mut centers, max_iter, tol0) = {
            let s = self.st();
            (s.init.clone().unwrap(), s.max_iter, s.tol)
        };
        check_features(&m, centers.d, "KMeans")?;
        if (m.n as i64) < centers.n as i64 {
            return Err(verr(format!("n_samples={} should be >= n_clusters={}.", m.n, centers.n)));
        }
        let vars = m.col_vars();
        let tol = vars.iter().sum::<f64>() / vars.len() as f64 * tol0;
        let k = centers.n;
        let mut labels: Vec<usize> = vec![usize::MAX; m.n];
        let mut strict = false;
        let mut iters = 0;
        for it in 0..max_iter {
            iters = it + 1;
            let new_labels: Vec<usize> = (0..m.n).map(|i| nearest(m.row(i), &centers).0).collect();
            let mut sums = Mat::zeros(k, m.d);
            let mut counts = vec![0usize; k];
            for i in 0..m.n {
                counts[new_labels[i]] += 1;
                for j in 0..m.d {
                    sums.v[new_labels[i] * m.d + j] += m.at(i, j);
                }
            }
            let mut new_centers = centers.clone();
            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..m.d {
                        new_centers.v[c * m.d + j] = sums.v[c * m.d + j] / counts[c] as f64;
                    }
                }
            }
            let shift: f64 = centers.v.iter().zip(&new_centers.v).map(|(a, b)| (a - b) * (a - b)).sum();
            centers = new_centers;
            if new_labels == labels {
                strict = true;
                break;
            }
            labels = new_labels;
            if shift <= tol {
                break;
            }
        }
        if !strict {
            labels = (0..m.n).map(|i| nearest(m.row(i), &centers).0).collect();
        }
        let inertia: f64 = (0..m.n).map(|i| {
            let c = centers.row(labels[i]);
            m.row(i).iter().zip(c).fold(0.0, |s, (a, b)| s + (a - b) * (a - b))
        }).sum();
        let mut st = self.0.borrow_mut();
        st.centers = Some(centers);
        st.labels = labels;
        st.inertia = inertia;
        st.n_iter = iters;
        drop(st);
        Ok(self.clone())
    }
    /// `model.predict(X)`
    pub fn predict(&self, x: &NdArray) -> Result<NdArray, PyErr> {
        let s = self.st();
        let Some(c) = &s.centers else { return Err(not_fitted("KMeans")) };
        let m = matrix(x)?;
        check_features(&m, c.d, "KMeans")?;
        Ok(NdArray::ints((0..m.n).map(|i| nearest(m.row(i), c).0 as i64).collect(), vec![m.n]))
    }
    /// `model.cluster_centers_`
    pub fn cluster_centers_(&self) -> Result<NdArray, PyErr> {
        self.st().centers.as_ref().map(|c| c.to_array()).ok_or_else(|| not_fitted("KMeans"))
    }
    /// `model.labels_`
    pub fn labels_(&self) -> Result<NdArray, PyErr> {
        self.cluster_centers_()?;
        let l = &self.st().labels;
        Ok(NdArray::ints(l.iter().map(|&v| v as i64).collect(), vec![l.len()]))
    }
    /// `model.inertia_`
    pub fn inertia_(&self) -> Result<NdArray, PyErr> {
        self.cluster_centers_()?;
        Ok(scalar(self.st().inertia))
    }
    /// `model.n_iter_`
    pub fn n_iter_(&self) -> Result<i64, PyErr> {
        self.cluster_centers_()?;
        Ok(self.st().n_iter)
    }
}

// ------------------------------------------------------------ PCA

#[derive(Default)]
pub struct PcaState {
    k: Option<i64>,
    mean: Option<Vec<f64>>,
    components: Mat,
    sv: Vec<f64>,
    var: Vec<f64>,
    ratio: Vec<f64>,
}
handle!(PCA, PcaState);

impl PCA {
    /// `PCA(n_components=None)`
    pub fn new(n_components: Option<i64>) -> PCA {
        PCA(Rc::new(RefCell::new(PcaState { k: n_components, ..Default::default() })))
    }
    /// `repr(model)`
    pub fn __repr__(&self) -> String {
        match self.st().k {
            Some(k) => repr("PCA", vec![("n_components", k.to_string())]),
            None => "PCA()".into(),
        }
    }
    /// `model.fit(X)`: SVD of the centered data; each component's largest
    /// coordinate (in absolute value) positive, as scikit-learn's `svd_flip`
    pub fn fit(&self, x: &NdArray) -> Result<PCA, PyErr> {
        let m = matrix(x)?;
        let kmax = m.n.min(m.d);
        let k = match self.st().k {
            None => kmax,
            Some(k) if k >= 1 && k as usize <= kmax => k as usize,
            Some(k) => {
                return Err(verr(format!(
                    "n_components={} must be between 0 and min(n_samples, n_features)={} with svd_solver='covariance_eigh'",
                    k, kmax
                )))
            }
        };
        let mean = m.col_means();
        let mut c = m.clone();
        for i in 0..m.n {
            for j in 0..m.d {
                c.v[i * m.d + j] -= mean[j];
            }
        }
        let (_, s, mut vt) = svd(&c);
        for r in 0..vt.n {
            let row = &mut vt.v[r * vt.d..(r + 1) * vt.d];
            let big = (0..row.len()).fold(0, |b, j| if row[j].abs() > row[b].abs() { j } else { b });
            if row[big] < 0.0 {
                for v in row.iter_mut() {
                    *v = -*v;
                }
            }
        }
        let var: Vec<f64> = s.iter().map(|v| v * v / (m.n as f64 - 1.0)).collect();
        let total: f64 = var.iter().sum();
        let mut st = self.0.borrow_mut();
        st.components = Mat { n: k, d: m.d, v: vt.v[..k * m.d].to_vec() };
        st.ratio = var[..k].iter().map(|v| v / total).collect();
        st.var = var[..k].to_vec();
        st.sv = s[..k].to_vec();
        st.mean = Some(mean);
        drop(st);
        Ok(self.clone())
    }
    /// `model.transform(X)`
    pub fn transform(&self, x: &NdArray) -> Result<NdArray, PyErr> {
        let s = self.st();
        let Some(mean) = &s.mean else { return Err(not_fitted("PCA")) };
        let m = matrix(x)?;
        check_features(&m, mean.len(), "PCA")?;
        let k = s.components.n;
        let mut out = Mat::zeros(m.n, k);
        for i in 0..m.n {
            for c in 0..k {
                out.v[i * k + c] = (0..m.d).fold(0.0, |a, j| a + (m.at(i, j) - mean[j]) * s.components.at(c, j));
            }
        }
        Ok(out.to_array())
    }
    /// `model.fit_transform(X)`
    pub fn fit_transform(&self, x: &NdArray) -> Result<NdArray, PyErr> {
        self.fit(x)?;
        self.transform(x)
    }
    fn fitted(&self) -> Result<(), PyErr> {
        self.st().mean.as_ref().map(|_| ()).ok_or_else(|| not_fitted("PCA"))
    }
    /// `model.components_`
    pub fn components_(&self) -> Result<NdArray, PyErr> {
        self.fitted()?;
        Ok(self.st().components.to_array())
    }
    /// `model.explained_variance_`
    pub fn explained_variance_(&self) -> Result<NdArray, PyErr> {
        self.fitted()?;
        Ok(vec1(&self.st().var))
    }
    /// `model.explained_variance_ratio_`
    pub fn explained_variance_ratio_(&self) -> Result<NdArray, PyErr> {
        self.fitted()?;
        Ok(vec1(&self.st().ratio))
    }
    /// `model.singular_values_`
    pub fn singular_values_(&self) -> Result<NdArray, PyErr> {
        self.fitted()?;
        Ok(vec1(&self.st().sv))
    }
    /// `model.mean_`
    pub fn mean_(&self) -> Result<NdArray, PyErr> {
        self.fitted()?;
        Ok(vec1(self.st().mean.as_ref().unwrap()))
    }
}

//! `np.dot`, `@`, `np.outer` and `np.linalg` (`inv`, `solve`, `det`, `norm`).
//! Floating-point products sum in order (NumPy's BLAS may differ in the last bits).

use super::reduce::dot_lane;
use super::{shape_err, verr, Data, Kind, NdArray, Scalar};
use crate::pyrt::err::PyErr;

const LINALG_ERROR: &[&str] = &["LinAlgError", "ValueError", "Exception", "BaseException"];

fn lin_err(msg: &str) -> PyErr {
    PyErr::new(LINALG_ERROR, msg.to_string())
}

fn out(items: Vec<Scalar>, k: Kind, shape: Vec<usize>) -> NdArray {
    let mut r = NdArray::from_data(Data::from_scalars(k, items.into_iter()), shape);
    r.scalar = r.shape.is_empty();
    r
}

/// (n, k) @ (k, m) on row-major element lists.
fn mm(a: &[Scalar], b: &[Scalar], n: usize, kk: usize, m: usize, k: Kind) -> Vec<Scalar> {
    let mut r = Vec::with_capacity(n * m);
    for i in 0..n {
        let row = &a[i * kk..(i + 1) * kk];
        for j in 0..m {
            let col: Vec<Scalar> = (0..kk).map(|t| b[t * m + j]).collect();
            r.push(dot_lane(row, &col, k));
        }
    }
    r
}

/// `a @ b` (`np.matmul`)
pub fn matmul(a: &NdArray, b: &NdArray) -> Result<NdArray, PyErr> {
    if a.shape.is_empty() || b.shape.is_empty() {
        return Err(verr("matmul: Input operand 0 does not have enough dimensions (has 0, gufunc core with signature (n?,k),(k,m?)->(n?,m?) requires 1)"));
    }
    let k = a.kind().max(b.kind());
    let (a2, a1) = if a.shape.len() == 1 { (super::ops::expand_dims(a, 0), true) } else { (a.clone(), false) };
    let (b2, b1) = if b.shape.len() == 1 { (super::ops::expand_dims(b, 1), true) } else { (b.clone(), false) };
    let (n, ka) = (a2.shape[a2.shape.len() - 2], a2.shape[a2.shape.len() - 1]);
    let (kb, m) = (b2.shape[b2.shape.len() - 2], b2.shape[b2.shape.len() - 1]);
    if ka != kb {
        return Err(verr(format!(
            "matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size {} is different from {})",
            kb, ka
        )));
    }
    let batch = super::ops::broadcast_shapes(&a2.shape[..a2.shape.len() - 2], &b2.shape[..b2.shape.len() - 2])?;
    let nb: usize = batch.iter().product();
    let mut ashape = batch.clone();
    ashape.extend([n, ka]);
    let mut bshape = batch.clone();
    bshape.extend([kb, m]);
    let ai = super::ops::broadcast_items(&super::ops::Arg::A(a2), &ashape);
    let bi = super::ops::broadcast_items(&super::ops::Arg::A(b2), &bshape);
    let mut items = Vec::with_capacity(nb * n * m);
    for t in 0..nb {
        items.extend(mm(&ai[t * n * ka..(t + 1) * n * ka], &bi[t * ka * m..(t + 1) * ka * m], n, ka, m, k));
    }
    let mut shape = batch;
    if !a1 {
        shape.push(n);
    }
    if !b1 {
        shape.push(m);
    }
    Ok(out(items, k, shape))
}

/// `np.dot(a, b)`
pub fn dot(a: &NdArray, b: &NdArray) -> Result<NdArray, PyErr> {
    if a.shape.is_empty() || b.shape.is_empty() {
        return super::ops::binary(super::ops::Op::Mul, &super::ops::Arg::A(a.clone()), &super::ops::Arg::A(b.clone()));
    }
    let ka = *a.shape.last().unwrap();
    let kb = if b.shape.len() == 1 { b.shape[0] } else { b.shape[b.shape.len() - 2] };
    if ka != kb {
        return Err(verr(format!(
            "shapes {} and {} not aligned: {} (dim {}) != {} (dim {})",
            shape_err(&a.shape),
            shape_err(&b.shape),
            ka,
            a.shape.len() - 1,
            kb,
            if b.shape.len() == 1 { 0 } else { b.shape.len() - 2 }
        )));
    }
    if a.shape.len() <= 2 && b.shape.len() <= 2 {
        return matmul(a, b);
    }
    Err(verr("ruai: np.dot of arrays with more than 2 dimensions is not supported (use matmul)"))
}

/// `np.outer(a, b)`
pub fn outer(a: &NdArray, b: &NdArray) -> NdArray {
    let (x, y) = (a.scalars(), b.scalars());
    let k = a.kind().max(b.kind());
    let mut items = Vec::with_capacity(x.len() * y.len());
    for &p in &x {
        for &q in &y {
            items.push(dot_lane(&[p], &[q], k));
        }
    }
    out(items, k, vec![x.len(), y.len()])
}

fn square(a: &NdArray) -> Result<(usize, Vec<f64>), PyErr> {
    if a.shape.len() < 2 {
        return Err(lin_err(&format!(
            "{}-dimensional array given. Array must be at least two-dimensional",
            a.shape.len()
        )));
    }
    if a.shape.len() > 2 {
        return Err(verr("ruai: stacked matrices are not supported by np.linalg"));
    }
    if a.shape[0] != a.shape[1] {
        return Err(lin_err("Last 2 dimensions of the array must be square"));
    }
    Ok((a.shape[0], a.to_f64()))
}

/// LU with partial pivoting: (LU, permutation sign), None when singular.
fn lu(n: usize, mut m: Vec<f64>, rhs: &mut [Vec<f64>]) -> Option<(Vec<f64>, f64)> {
    let mut sign = 1.0;
    for c in 0..n {
        let mut p = c;
        for r in c + 1..n {
            if m[r * n + c].abs() > m[p * n + c].abs() {
                p = r;
            }
        }
        if m[p * n + c] == 0.0 {
            return None;
        }
        if p != c {
            for j in 0..n {
                m.swap(c * n + j, p * n + j);
            }
            for v in rhs.iter_mut() {
                v.swap(c, p);
            }
            sign = -sign;
        }
        for r in c + 1..n {
            let f = m[r * n + c] / m[c * n + c];
            m[r * n + c] = f;
            for j in c + 1..n {
                m[r * n + j] -= f * m[c * n + j];
            }
            for v in rhs.iter_mut() {
                v[r] -= f * v[c];
            }
        }
    }
    Some((m, sign))
}

fn back(n: usize, lu: &[f64], v: &mut [f64]) {
    for r in (0..n).rev() {
        let mut s = v[r];
        for j in r + 1..n {
            s -= lu[r * n + j] * v[j];
        }
        v[r] = s / lu[r * n + r];
    }
}

/// `np.linalg.inv(a)`
pub fn inv(a: &NdArray) -> Result<NdArray, PyErr> {
    let (n, m) = square(a)?;
    let mut cols: Vec<Vec<f64>> = (0..n).map(|j| (0..n).map(|i| (i == j) as i64 as f64).collect()).collect();
    let (lu, _) = lu(n, m, &mut cols).ok_or_else(|| lin_err("Singular matrix"))?;
    for c in cols.iter_mut() {
        back(n, &lu, c);
    }
    let mut v = vec![0.0; n * n];
    for (j, c) in cols.iter().enumerate() {
        for i in 0..n {
            v[i * n + j] = c[i];
        }
    }
    Ok(NdArray::floats(v, vec![n, n]))
}

/// `np.linalg.solve(a, b)` for a vector or a matrix `b`
pub fn solve(a: &NdArray, b: &NdArray) -> Result<NdArray, PyErr> {
    let (n, m) = square(a)?;
    let bv = b.to_f64();
    let (rows, k) = match b.shape.len() {
        1 => (b.shape[0], 1),
        2 => (b.shape[0], b.shape[1]),
        _ => return Err(verr("ruai: np.linalg.solve needs a 1-d or 2-d b")),
    };
    if rows != n {
        return Err(verr(format!(
            "solve: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (m,m),(m,n)->(m,n) (size {} is different from {})",
            rows, n
        )));
    }
    let mut cols: Vec<Vec<f64>> = (0..k).map(|j| (0..n).map(|i| bv[i * k + j]).collect()).collect();
    let (lu, _) = lu(n, m, &mut cols).ok_or_else(|| lin_err("Singular matrix"))?;
    for c in cols.iter_mut() {
        back(n, &lu, c);
    }
    let mut v = vec![0.0; n * k];
    for (j, c) in cols.iter().enumerate() {
        for i in 0..n {
            v[i * k + j] = c[i];
        }
    }
    Ok(NdArray::floats(v, b.shape.clone()))
}

/// `np.linalg.det(a)`
pub fn det(a: &NdArray) -> Result<NdArray, PyErr> {
    let (n, m) = square(a)?;
    let d = match lu(n, m, &mut []) {
        None => 0.0,
        Some((lu, sign)) => (0..n).fold(sign, |p, i| p * lu[i * n + i]),
    };
    Ok(NdArray::from_scalar(Scalar::F(d)))
}

/// `np.linalg.norm(a)`: the 2-norm of a vector, the Frobenius norm of a matrix.
pub fn norm(a: &NdArray) -> NdArray {
    let s = a.to_f64().iter().fold(0.0, |s, x| s + x * x);
    NdArray::from_scalar(Scalar::F(s.sqrt()))
}

/// `np.trace(a)`
pub fn trace(a: &NdArray) -> Result<NdArray, PyErr> {
    if a.shape.len() != 2 {
        return Err(verr("ruai: np.trace needs a 2-d array"));
    }
    let k = if a.kind() == Kind::Bool { Kind::Int } else { a.kind() };
    let items = a.scalars();
    let n = a.shape[0].min(a.shape[1]);
    let diag: Vec<Scalar> = (0..n).map(|i| items[i * a.shape[1] + i]).collect();
    let ones = vec![Scalar::I(1); n];
    Ok(out(vec![dot_lane(&diag, &ones, k)], k, vec![]))
}

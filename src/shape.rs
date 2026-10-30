//! Shape helpers (all storage is contiguous, row-major).

/// Number of elements in `shape` (1 for a scalar `[]`).
pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Row-major strides of a contiguous tensor.
pub fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// NumPy-style broadcast of two shapes.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let rank = a.len().max(b.len());
    let mut out = vec![0; rank];
    for i in 0..rank {
        let da = if i < rank - a.len() { 1 } else { a[i - (rank - a.len())] };
        let db = if i < rank - b.len() { 1 } else { b[i - (rank - b.len())] };
        out[i] = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            panic!("shapes {a:?} and {b:?} are not broadcastable");
        };
    }
    out
}

/// Strides for reading `shape` as if broadcast to `out_shape` (0 on broadcast dims).
pub fn broadcast_strides(shape: &[usize], out_shape: &[usize]) -> Vec<usize> {
    let rank = out_shape.len();
    let st = strides(shape);
    let offset = rank - shape.len();
    (0..rank)
        .map(|i| {
            if i < offset || shape[i - offset] == 1 {
                0
            } else {
                st[i - offset]
            }
        })
        .collect()
}

/// Normalize a possibly-negative dimension index.
pub fn norm_dim(dim: isize, rank: usize) -> usize {
    let r = rank as isize;
    let d = if dim < 0 { dim + r } else { dim };
    assert!(d >= 0 && d < r.max(1), "dim {dim} out of range for rank {rank}");
    d as usize
}

/// Split a shape around `axis` into (outer, len, inner) for reductions.
pub fn split_axis(shape: &[usize], axis: usize) -> (usize, usize, usize) {
    let outer = shape[..axis].iter().product();
    let len = shape[axis];
    let inner = shape[axis + 1..].iter().product();
    (outer, len, inner)
}

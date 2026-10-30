//! CPU backend: plain `Vec<f32>` storage, parallelized with rayon.

use super::{Backend, BinaryOp, Conv2dGeom, MatmulSpec, ReduceOp, Storage, UnaryOp};
use crate::device::Device;
use crate::shape::{broadcast_strides, numel, split_axis, strides};
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::Arc;

/// Below this many elements, work is done on the calling thread.
const PAR_THRESHOLD: usize = 1 << 14;

pub struct CpuBackend;

fn data(s: &Storage) -> &[f32] {
    match s {
        Storage::Cpu(v) => v.as_slice(),
        #[allow(unreachable_patterns)]
        _ => panic!("CPU backend received a non-CPU storage ({})", s.device()),
    }
}

fn wrap(v: Vec<f32>) -> Storage {
    Storage::Cpu(Arc::new(v))
}

fn map_into<F: Fn(f32) -> f32 + Sync + Send>(x: &[f32], f: F) -> Vec<f32> {
    if x.len() >= PAR_THRESHOLD {
        x.par_iter().map(|&v| f(v)).collect()
    } else {
        x.iter().map(|&v| f(v)).collect()
    }
}

/// Fill `out` row by row (rows of length `inner`), with `f(row_index, row)`.
fn for_rows<F: Fn(usize, &mut [f32]) + Sync + Send>(out: &mut [f32], inner: usize, f: F) {
    if out.is_empty() || inner == 0 {
        return;
    }
    if out.len() >= PAR_THRESHOLD {
        let min_rows = (PAR_THRESHOLD / inner).max(1);
        out.par_chunks_mut(inner)
            .with_min_len(min_rows)
            .enumerate()
            .for_each(|(r, row)| f(r, row));
    } else {
        out.chunks_mut(inner).enumerate().for_each(|(r, row)| f(r, row));
    }
}

/// Offsets into two broadcast inputs for output row `row` (all dims but the last).
#[inline]
fn row_offsets(row: usize, out_shape: &[usize], sa: &[usize], sb: &[usize]) -> (usize, usize) {
    let mut rem = row;
    let (mut oa, mut ob) = (0, 0);
    for d in (0..out_shape.len() - 1).rev() {
        let i = rem % out_shape[d];
        rem /= out_shape[d];
        oa += i * sa[d];
        ob += i * sb[d];
    }
    (oa, ob)
}

fn broadcast_apply<F: Fn(f32, f32) -> f32 + Sync + Send>(
    f: F,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    out_shape: &[usize],
) -> Vec<f32> {
    let n = numel(out_shape);
    if n == 0 {
        return vec![];
    }
    if a_shape == b_shape {
        return if n >= PAR_THRESHOLD {
            a.par_iter().zip(b.par_iter()).map(|(&x, &y)| f(x, y)).collect()
        } else {
            a.iter().zip(b).map(|(&x, &y)| f(x, y)).collect()
        };
    }
    if b.len() == 1 && a.len() == n {
        let y = b[0];
        return map_into(a, |x| f(x, y));
    }
    if a.len() == 1 && b.len() == n {
        let x = a[0];
        return map_into(b, |y| f(x, y));
    }
    let rank = out_shape.len();
    let sa = broadcast_strides(a_shape, out_shape);
    let sb = broadcast_strides(b_shape, out_shape);
    let inner = out_shape[rank - 1];
    let (ia, ib) = (sa[rank - 1], sb[rank - 1]);
    let mut out = vec![0.0; n];
    for_rows(&mut out, inner, |r, row| {
        let (oa, ob) = row_offsets(r, out_shape, &sa, &sb);
        for (j, o) in row.iter_mut().enumerate() {
            *o = f(a[oa + j * ia], b[ob + j * ib]);
        }
    });
    out
}

#[inline]
fn gelu(x: f32) -> f32 {
    const S: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (S * (x + 0.044715 * x * x * x)).tanh())
}

#[inline]
fn gelu_grad(x: f32) -> f32 {
    const S: f32 = 0.797_884_6;
    let u = S * (x + 0.044715 * x * x * x);
    let t = u.tanh();
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * S * (1.0 + 3.0 * 0.044715 * x * x)
}

/// Transpose each `[rows, cols]` block of `x` into `[cols, rows]`.
fn transpose_blocks(x: &[f32], blocks: usize, rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; blocks * rows * cols];
    let sz = rows * cols;
    out.par_chunks_mut(sz.max(1)).enumerate().for_each(|(bi, ob)| {
        let ib = &x[bi * sz..(bi + 1) * sz];
        for r in 0..rows {
            for c in 0..cols {
                ob[c * rows + r] = ib[r * cols + c];
            }
        }
    });
    out
}

impl Backend for CpuBackend {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn from_vec(&self, data: Vec<f32>) -> Storage {
        wrap(data)
    }

    fn to_vec(&self, x: &Storage) -> Vec<f32> {
        data(x).to_vec()
    }

    fn full(&self, len: usize, value: f32) -> Storage {
        wrap(vec![value; len])
    }

    fn unary(&self, op: UnaryOp, x: &Storage) -> Storage {
        let x = data(x);
        let out = match op {
            UnaryOp::Neg => map_into(x, |v| -v),
            UnaryOp::Exp => map_into(x, f32::exp),
            UnaryOp::Log => map_into(x, f32::ln),
            UnaryOp::Sqrt => map_into(x, f32::sqrt),
            UnaryOp::Recip => map_into(x, |v| 1.0 / v),
            UnaryOp::Abs => map_into(x, f32::abs),
            UnaryOp::Tanh => map_into(x, f32::tanh),
            UnaryOp::Sigmoid => map_into(x, |v| 1.0 / (1.0 + (-v).exp())),
            UnaryOp::Relu => map_into(x, |v| v.max(0.0)),
            UnaryOp::Gelu => map_into(x, gelu),
            UnaryOp::GeluGrad => map_into(x, gelu_grad),
            UnaryOp::Sin => map_into(x, f32::sin),
            UnaryOp::Cos => map_into(x, f32::cos),
        };
        wrap(out)
    }

    fn binary(
        &self,
        op: BinaryOp,
        a: &Storage,
        a_shape: &[usize],
        b: &Storage,
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> Storage {
        let (a, b) = (data(a), data(b));
        let (s1, s2) = (a_shape, b_shape);
        let out = match op {
            BinaryOp::Add => broadcast_apply(|x, y| x + y, a, s1, b, s2, out_shape),
            BinaryOp::Sub => broadcast_apply(|x, y| x - y, a, s1, b, s2, out_shape),
            BinaryOp::Mul => broadcast_apply(|x, y| x * y, a, s1, b, s2, out_shape),
            BinaryOp::Div => broadcast_apply(|x, y| x / y, a, s1, b, s2, out_shape),
            BinaryOp::Pow => broadcast_apply(f32::powf, a, s1, b, s2, out_shape),
            BinaryOp::Maximum => broadcast_apply(f32::max, a, s1, b, s2, out_shape),
            BinaryOp::Minimum => broadcast_apply(f32::min, a, s1, b, s2, out_shape),
            BinaryOp::Eq => broadcast_apply(|x, y| (x == y) as u8 as f32, a, s1, b, s2, out_shape),
            BinaryOp::Gt => broadcast_apply(|x, y| (x > y) as u8 as f32, a, s1, b, s2, out_shape),
            BinaryOp::Lt => broadcast_apply(|x, y| (x < y) as u8 as f32, a, s1, b, s2, out_shape),
        };
        wrap(out)
    }

    fn expand(&self, x: &Storage, shape: &[usize], out_shape: &[usize]) -> Storage {
        let x = data(x);
        let n = numel(out_shape);
        if shape == out_shape {
            return wrap(x.to_vec());
        }
        if x.len() == 1 {
            return wrap(vec![x[0]; n]);
        }
        let rank = out_shape.len();
        let sx = broadcast_strides(shape, out_shape);
        let inner = out_shape[rank - 1];
        let ix = sx[rank - 1];
        let mut out = vec![0.0; n];
        for_rows(&mut out, inner, |r, row| {
            let (ox, _) = row_offsets(r, out_shape, &sx, &sx);
            for (j, o) in row.iter_mut().enumerate() {
                *o = x[ox + j * ix];
            }
        });
        wrap(out)
    }

    fn reduce(&self, op: ReduceOp, x: &Storage, shape: &[usize], axis: usize) -> Storage {
        let x = data(x);
        let (outer, len, inner) = split_axis(shape, axis);
        let mut out = vec![0.0; outer * inner];
        if out.is_empty() {
            return wrap(out);
        }
        let reduce_block = |o: usize, dst: &mut [f32]| {
            let base = o * len * inner;
            match op {
                ReduceOp::Sum => {
                    dst.fill(0.0);
                    for l in 0..len {
                        let src = &x[base + l * inner..base + (l + 1) * inner];
                        for (d, s) in dst.iter_mut().zip(src) {
                            *d += s;
                        }
                    }
                }
                ReduceOp::Max | ReduceOp::Min => {
                    let init = if op == ReduceOp::Max { f32::NEG_INFINITY } else { f32::INFINITY };
                    dst.fill(init);
                    for l in 0..len {
                        let src = &x[base + l * inner..base + (l + 1) * inner];
                        for (d, &s) in dst.iter_mut().zip(src) {
                            *d = if op == ReduceOp::Max { d.max(s) } else { d.min(s) };
                        }
                    }
                }
                ReduceOp::ArgMax => {
                    for (i, d) in dst.iter_mut().enumerate() {
                        let mut best = f32::NEG_INFINITY;
                        let mut bi = 0;
                        for l in 0..len {
                            let v = x[base + l * inner + i];
                            if v > best {
                                best = v;
                                bi = l;
                            }
                        }
                        *d = bi as f32;
                    }
                }
            }
        };
        if x.len() >= PAR_THRESHOLD {
            out.par_chunks_mut(inner)
                .with_min_len((PAR_THRESHOLD / (len * inner).max(1)).max(1))
                .enumerate()
                .for_each(|(o, d)| reduce_block(o, d));
        } else {
            out.chunks_mut(inner).enumerate().for_each(|(o, d)| reduce_block(o, d));
        }
        wrap(out)
    }

    fn matmul(&self, a: &Storage, b: &Storage, spec: MatmulSpec) -> Storage {
        let MatmulSpec { batch, m, k, n, trans_a, trans_b, a_batch_stride, b_batch_stride } = spec;
        let (a, b) = (data(a), data(b));
        let a_blocks = if a_batch_stride == 0 { 1 } else { batch };
        let b_blocks = if b_batch_stride == 0 { 1 } else { batch };
        // Bring A to [m, k] and B to [k, n] row-major blocks.
        let a: Cow<[f32]> = if trans_a {
            Cow::Owned(transpose_blocks(&a[..a_blocks * m * k], a_blocks, k, m))
        } else {
            Cow::Borrowed(a)
        };
        let b: Cow<[f32]> = if trans_b {
            Cow::Owned(transpose_blocks(&b[..b_blocks * k * n], b_blocks, n, k))
        } else {
            Cow::Borrowed(b)
        };
        let a_bs = if a_batch_stride == 0 { 0 } else { m * k };
        let b_bs = if b_batch_stride == 0 { 0 } else { k * n };
        let mut out = vec![0.0f32; batch * m * n];
        if out.is_empty() {
            return wrap(out);
        }
        let kernel = |row: usize, crow: &mut [f32]| {
            let bi = row / m;
            let i = row % m;
            let arow = &a[bi * a_bs + i * k..bi * a_bs + (i + 1) * k];
            let bm = &b[bi * b_bs..bi * b_bs + k * n];
            for (p, &av) in arow.iter().enumerate() {
                if av == 0.0 {
                    continue;
                }
                let brow = &bm[p * n..(p + 1) * n];
                for (c, &bv) in crow.iter_mut().zip(brow) {
                    *c += av * bv;
                }
            }
        };
        if batch * m * n * k >= PAR_THRESHOLD {
            let min_rows = (PAR_THRESHOLD / (n * k).max(1)).max(1);
            out.par_chunks_mut(n)
                .with_min_len(min_rows)
                .enumerate()
                .for_each(|(r, c)| kernel(r, c));
        } else {
            out.chunks_mut(n).enumerate().for_each(|(r, c)| kernel(r, c));
        }
        wrap(out)
    }

    fn permute(&self, x: &Storage, shape: &[usize], perm: &[usize]) -> Storage {
        let x = data(x);
        let st = strides(shape);
        let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
        let in_st: Vec<usize> = perm.iter().map(|&p| st[p]).collect();
        let n = numel(shape);
        let mut out = vec![0.0; n];
        if n == 0 {
            return wrap(out);
        }
        if out_shape.is_empty() {
            return wrap(x.to_vec());
        }
        let rank = out_shape.len();
        let inner = out_shape[rank - 1];
        let ix = in_st[rank - 1];
        for_rows(&mut out, inner, |r, row| {
            let (ox, _) = row_offsets(r, &out_shape, &in_st, &in_st);
            for (j, o) in row.iter_mut().enumerate() {
                *o = x[ox + j * ix];
            }
        });
        wrap(out)
    }

    fn index_select(&self, table: &Storage, cols: usize, idx: &[usize]) -> Storage {
        let t = data(table);
        let rows = t.len() / cols.max(1);
        let mut out = vec![0.0; idx.len() * cols];
        for_rows(&mut out, cols, |r, row| {
            let i = idx[r];
            assert!(i < rows, "index {i} out of range for {rows} rows");
            row.copy_from_slice(&t[i * cols..(i + 1) * cols]);
        });
        wrap(out)
    }

    fn index_add(&self, src: &Storage, rows: usize, cols: usize, idx: &[usize]) -> Storage {
        let s = data(src);
        let mut out = vec![0.0; rows * cols];
        for (r, &i) in idx.iter().enumerate() {
            let dst = &mut out[i * cols..(i + 1) * cols];
            for (d, v) in dst.iter_mut().zip(&s[r * cols..(r + 1) * cols]) {
                *d += v;
            }
        }
        wrap(out)
    }

    fn im2col(&self, x: &Storage, g: Conv2dGeom) -> Storage {
        let x = data(x);
        let (oh, ow) = g.out_hw();
        let l = oh * ow;
        let ckk = g.c * g.kh * g.kw;
        let mut out = vec![0.0; g.n * ckk * l];
        // one row per (n, c, ki, kj)
        for_rows(&mut out, l, |r, row| {
            let kj = r % g.kw;
            let ki = (r / g.kw) % g.kh;
            let c = (r / (g.kw * g.kh)) % g.c;
            let n = r / ckk;
            let plane = &x[(n * g.c + c) * g.h * g.w..(n * g.c + c + 1) * g.h * g.w];
            for y in 0..oh {
                let iy = (y * g.stride + ki) as isize - g.padding as isize;
                if iy < 0 || iy >= g.h as isize {
                    continue; // stays zero (padding)
                }
                let src = &plane[iy as usize * g.w..(iy as usize + 1) * g.w];
                let dst = &mut row[y * ow..(y + 1) * ow];
                for (xo, d) in dst.iter_mut().enumerate() {
                    let ix = (xo * g.stride + kj) as isize - g.padding as isize;
                    if ix >= 0 && ix < g.w as isize {
                        *d = src[ix as usize];
                    }
                }
            }
        });
        wrap(out)
    }

    fn col2im(&self, cols: &Storage, g: Conv2dGeom) -> Storage {
        let cols = data(cols);
        let (oh, ow) = g.out_hw();
        let l = oh * ow;
        let ckk = g.c * g.kh * g.kw;
        let hw = g.h * g.w;
        let mut out = vec![0.0; g.n * g.c * hw];
        // one plane per (n, c): accumulate all its (ki, kj) rows; planes are independent.
        for_rows(&mut out, hw, |p, plane| {
            let c = p % g.c;
            let n = p / g.c;
            for ki in 0..g.kh {
                for kj in 0..g.kw {
                    let r = n * ckk + (c * g.kh + ki) * g.kw + kj;
                    let row = &cols[r * l..(r + 1) * l];
                    for y in 0..oh {
                        let iy = (y * g.stride + ki) as isize - g.padding as isize;
                        if iy < 0 || iy >= g.h as isize {
                            continue;
                        }
                        for xo in 0..ow {
                            let ix = (xo * g.stride + kj) as isize - g.padding as isize;
                            if ix >= 0 && ix < g.w as isize {
                                plane[iy as usize * g.w + ix as usize] += row[y * ow + xo];
                            }
                        }
                    }
                }
            }
        });
        wrap(out)
    }
}

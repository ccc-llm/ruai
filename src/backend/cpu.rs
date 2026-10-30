//! CPU backend: host `Vec` storage, parallelized with rayon. Matmul uses the
//! pure-Rust `matrixmultiply` GEMM (or Apple Accelerate with `--features accelerate`).

use super::{
    reference::Unfused, AdamParams, Backend, BinaryOp, Conv2dGeom, CpuStorage, DType, MatmulSpec, ReduceOp, Storage, UnaryOp,
};
use crate::device::Device;
use crate::shape::{broadcast_strides, numel, split_axis, strides};
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::Arc;

/// Below this many elements, work is done on the calling thread.
const PAR_THRESHOLD: usize = 1 << 15;

pub struct CpuBackend;

fn f32s(s: &Storage) -> &[f32] {
    match s {
        Storage::Cpu(CpuStorage::F32(v)) => v.as_slice(),
        #[allow(unreachable_patterns)]
        _ => panic!("CPU backend expected an f32 CPU storage, got {} on {}", s.dtype(), s.device()),
    }
}

fn i32s(s: &Storage) -> &[i32] {
    match s {
        Storage::Cpu(CpuStorage::I32(v)) => v.as_slice(),
        #[allow(unreachable_patterns)]
        _ => panic!("CPU backend expected an i32 CPU storage (indices), got {} on {}", s.dtype(), s.device()),
    }
}

/// View any CPU storage as f32 (converting integers).
fn as_f32(s: &Storage) -> Cow<'_, [f32]> {
    match s {
        Storage::Cpu(CpuStorage::F32(v)) => Cow::Borrowed(v.as_slice()),
        Storage::Cpu(CpuStorage::I32(v)) => Cow::Owned(v.iter().map(|&x| x as f32).collect()),
        #[allow(unreachable_patterns)]
        _ => panic!("CPU backend received a non-CPU storage ({})", s.device()),
    }
}

fn wrap(v: Vec<f32>) -> Storage {
    Storage::Cpu(CpuStorage::F32(Arc::new(v)))
}

fn wrap_i32(v: Vec<i32>) -> Storage {
    Storage::Cpu(CpuStorage::I32(Arc::new(v)))
}

fn map_into<F: Fn(f32) -> f32 + Sync + Send>(x: &[f32], f: F) -> Vec<f32> {
    if x.len() >= PAR_THRESHOLD {
        x.par_iter().with_min_len(PAR_THRESHOLD / 4).map(|&v| f(v)).collect()
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
        let min_rows = (PAR_THRESHOLD / 4 / inner).max(1);
        out.par_chunks_mut(inner).with_min_len(min_rows).enumerate().for_each(|(r, row)| f(r, row));
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
            a.par_iter().zip(b.par_iter()).with_min_len(PAR_THRESHOLD / 4).map(|(&x, &y)| f(x, y)).collect()
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
        match (ia, ib) {
            (1, 1) => {
                for ((o, &x), &y) in row.iter_mut().zip(&a[oa..oa + inner]).zip(&b[ob..ob + inner]) {
                    *o = f(x, y);
                }
            }
            (1, 0) => {
                let y = b[ob];
                for (o, &x) in row.iter_mut().zip(&a[oa..oa + inner]) {
                    *o = f(x, y);
                }
            }
            (0, 1) => {
                let x = a[oa];
                for (o, &y) in row.iter_mut().zip(&b[ob..ob + inner]) {
                    *o = f(x, y);
                }
            }
            _ => {
                for (j, o) in row.iter_mut().enumerate() {
                    *o = f(a[oa + j * ia], b[ob + j * ib]);
                }
            }
        }
    });
    out
}

/// tanh via a single exp (noticeably faster than libm `tanhf`).
#[inline]
fn fast_tanh(u: f32) -> f32 {
    let u = u.clamp(-15.0, 15.0);
    1.0 - 2.0 / ((2.0 * u).exp() + 1.0)
}

#[inline]
fn gelu(x: f32) -> f32 {
    const S: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + fast_tanh(S * (x + 0.044715 * x * x * x)))
}

#[inline]
fn gelu_grad(x: f32) -> f32 {
    const S: f32 = 0.797_884_6;
    let u = S * (x + 0.044715 * x * x * x);
    let t = fast_tanh(u);
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * S * (1.0 + 3.0 * 0.044715 * x * x)
}

fn narrow_t<T: Copy>(x: &[T], shape: &[usize], axis: usize, start: usize, len: usize) -> Vec<T> {
    let (outer, l, inner) = split_axis(shape, axis);
    let mut out = Vec::with_capacity(outer * len * inner);
    for o in 0..outer {
        out.extend_from_slice(&x[(o * l + start) * inner..(o * l + start + len) * inner]);
    }
    out
}

fn cat_t<T: Copy>(xs: &[&[T]], shapes: &[&[usize]], axis: usize) -> Vec<T> {
    let (outer, _, inner) = split_axis(shapes[0], axis);
    let total: usize = xs.iter().map(|x| x.len()).sum();
    let mut out = Vec::with_capacity(total);
    for o in 0..outer {
        for (x, s) in xs.iter().zip(shapes) {
            let blk = s[axis] * inner;
            out.extend_from_slice(&x[o * blk..(o + 1) * blk]);
        }
    }
    out
}

/// Single-threaded GEMM on one block: `C (m x n, row stride rsc) = A @ B`
/// with arbitrary element strides for A and B.
#[allow(clippy::too_many_arguments)]
#[cfg(not(feature = "accelerate"))]
fn gemm(m: usize, k: usize, n: usize, a: &[f32], rsa: usize, csa: usize, b: &[f32], rsb: usize, csb: usize, c: &mut [f32]) {
    // Safety: callers guarantee the slices cover the strided extents.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            rsa as isize,
            csa as isize,
            b.as_ptr(),
            rsb as isize,
            csb as isize,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

#[cfg(feature = "accelerate")]
mod accelerate {
    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        #[allow(clippy::too_many_arguments)]
        pub fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }
    pub const ROW_MAJOR: i32 = 101;
    pub const NO_TRANS: i32 = 111;
    pub const TRANS: i32 = 112;
}

impl Backend for CpuBackend {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn from_f32(&self, data: Vec<f32>) -> Storage {
        wrap(data)
    }

    fn from_i32(&self, data: Vec<i32>) -> Storage {
        wrap_i32(data)
    }

    fn to_f32(&self, x: &Storage) -> Vec<f32> {
        as_f32(x).into_owned()
    }

    fn to_i32(&self, x: &Storage) -> Vec<i32> {
        match x {
            Storage::Cpu(CpuStorage::I32(v)) => v.to_vec(),
            _ => f32s(x).iter().map(|&v| v as i32).collect(),
        }
    }

    fn full(&self, len: usize, value: f32, dtype: DType) -> Storage {
        match dtype {
            DType::F32 => wrap(vec![value; len]),
            DType::I32 => wrap_i32(vec![value as i32; len]),
        }
    }

    fn cast(&self, x: &Storage, dtype: DType) -> Storage {
        if x.dtype() == dtype {
            return x.clone();
        }
        match dtype {
            DType::F32 => wrap(self.to_f32(x)),
            DType::I32 => wrap_i32(self.to_i32(x)),
        }
    }

    fn unary(&self, op: UnaryOp, x: &Storage) -> Storage {
        let x = f32s(x);
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
        let (a, b) = (as_f32(a), as_f32(b));
        let (a, b) = (a.as_ref(), b.as_ref());
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
        let x = f32s(x);
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
            if ix == 1 {
                row.copy_from_slice(&x[ox..ox + inner]);
            } else {
                for (j, o) in row.iter_mut().enumerate() {
                    *o = x[ox + j * ix];
                }
            }
        });
        wrap(out)
    }

    fn where_cond(&self, cond: &Storage, a: &Storage, b: &Storage) -> Storage {
        let (c, a, b) = (as_f32(cond), f32s(a), f32s(b));
        wrap(c.iter().zip(a).zip(b).map(|((&c, &x), &y)| if c != 0.0 { x } else { y }).collect())
    }

    fn reduce(&self, op: ReduceOp, x: &Storage, shape: &[usize], axis: usize) -> Storage {
        let x = f32s(x);
        let (outer, len, inner) = split_axis(shape, axis);
        let mut out = vec![0.0; outer * inner];
        if out.is_empty() {
            return if op == ReduceOp::ArgMax { wrap_i32(vec![]) } else { wrap(out) };
        }
        let reduce_block = |o: usize, dst: &mut [f32]| {
            let base = o * len * inner;
            match op {
                ReduceOp::Sum => {
                    if inner == 1 {
                        dst[0] = x[base..base + len].iter().sum();
                        return;
                    }
                    dst.fill(0.0);
                    for l in 0..len {
                        let src = &x[base + l * inner..base + (l + 1) * inner];
                        for (d, s) in dst.iter_mut().zip(src) {
                            *d += s;
                        }
                    }
                }
                ReduceOp::Max | ReduceOp::Min => {
                    let is_max = op == ReduceOp::Max;
                    dst.fill(if is_max { f32::NEG_INFINITY } else { f32::INFINITY });
                    for l in 0..len {
                        let src = &x[base + l * inner..base + (l + 1) * inner];
                        for (d, &s) in dst.iter_mut().zip(src) {
                            *d = if is_max { d.max(s) } else { d.min(s) };
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
                .with_min_len((PAR_THRESHOLD / 4 / (len * inner).max(1)).max(1))
                .enumerate()
                .for_each(|(o, d)| reduce_block(o, d));
        } else {
            out.chunks_mut(inner).enumerate().for_each(|(o, d)| reduce_block(o, d));
        }
        if op == ReduceOp::ArgMax {
            wrap_i32(out.into_iter().map(|v| v as i32).collect())
        } else {
            wrap(out)
        }
    }

    fn matmul(&self, a: &Storage, b: &Storage, spec: MatmulSpec) -> Storage {
        let MatmulSpec { batch, m, k, n, trans_a, trans_b, a_batch_stride, b_batch_stride } = spec;
        let (a, b) = (f32s(a), f32s(b));
        let mut out = vec![0.0f32; batch * m * n];
        if out.is_empty() || k == 0 {
            return wrap(out);
        }
        let a_bs = if a_batch_stride == 0 { 0 } else { m * k };
        let b_bs = if b_batch_stride == 0 { 0 } else { k * n };

        #[cfg(feature = "accelerate")]
        {
            use accelerate::*;
            let call = |bi: usize, c: &mut [f32]| unsafe {
                cblas_sgemm(
                    ROW_MAJOR,
                    if trans_a { TRANS } else { NO_TRANS },
                    if trans_b { TRANS } else { NO_TRANS },
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a[bi * a_bs..].as_ptr(),
                    if trans_a { m } else { k } as i32,
                    b[bi * b_bs..].as_ptr(),
                    if trans_b { k } else { n } as i32,
                    0.0,
                    c.as_mut_ptr(),
                    n as i32,
                )
            };
            // Accelerate threads large GEMMs itself; fan out only many small ones.
            if batch > 1 && m * n * k < 128 * 128 * 128 {
                out.par_chunks_mut(m * n).enumerate().for_each(|(bi, c)| call(bi, c));
            } else {
                out.chunks_mut(m * n).enumerate().for_each(|(bi, c)| call(bi, c));
            }
            #[allow(clippy::needless_return)]
            return wrap(out);
        }

        #[cfg(not(feature = "accelerate"))]
        {
            // element (i, p) of A and (p, j) of B
            let (rsa, csa) = if trans_a { (1, m) } else { (k, 1) };
            let (rsb, csb) = if trans_b { (1, k) } else { (n, 1) };
            let threads = rayon::current_num_threads();
            let flops = batch * m * n * k;
            // split each batch into row blocks so that all threads have work
            let chunks = if flops < 1 << 18 || batch >= threads { 1 } else { threads.div_ceil(batch) };
            let rows_per = m.div_ceil(chunks).max(1);
            let block = |bi: usize, ci: usize, c: &mut [f32]| {
                let r0 = ci * rows_per;
                let rows = c.len() / n;
                let a_blk = &a[bi * a_bs + r0 * rsa..];
                let b_blk = &b[bi * b_bs..];
                gemm(rows, k, n, a_blk, rsa, csa, b_blk, rsb, csb, c);
            };
            if flops < 1 << 18 {
                out.chunks_mut(m * n).enumerate().for_each(|(bi, cb)| block(bi, 0, cb));
            } else {
                out.par_chunks_mut(m * n).enumerate().for_each(|(bi, cb)| {
                    cb.par_chunks_mut(rows_per * n).enumerate().for_each(|(ci, c)| block(bi, ci, c));
                });
            }
            wrap(out)
        }
    }

    fn permute(&self, x: &Storage, shape: &[usize], perm: &[usize]) -> Storage {
        let x = f32s(x);
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
            if ix == 1 {
                row.copy_from_slice(&x[ox..ox + inner]);
            } else {
                for (j, o) in row.iter_mut().enumerate() {
                    *o = x[ox + j * ix];
                }
            }
        });
        wrap(out)
    }

    fn narrow(&self, x: &Storage, shape: &[usize], axis: usize, start: usize, len: usize) -> Storage {
        match x {
            Storage::Cpu(CpuStorage::F32(v)) => wrap(narrow_t(v, shape, axis, start, len)),
            Storage::Cpu(CpuStorage::I32(v)) => wrap_i32(narrow_t(v, shape, axis, start, len)),
            #[allow(unreachable_patterns)]
            _ => panic!("CPU backend received a non-CPU storage"),
        }
    }

    fn cat(&self, xs: &[&Storage], shapes: &[&[usize]], axis: usize) -> Storage {
        match xs[0].dtype() {
            DType::F32 => wrap(cat_t(&xs.iter().map(|s| f32s(s)).collect::<Vec<_>>(), shapes, axis)),
            DType::I32 => wrap_i32(cat_t(&xs.iter().map(|s| i32s(s)).collect::<Vec<_>>(), shapes, axis)),
        }
    }

    fn index_select(&self, table: &Storage, cols: usize, idx: &Storage) -> Storage {
        let (t, idx) = (f32s(table), i32s(idx));
        let rows = t.len() / cols.max(1);
        let mut out = vec![0.0; idx.len() * cols];
        for_rows(&mut out, cols, |r, row| {
            let i = idx[r] as usize;
            assert!(i < rows, "index {} out of range for {rows} rows", idx[r]);
            row.copy_from_slice(&t[i * cols..(i + 1) * cols]);
        });
        wrap(out)
    }

    fn index_add(&self, src: &Storage, rows: usize, cols: usize, idx: &Storage) -> Storage {
        let (s, idx) = (f32s(src), i32s(idx));
        let mut out = vec![0.0; rows * cols];
        for (r, &i) in idx.iter().enumerate() {
            let i = i as usize;
            let dst = &mut out[i * cols..(i + 1) * cols];
            for (d, v) in dst.iter_mut().zip(&s[r * cols..(r + 1) * cols]) {
                *d += v;
            }
        }
        wrap(out)
    }

    fn gather(&self, x: &Storage, shape: &[usize], axis: usize, idx: &Storage, idx_len: usize) -> Storage {
        let (x, idx) = (f32s(x), i32s(idx));
        let (_, len, inner) = split_axis(shape, axis);
        let mut out = vec![0.0; idx.len()];
        for_rows(&mut out, idx_len * inner, |o, blk| {
            for j in 0..idx_len {
                for i in 0..inner {
                    let p = o * idx_len * inner + j * inner + i;
                    let l = idx[p] as usize;
                    assert!(l < len, "gather index {l} out of range for size {len}");
                    blk[j * inner + i] = x[(o * len + l) * inner + i];
                }
            }
        });
        wrap(out)
    }

    fn scatter_add(&self, src: &Storage, idx: &Storage, idx_len: usize, shape: &[usize], axis: usize) -> Storage {
        let (src, idx) = (f32s(src), i32s(idx));
        let (_, len, inner) = split_axis(shape, axis);
        let mut out = vec![0.0; numel(shape)];
        for_rows(&mut out, len * inner, |o, blk| {
            for j in 0..idx_len {
                for i in 0..inner {
                    let p = o * idx_len * inner + j * inner + i;
                    blk[idx[p] as usize * inner + i] += src[p];
                }
            }
        });
        wrap(out)
    }

    fn im2col(&self, x: &Storage, g: Conv2dGeom) -> Storage {
        let x = f32s(x);
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
        let cols = f32s(cols);
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

    // ---- fused fast paths (last axis); other axes use the composed default ----

    fn softmax(&self, x: &Storage, shape: &[usize], axis: usize) -> Storage {
        if axis + 1 != shape.len() {
            return Unfused(self.static_ref()).softmax(x, shape, axis);
        }
        let x = f32s(x);
        let d = shape[axis];
        let mut out = vec![0.0; x.len()];
        for_rows(&mut out, d, |r, row| {
            let src = &x[r * d..(r + 1) * d];
            let m = src.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0.0;
            for (o, &v) in row.iter_mut().zip(src) {
                *o = (v - m).exp();
                s += *o;
            }
            let inv = 1.0 / s;
            row.iter_mut().for_each(|o| *o *= inv);
        });
        wrap(out)
    }

    fn softmax_backward(&self, y: &Storage, g: &Storage, shape: &[usize], axis: usize) -> Storage {
        if axis + 1 != shape.len() {
            return Unfused(self.static_ref()).softmax_backward(y, g, shape, axis);
        }
        let (y, g) = (f32s(y), f32s(g));
        let d = shape[axis];
        let mut out = vec![0.0; y.len()];
        for_rows(&mut out, d, |r, row| {
            let (yr, gr) = (&y[r * d..(r + 1) * d], &g[r * d..(r + 1) * d]);
            let dot: f32 = yr.iter().zip(gr).map(|(a, b)| a * b).sum();
            for ((o, &yv), &gv) in row.iter_mut().zip(yr).zip(gr) {
                *o = yv * (gv - dot);
            }
        });
        wrap(out)
    }

    fn log_softmax(&self, x: &Storage, shape: &[usize], axis: usize) -> Storage {
        if axis + 1 != shape.len() {
            return Unfused(self.static_ref()).log_softmax(x, shape, axis);
        }
        let x = f32s(x);
        let d = shape[axis];
        let mut out = vec![0.0; x.len()];
        for_rows(&mut out, d, |r, row| {
            let src = &x[r * d..(r + 1) * d];
            let m = src.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = m + src.iter().map(|&v| (v - m).exp()).sum::<f32>().ln();
            for (o, &v) in row.iter_mut().zip(src) {
                *o = v - lse;
            }
        });
        wrap(out)
    }

    fn log_softmax_backward(&self, y: &Storage, g: &Storage, shape: &[usize], axis: usize) -> Storage {
        if axis + 1 != shape.len() {
            return Unfused(self.static_ref()).log_softmax_backward(y, g, shape, axis);
        }
        let (y, g) = (f32s(y), f32s(g));
        let d = shape[axis];
        let mut out = vec![0.0; y.len()];
        for_rows(&mut out, d, |r, row| {
            let (yr, gr) = (&y[r * d..(r + 1) * d], &g[r * d..(r + 1) * d]);
            let s: f32 = gr.iter().sum();
            for ((o, &yv), &gv) in row.iter_mut().zip(yr).zip(gr) {
                *o = gv - yv.exp() * s;
            }
        });
        wrap(out)
    }

    fn layer_norm(&self, x: &Storage, rows: usize, dim: usize, eps: f32) -> (Storage, Storage) {
        let x = f32s(x);
        let stats = |r: usize| {
            let src = &x[r * dim..(r + 1) * dim];
            let mean = src.iter().sum::<f32>() / dim as f32;
            let var = src.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
            (mean, 1.0 / (var + eps).sqrt())
        };
        let st: Vec<(f32, f32)> = if x.len() >= PAR_THRESHOLD {
            (0..rows).into_par_iter().map(stats).collect()
        } else {
            (0..rows).map(stats).collect()
        };
        let mut out = vec![0.0; x.len()];
        for_rows(&mut out, dim, |r, row| {
            let (mean, rstd) = st[r];
            for (o, &v) in row.iter_mut().zip(&x[r * dim..(r + 1) * dim]) {
                *o = (v - mean) * rstd;
            }
        });
        (wrap(out), wrap(st.into_iter().map(|s| s.1).collect()))
    }

    fn layer_norm_backward(&self, g: &Storage, xhat: &Storage, rstd: &Storage, _rows: usize, dim: usize) -> Storage {
        let (g, xh, rs) = (f32s(g), f32s(xhat), f32s(rstd));
        let mut out = vec![0.0; g.len()];
        let inv = 1.0 / dim as f32;
        for_rows(&mut out, dim, |r, row| {
            let (gr, xr) = (&g[r * dim..(r + 1) * dim], &xh[r * dim..(r + 1) * dim]);
            let mg = gr.iter().sum::<f32>() * inv;
            let mgx = gr.iter().zip(xr).map(|(a, b)| a * b).sum::<f32>() * inv;
            for ((o, &gv), &xv) in row.iter_mut().zip(gr).zip(xr) {
                *o = rs[r] * (gv - mg - xv * mgx);
            }
        });
        wrap(out)
    }

    fn adam_step(&self, p: &Storage, g: &Storage, m: &Storage, v: &Storage, h: AdamParams) -> (Storage, Storage, Storage) {
        let (p, g, m, v) = (f32s(p), f32s(g), f32s(m), f32s(v));
        let n = p.len();
        let (mut np, mut nm, mut nv) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let (ib1, ib2) = (1.0 / h.bias_correction1, 1.0 / h.bias_correction2);
        let chunk = 1 << 14;
        np.par_chunks_mut(chunk)
            .zip(nm.par_chunks_mut(chunk))
            .zip(nv.par_chunks_mut(chunk))
            .enumerate()
            .for_each(|(c, ((op, om), ov))| {
                let base = c * chunk;
                for i in 0..op.len() {
                    let j = base + i;
                    let mut gj = g[j];
                    if h.weight_decay != 0.0 && !h.decoupled {
                        gj += h.weight_decay * p[j];
                    }
                    let mj = h.beta1 * m[j] + (1.0 - h.beta1) * gj;
                    let vj = h.beta2 * v[j] + (1.0 - h.beta2) * gj * gj;
                    let mut pj = p[j] - h.lr * (mj * ib1) / ((vj * ib2).sqrt() + h.eps);
                    if h.weight_decay != 0.0 && h.decoupled {
                        pj -= h.lr * h.weight_decay * p[j];
                    }
                    op[i] = pj;
                    om[i] = mj;
                    ov[i] = vj;
                }
            });
        (wrap(np), wrap(nm), wrap(nv))
    }
}

impl CpuBackend {
    fn static_ref(&self) -> &'static dyn Backend {
        super::backend_for(Device::Cpu)
    }
}

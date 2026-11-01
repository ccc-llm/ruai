// WGSL compute kernels of the WebGPU backend (webgpu.rs).
//
// Each `//! kernel: NAME` section is compiled as its own module after this
// prelude, with entry point `main`. Binding 0 holds the parameters as u32
// words (floats bitcast); bindings 1.. are the buffers, in the order webgpu.rs
// passes them. Storage is 32-bit: f32, i32, half floats as f32 values rounded
// to f16 / bf16 (`round_f16`, `round_bf16`), i8 as packed bytes.
// Elementwise kernels run 256 threads per workgroup on a 2-D grid of
// workgroups (more than 65535 of them): `gid` is the global index.

@group(0) @binding(0) var<storage, read> p: array<u32>;

fn pf(i: u32) -> f32 {
    return bitcast<f32>(p[i]);
}

fn gid(wg: vec3<u32>, nwg: vec3<u32>, li: u32) -> u32 {
    return (wg.y * nwg.x + wg.x) * 256u + li;
}

// infinity from a runtime value (a constant infinity is a shader-creation error)
var<private> inf_bits: u32 = 0x7f800000u;
fn inf() -> f32 {
    return bitcast<f32>(inf_bits);
}

// tanh of large arguments may be NaN on some drivers; tanh(±15) is ±1 in f32
fn tanh_f(x: f32) -> f32 {
    return tanh(clamp(x, -15.0, 15.0));
}

// `as` conversions of Rust: NaN → 0, saturating
fn to_i32(x: f32) -> i32 {
    if (x != x) {
        return 0;
    }
    if (x >= 2147483647.0) {
        return 2147483647;
    }
    if (x <= -2147483648.0) {
        return -2147483647 - 1;
    }
    return i32(x);
}

fn to_i8(x: f32) -> i32 {
    if (x != x) {
        return 0;
    }
    return i32(clamp(x, -128.0, 127.0));
}

//! kernel: fill
// p: n, bits
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = gid(wg, nwg, li);
    if (i < p[0]) {
        out[i] = p[1];
    }
}

//! kernel: round_f16
// p: n — f32 values rounded to the nearest f16 (ties to even)
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
fn f16_round(v: f32) -> f32 {
    let u = bitcast<u32>(v);
    let sign = u & 0x80000000u;
    let a = u & 0x7fffffffu;
    if (a > 0x7f800000u) {
        return v;
    }
    if (a >= 0x477ff000u) {
        return bitcast<f32>(sign | 0x7f800000u);
    }
    if (a < 0x38800000u) {
        // subnormal f16: multiples of 2^-24
        let q = round(bitcast<f32>(a) * 16777216.0) * 5.9604644775390625e-8;
        return bitcast<f32>(sign | bitcast<u32>(q));
    }
    let lsb = (a >> 13u) & 1u;
    return bitcast<f32>(sign | ((a + 0xfffu + lsb) & 0xffffe000u));
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = gid(wg, nwg, li);
    if (i < p[0]) {
        out[i] = f16_round(x[i]);
    }
}

//! kernel: round_bf16
// p: n — f32 values rounded to the nearest bf16 (ties to even)
@group(0) @binding(1) var<storage, read> x: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = gid(wg, nwg, li);
    if (i >= p[0]) {
        return;
    }
    let u = x[i];
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        out[i] = 0x7fc00000u | (u & 0x80000000u);
    } else {
        out[i] = (u + 0x7fffu + ((u >> 16u) & 1u)) & 0xffff0000u;
    }
}

//! kernel: cast_f32_i32
// p: n
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<i32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = gid(wg, nwg, li);
    if (i < p[0]) {
        out[i] = to_i32(x[i]);
    }
}

//! kernel: cast_i32_f32
// p: n
@group(0) @binding(1) var<storage, read> x: array<i32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = gid(wg, nwg, li);
    if (i < p[0]) {
        out[i] = f32(x[i]);
    }
}

//! kernel: cast_i8_x
// p: n, to_float — packed bytes → f32 (to_float = 1) or i32
@group(0) @binding(1) var<storage, read> q: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = gid(wg, nwg, li);
    if (i >= p[0]) {
        return;
    }
    let b = (q[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu;
    let v = bitcast<i32>(b << 24u) >> 24u;
    if (p[1] != 0u) {
        out[i] = bitcast<u32>(f32(v));
    } else {
        out[i] = bitcast<u32>(v);
    }
}

//! kernel: cast_x_i8
// p: n, from_float — f32 (from_float = 1) or i32 → packed bytes, one word per thread
@group(0) @binding(1) var<storage, read> x: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let w = gid(wg, nwg, li);
    let n = p[0];
    if (w * 4u >= n) {
        return;
    }
    var word = 0u;
    for (var k = 0u; k < 4u; k++) {
        let i = w * 4u + k;
        if (i < n) {
            var v: i32;
            if (p[1] != 0u) {
                v = to_i8(bitcast<f32>(x[i]));
            } else {
                v = clamp(bitcast<i32>(x[i]), -128, 127);
            }
            word |= (bitcast<u32>(v) & 0xffu) << (k * 8u);
        }
    }
    out[w] = word;
}

//! kernel: fill_i8
// p: n words, byte
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = gid(wg, nwg, li);
    if (i < p[0]) {
        let b = p[1] & 0xffu;
        out[i] = b | (b << 8u) | (b << 16u) | (b << 24u);
    }
}

//! kernel: unary
// p: n, op (UnaryOp order: neg exp log sqrt recip abs tanh sigmoid relu gelu gelu_grad sin cos)
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = gid(wg, nwg, li);
    if (i >= p[0]) {
        return;
    }
    let v = x[i];
    let S = 0.7978845608;
    var r: f32;
    switch (p[1]) {
        case 0u: { r = -v; }
        case 1u: { r = exp(v); }
        case 2u: { r = log(v); }
        case 3u: { r = sqrt(v); }
        case 4u: { r = 1.0 / v; }
        case 5u: { r = abs(v); }
        case 6u: { r = tanh_f(v); }
        case 7u: { r = 1.0 / (1.0 + exp(-v)); }
        case 8u: { r = max(v, 0.0); }
        case 9u: { r = 0.5 * v * (1.0 + tanh_f(S * (v + 0.044715 * v * v * v))); }
        case 10u: {
            let t = tanh_f(S * (v + 0.044715 * v * v * v));
            r = 0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * S * (1.0 + 3.0 * 0.044715 * v * v);
        }
        case 11u: { r = sin(v); }
        default: { r = cos(v); }
    }
    out[i] = r;
}

//! kernel: binary
// p: n, rank, op, pad, shape[8], sa[8], sb[8]
// (BinaryOp order: add sub mul div pow maximum minimum eq gt lt)
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
fn powf(x: f32, y: f32) -> f32 {
    // pow of WGSL is undefined for x <= 0: as C's powf
    if (y == 0.0) {
        return 1.0;
    }
    if (x == 0.0) {
        if (y > 0.0) {
            if (fract(y * 0.5) == 0.5) {
                return x;
            }
            return 0.0;
        }
        if (fract(y * 0.5) == 0.5) {
            return 1.0 / x;
        }
        return inf();
    }
    if (x < 0.0) {
        if (fract(y) != 0.0) {
            return (x - x) / (x - x);
        }
        let m = pow(-x, y);
        if (fract(y * 0.5) != 0.0) {
            return -m;
        }
        return m;
    }
    return pow(x, y);
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = gid(wg, nwg, li);
    if (i >= p[0]) {
        return;
    }
    var rem = i;
    var oa = 0u;
    var ob = 0u;
    for (var d = i32(p[1]) - 1; d >= 0; d--) {
        let s = p[4u + u32(d)];
        let c = rem % s;
        rem = rem / s;
        oa += c * p[12u + u32(d)];
        ob += c * p[20u + u32(d)];
    }
    let x = a[oa];
    let y = b[ob];
    var r: f32;
    switch (p[2]) {
        case 0u: { r = x + y; }
        case 1u: { r = x - y; }
        case 2u: { r = x * y; }
        case 3u: { r = x / y; }
        case 4u: { r = powf(x, y); }
        case 5u: { r = max(x, y); }
        case 6u: { r = min(x, y); }
        case 7u: { r = select(0.0, 1.0, x == y); }
        case 8u: { r = select(0.0, 1.0, x > y); }
        default: { r = select(0.0, 1.0, x < y); }
    }
    out[i] = r;
}

//! kernel: strided_copy
// p: n, rank, base, pad, shape[8], strides[8] — out[i] = x[base + strided offset of i]
@group(0) @binding(1) var<storage, read> x: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = gid(wg, nwg, li);
    if (i >= p[0]) {
        return;
    }
    var rem = i;
    var o = p[2];
    for (var d = i32(p[1]) - 1; d >= 0; d--) {
        let s = p[4u + u32(d)];
        o += (rem % s) * p[12u + u32(d)];
        rem = rem / s;
    }
    out[i] = x[o];
}

//! kernel: cat_copy
// p: n, blk_in, blk_out, offset
@group(0) @binding(1) var<storage, read> x: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = gid(wg, nwg, li);
    if (i < p[0]) {
        out[(i / p[1]) * p[2] + p[3] + i % p[1]] = x[i];
    }
}

//! kernel: where
// p: n — out = cond != 0 ? a : b (any 32-bit dtype)
@group(0) @binding(1) var<storage, read> c: array<f32>;
@group(0) @binding(2) var<storage, read> a: array<u32>;
@group(0) @binding(3) var<storage, read> b: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = gid(wg, nwg, li);
    if (i < p[0]) {
        out[i] = select(b[i], a[i], c[i] != 0.0);
    }
}

//! kernel: reduce_strided
// p: outer, len, inner, op (0 sum, 1 max, 2 min, 3 argmax → i32)
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = gid(wg, nwg, li);
    let inner = p[2];
    if (t >= p[0] * inner) {
        return;
    }
    let base = (t / inner) * p[1] * inner + t % inner;
    let op = p[3];
    if (op == 3u) {
        var best = -inf();
        var bi = 0u;
        for (var l = 0u; l < p[1]; l++) {
            let v = x[base + l * inner];
            if (v > best) {
                best = v;
                bi = l;
            }
        }
        out[t] = bi;
        return;
    }
    var acc = select(select(-inf(), inf(), op == 2u), 0.0, op == 0u);
    for (var l = 0u; l < p[1]; l++) {
        let v = x[base + l * inner];
        if (op == 0u) {
            acc += v;
        } else if (op == 1u) {
            acc = max(acc, v);
        } else {
            acc = min(acc, v);
        }
    }
    out[t] = bitcast<u32>(acc);
}

//! kernel: reduce_rows
// p: rows, len, inner (1), op (0 sum, 1 max, 2 min) — one workgroup per row
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<workgroup> sh: array<f32, 256>;
fn comb(op: u32, a: f32, b: f32) -> f32 {
    if (op == 0u) {
        return a + b;
    }
    if (op == 1u) {
        return max(a, b);
    }
    return min(a, b);
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let row = wg.y * nwg.x + wg.x;
    if (row >= p[0]) {
        return;
    }
    let op = p[3];
    let d = p[1];
    var acc = select(select(-inf(), inf(), op == 2u), 0.0, op == 0u);
    for (var l = tid; l < d; l += 256u) {
        acc = comb(op, acc, x[row * d + l]);
    }
    sh[tid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sh[tid] = comb(op, sh[tid], sh[tid + s]);
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        out[row] = sh[0];
    }
}

//! kernel: softmax_rows
// p: rows, d, log — one workgroup per row
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
var<workgroup> sh: array<f32, 256>;
fn wg_max(v: f32, tid: u32) -> f32 {
    sh[tid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sh[tid] = max(sh[tid], sh[tid + s]);
        }
        workgroupBarrier();
    }
    let r = sh[0];
    workgroupBarrier();
    return r;
}
fn wg_sum(v: f32, tid: u32) -> f32 {
    sh[tid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sh[tid] = sh[tid] + sh[tid + s];
        }
        workgroupBarrier();
    }
    let r = sh[0];
    workgroupBarrier();
    return r;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let row = wg.y * nwg.x + wg.x;
    if (row >= p[0]) {
        return;
    }
    let d = p[1];
    let off = row * d;
    var m = -inf();
    for (var l = tid; l < d; l += 256u) {
        m = max(m, x[off + l]);
    }
    m = wg_max(m, tid);
    var s = 0.0;
    for (var l = tid; l < d; l += 256u) {
        s += exp(x[off + l] - m);
    }
    s = wg_sum(s, tid);
    if (p[2] != 0u) {
        let lse = m + log(s);
        for (var l = tid; l < d; l += 256u) {
            y[off + l] = x[off + l] - lse;
        }
    } else {
        let inv = 1.0 / s;
        for (var l = tid; l < d; l += 256u) {
            y[off + l] = exp(x[off + l] - m) * inv;
        }
    }
}

//! kernel: softmax_backward_rows
// p: rows, d, log
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
var<workgroup> sh: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let row = wg.y * nwg.x + wg.x;
    if (row >= p[0]) {
        return;
    }
    let d = p[1];
    let off = row * d;
    let lg = p[2] != 0u;
    var s = 0.0;
    for (var l = tid; l < d; l += 256u) {
        if (lg) {
            s += g[off + l];
        } else {
            s += g[off + l] * y[off + l];
        }
    }
    sh[tid] = s;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k >>= 1u) {
        if (tid < k) {
            sh[tid] = sh[tid] + sh[tid + k];
        }
        workgroupBarrier();
    }
    let tot = sh[0];
    for (var l = tid; l < d; l += 256u) {
        let yv = y[off + l];
        let gv = g[off + l];
        if (lg) {
            out[off + l] = gv - exp(yv) * tot;
        } else {
            out[off + l] = yv * (gv - tot);
        }
    }
}

//! kernel: layer_norm_rows
// p: rows, d, eps
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> xhat: array<f32>;
@group(0) @binding(3) var<storage, read_write> rstd: array<f32>;
var<workgroup> sh: array<f32, 256>;
fn wg_sum(v: f32, tid: u32) -> f32 {
    sh[tid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sh[tid] = sh[tid] + sh[tid + s];
        }
        workgroupBarrier();
    }
    let r = sh[0];
    workgroupBarrier();
    return r;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let row = wg.y * nwg.x + wg.x;
    if (row >= p[0]) {
        return;
    }
    let d = p[1];
    let off = row * d;
    var s = 0.0;
    for (var l = tid; l < d; l += 256u) {
        s += x[off + l];
    }
    let mean = wg_sum(s, tid) / f32(d);
    var v = 0.0;
    for (var l = tid; l < d; l += 256u) {
        let c = x[off + l] - mean;
        v += c * c;
    }
    let r = 1.0 / sqrt(wg_sum(v, tid) / f32(d) + pf(2u));
    for (var l = tid; l < d; l += 256u) {
        xhat[off + l] = (x[off + l] - mean) * r;
    }
    if (tid == 0u) {
        rstd[row] = r;
    }
}

//! kernel: layer_norm_backward_rows
// p: rows, d
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> xhat: array<f32>;
@group(0) @binding(3) var<storage, read> rstd: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
var<workgroup> sh: array<f32, 256>;
fn wg_sum(v: f32, tid: u32) -> f32 {
    sh[tid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sh[tid] = sh[tid] + sh[tid + s];
        }
        workgroupBarrier();
    }
    let r = sh[0];
    workgroupBarrier();
    return r;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let row = wg.y * nwg.x + wg.x;
    if (row >= p[0]) {
        return;
    }
    let d = p[1];
    let off = row * d;
    var a = 0.0;
    var b = 0.0;
    for (var l = tid; l < d; l += 256u) {
        a += g[off + l];
        b += g[off + l] * xhat[off + l];
    }
    let mg = wg_sum(a, tid) / f32(d);
    let mgx = wg_sum(b, tid) / f32(d);
    let r = rstd[row];
    for (var l = tid; l < d; l += 256u) {
        out[off + l] = r * (g[off + l] - mg - xhat[off + l] * mgx);
    }
}

//! kernel: adam_step
// p: n, lr, beta1, beta2, eps, wd, decoupled, ib1, ib2
@group(0) @binding(1) var<storage, read> pa: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read> m: array<f32>;
@group(0) @binding(4) var<storage, read> v: array<f32>;
@group(0) @binding(5) var<storage, read_write> np: array<f32>;
@group(0) @binding(6) var<storage, read_write> nm: array<f32>;
@group(0) @binding(7) var<storage, read_write> nv: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = gid(wg, nwg, li);
    if (i >= p[0]) {
        return;
    }
    let lr = pf(1u);
    let b1 = pf(2u);
    let b2 = pf(3u);
    let wd = pf(5u);
    var gi = g[i];
    if (wd != 0.0 && p[6] == 0u) {
        gi += wd * pa[i];
    }
    let mi = b1 * m[i] + (1.0 - b1) * gi;
    let vi = b2 * v[i] + (1.0 - b2) * gi * gi;
    var pi = pa[i] - lr * (mi * pf(7u)) / (sqrt(vi * pf(8u)) + pf(4u));
    if (wd != 0.0 && p[6] != 0u) {
        pi -= lr * wd * pa[i];
    }
    np[i] = pi;
    nm[i] = mi;
    nv[i] = vi;
}

//! kernel: index_select
// p: n, cols — out[r, c] = t[idx[r], c] (any 32-bit dtype)
@group(0) @binding(1) var<storage, read> t: array<u32>;
@group(0) @binding(2) var<storage, read> idx: array<i32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = gid(wg, nwg, li);
    if (i < p[0]) {
        let cols = p[1];
        out[i] = t[u32(idx[i / cols]) * cols + i % cols];
    }
}

//! kernel: index_add
// p: n, cols — out[idx[r], c] += src[r, c]
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> idx: array<i32>;
@group(0) @binding(3) var<storage, read_write> out: array<atomic<u32>>;
fn add_f32(i: u32, v: f32) {
    var old = atomicLoad(&out[i]);
    loop {
        let r = atomicCompareExchangeWeak(&out[i], old, bitcast<u32>(bitcast<f32>(old) + v));
        if (r.exchanged) {
            break;
        }
        old = r.old_value;
    }
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let i = gid(wg, nwg, li);
    if (i < p[0]) {
        let cols = p[1];
        add_f32(u32(idx[i / cols]) * cols + i % cols, src[i]);
    }
}

//! kernel: gather
// p: n, len, idx_len, inner
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> idx: array<i32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = gid(wg, nwg, li);
    if (t < p[0]) {
        let inner = p[3];
        let o = t / (p[2] * inner);
        out[t] = x[(o * p[1] + u32(idx[t])) * inner + t % inner];
    }
}

//! kernel: scatter_add
// p: n, len, idx_len, inner
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> idx: array<i32>;
@group(0) @binding(3) var<storage, read_write> out: array<atomic<u32>>;
fn add_f32(i: u32, v: f32) {
    var old = atomicLoad(&out[i]);
    loop {
        let r = atomicCompareExchangeWeak(&out[i], old, bitcast<u32>(bitcast<f32>(old) + v));
        if (r.exchanged) {
            break;
        }
        old = r.old_value;
    }
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = gid(wg, nwg, li);
    if (t < p[0]) {
        let inner = p[3];
        let o = t / (p[2] * inner);
        add_f32((o * p[1] + u32(idx[t])) * inner + t % inner, src[t]);
    }
}

//! kernel: im2col
// p: n, c, h, w, kh, kw, stride, padding, oh, ow — cols [N, C*KH*KW, OH*OW]
@group(0) @binding(1) var<storage, read> x: array<u32>;
@group(0) @binding(2) var<storage, read_write> cols: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = gid(wg, nwg, li);
    let C = p[1];
    let H = p[2];
    let W = p[3];
    let KH = p[4];
    let KW = p[5];
    let OH = p[8];
    let OW = p[9];
    let l = OH * OW;
    if (t >= p[0] * C * KH * KW * l) {
        return;
    }
    let ox = t % OW;
    let oy = (t / OW) % OH;
    let r = t / l;
    let kj = r % KW;
    let ki = (r / KW) % KH;
    let c = (r / (KW * KH)) % C;
    let n = r / (C * KH * KW);
    let iy = i32(oy * p[6] + ki) - i32(p[7]);
    let ix = i32(ox * p[6] + kj) - i32(p[7]);
    var v = 0u;
    if (iy >= 0 && iy < i32(H) && ix >= 0 && ix < i32(W)) {
        v = x[((n * C + c) * H + u32(iy)) * W + u32(ix)];
    }
    cols[t] = v;
}

//! kernel: col2im
// p: n, c, h, w, kh, kw, stride, padding, oh, ow — one thread per input pixel
@group(0) @binding(1) var<storage, read> cols: array<f32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = gid(wg, nwg, li);
    let C = p[1];
    let H = p[2];
    let W = p[3];
    let KH = p[4];
    let KW = p[5];
    let S = p[6];
    let P = p[7];
    let OH = p[8];
    let OW = p[9];
    if (t >= p[0] * C * H * W) {
        return;
    }
    let px = t % W;
    let py = (t / W) % H;
    let c = (t / (W * H)) % C;
    let n = t / (W * H * C);
    let l = OH * OW;
    var acc = 0.0;
    for (var ki = 0u; ki < KH; ki++) {
        let ty = i32(py + P) - i32(ki);
        if (ty < 0 || ty % i32(S) != 0) {
            continue;
        }
        let oy = u32(ty) / S;
        if (oy >= OH) {
            continue;
        }
        for (var kj = 0u; kj < KW; kj++) {
            let tx = i32(px + P) - i32(kj);
            if (tx < 0 || tx % i32(S) != 0) {
                continue;
            }
            let ox = u32(tx) / S;
            if (ox >= OW) {
                continue;
            }
            let r = n * (C * KH * KW) + (c * KH + ki) * KW + kj;
            acc += cols[r * l + oy * OW + ox];
        }
    }
    x[t] = acc;
}

//! kernel: gemv
// p: m, k, n, trans_a, trans_b — few rows (m ≤ 8): out[i, j] = sum_l a[i, l] * b[l, j];
// a workgroup of 256 covers 32 columns of one row, 8 groups of 32 split k
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
var<workgroup> part: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let M = p[0];
    let K = p[1];
    let N = p[2];
    let ta = p[3] != 0u;
    let tb = p[4] != 0u;
    let tiles = (N + 31u) / 32u;
    let g = wg.y * nwg.x + wg.x;
    if (g >= M * tiles) {
        return;
    }
    let i = g / tiles;
    let lane = tid % 32u;
    let s = tid / 32u;
    let j = (g % tiles) * 32u + lane;
    var acc = 0.0;
    if (j < N) {
        for (var l = s; l < K; l += 8u) {
            acc += A[select(i * K + l, l * M + i, ta)] * B[select(l * N + j, j * K + l, tb)];
        }
    }
    part[tid] = acc;
    workgroupBarrier();
    if (s == 0u && j < N) {
        var t = 0.0;
        for (var r = 0u; r < 8u; r++) {
            t += part[r * 32u + lane];
        }
        out[i * N + j] = t;
    }
}

//! kernel: matmul64
// p: batch, m, k, n, trans_a, trans_b, a_batch_stride, b_batch_stride — for small
// matrices: a 64 x 64 tile per workgroup of 16 x 16 threads, each 4 x 4 (four
// vec4 rows), k in steps of 16 (the large-matrix GEMM is generated in webgpu.rs)
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
var<workgroup> As: array<f32, 1024>;
var<workgroup> Bs: array<vec4<f32>, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = p[1];
    let K = p[2];
    let N = p[3];
    let ta = p[4] != 0u;
    let tb = p[5] != 0u;
    let aoff = wg.z * p[6];
    let boff = wg.z * p[7];
    let row0 = wg.y * 64u;
    let col0 = wg.x * 64u;
    let t = lid.y * 16u + lid.x;
    var c0 = vec4<f32>(0.0);
    var c1 = vec4<f32>(0.0);
    var c2 = vec4<f32>(0.0);
    var c3 = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < K; k0 += 16u) {
        for (var e = 0u; e < 4u; e++) {
            let idx = t + e * 256u;
            var r = idx / 16u;
            var ka = idx % 16u;
            if (ta) {
                r = idx % 64u;
                ka = idx / 64u;
            }
            let gr = row0 + r;
            let gk = k0 + ka;
            var av = 0.0;
            if (gr < M && gk < K) {
                av = A[aoff + select(gr * K + gk, gk * M + gr, ta)];
            }
            As[ka * 64u + r] = av;
        }
        {
            let kb = t / 16u;
            let c4 = t % 16u;
            let gk = k0 + kb;
            var bv = vec4<f32>(0.0);
            if (gk < K) {
                for (var j = 0u; j < 4u; j++) {
                    let gc = col0 + c4 * 4u + j;
                    if (gc < N) {
                        bv[j] = B[boff + select(gk * N + gc, gc * K + gk, tb)];
                    }
                }
            }
            Bs[kb * 16u + c4] = bv;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk++) {
            let b = Bs[kk * 16u + lid.x];
            let ab = kk * 64u + lid.y * 4u;
            c0 += As[ab] * b;
            c1 += As[ab + 1u] * b;
            c2 += As[ab + 2u] * b;
            c3 += As[ab + 3u] * b;
        }
        workgroupBarrier();
    }
    let coff = wg.z * M * N;
    let r = row0 + lid.y * 4u;
    let c = col0 + lid.x * 4u;
    for (var j = 0u; j < 4u; j++) {
        if (c + j < N) {
            if (r < M) {
                C[coff + r * N + c + j] = c0[j];
            }
            if (r + 1u < M) {
                C[coff + (r + 1u) * N + c + j] = c1[j];
            }
            if (r + 2u < M) {
                C[coff + (r + 2u) * N + c + j] = c2[j];
            }
            if (r + 3u < M) {
                C[coff + (r + 3u) * N + c + j] = c3[j];
            }
        }
    }
}

//! kernel: matmul_int8
// p: m, k, n — out[i, j] = scale[j] * sum_l x[i, l] * q[l, j] (q: packed bytes).
// A workgroup of 256 covers 32 columns of one row; 8 groups of 32 split k.
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> q: array<u32>;
@group(0) @binding(3) var<storage, read> sc: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
var<workgroup> part: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let K = p[1];
    let N = p[2];
    let tiles = (N + 31u) / 32u;
    let g = wg.y * nwg.x + wg.x;
    if (g >= p[0] * tiles) {
        return;
    }
    let i = g / tiles;
    let lane = tid % 32u;
    let s = tid / 32u;
    let j = (g % tiles) * 32u + lane;
    var acc = 0.0;
    if (j < N) {
        for (var l = s; l < K; l += 8u) {
            let e = l * N + j;
            let b = (q[e >> 2u] >> ((e & 3u) * 8u)) & 0xffu;
            acc += x[i * K + l] * f32(bitcast<i32>(b << 24u) >> 24u);
        }
    }
    part[tid] = acc;
    workgroupBarrier();
    if (s == 0u && j < N) {
        var t = 0.0;
        for (var r = 0u; r < 8u; r++) {
            t += part[r * 32u + lane];
        }
        out[i * N + j] = t * sc[j];
    }
}

//! kernel: matmul_int4
// p: m, k, n, group — out[i, j] = sum_l x[i, l] * w[l, j] * scale[l / group, j];
// byte (l / 2) * n + j holds rows l (low nibble) and l + 1 (high nibble)
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> q: array<u32>;
@group(0) @binding(3) var<storage, read> sc: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
var<workgroup> part: array<f32, 256>;
fn nib(v: u32) -> f32 {
    return f32(select(i32(v), i32(v) - 16, v >= 8u));
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let K = p[1];
    let N = p[2];
    let G = p[3];
    let tiles = (N + 31u) / 32u;
    let g = wg.y * nwg.x + wg.x;
    if (g >= p[0] * tiles) {
        return;
    }
    let i = g / tiles;
    let lane = tid % 32u;
    let s = tid / 32u;
    let j = (g % tiles) * 32u + lane;
    var acc = 0.0;
    if (j < N) {
        for (var r = s; r < K / 2u; r += 8u) {
            let e = r * N + j;
            let b = (q[e >> 2u] >> ((e & 3u) * 8u)) & 0xffu;
            let l = 2u * r;
            let s0 = sc[(l / G) * N + j];
            acc += (x[i * K + l] * nib(b & 15u) + x[i * K + l + 1u] * nib(b >> 4u)) * s0;
        }
    }
    part[tid] = acc;
    workgroupBarrier();
    if (s == 0u && j < N) {
        var t = 0.0;
        for (var r = 0u; r < 8u; r++) {
            t += part[r * 32u + lane];
        }
        out[i * N + j] = t;
    }
}

//! kernel: dequant_int4
// p: m (unused), k, n, group
@group(0) @binding(1) var<storage, read> q: array<u32>;
@group(0) @binding(2) var<storage, read> sc: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let t = gid(wg, nwg, li);
    let K = p[1];
    let N = p[2];
    if (t >= K * N) {
        return;
    }
    let l = t / N;
    let j = t % N;
    let e = (l >> 1u) * N + j;
    let b = (q[e >> 2u] >> ((e & 3u) * 8u)) & 0xffu;
    let v = select(b & 15u, b >> 4u, (l & 1u) == 1u);
    out[t] = f32(select(i32(v), i32(v) - 16, v >= 8u)) * sc[(l / p[3]) * N + j];
}

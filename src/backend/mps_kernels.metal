// Metal compute kernels for the MPS backend.
// Compiled at runtime (newLibraryWithSource) with safe math, so results
// match the CPU backend closely. Parameter structs mirror `#[repr(C)]`
// structs in mps.rs; all sizes are 32-bit.

#include <metal_stdlib>
using namespace metal;

#define MAX_RANK 8

// ---------------------------------------------------------------- memory

kernel void fill_u32(device uint* out [[buffer(0)]],
                     constant uint& n [[buffer(1)]],
                     constant uint& bits [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = bits;
}

kernel void cast_f32_i32(device const float* x [[buffer(0)]], device int* y [[buffer(1)]],
                         constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = int(x[i]);
}

kernel void cast_i32_f32(device const int* x [[buffer(0)]], device float* y [[buffer(1)]],
                         constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = float(x[i]);
}

// ---------------------------------------------------------------- elementwise

inline float gelu_f(float x) {
    const float S = 0.7978845608f;
    return 0.5f * x * (1.0f + tanh(S * (x + 0.044715f * x * x * x)));
}

inline float gelu_grad_f(float x) {
    const float S = 0.7978845608f;
    float u = S * (x + 0.044715f * x * x * x);
    float t = tanh(u);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * S * (1.0f + 3.0f * 0.044715f * x * x);
}

// op codes must match UnaryOp order in mps.rs
kernel void unary_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                      constant uint& n [[buffer(2)]], constant uint& op [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float v = x[i];
    float r;
    switch (op) {
        case 0: r = -v; break;
        case 1: r = exp(v); break;
        case 2: r = log(v); break;
        case 3: r = sqrt(v); break;
        case 4: r = 1.0f / v; break;
        case 5: r = fabs(v); break;
        case 6: r = tanh(v); break;
        case 7: r = 1.0f / (1.0f + exp(-v)); break;
        case 8: r = max(v, 0.0f); break;
        case 9: r = gelu_f(v); break;
        case 10: r = gelu_grad_f(v); break;
        case 11: r = sin(v); break;
        default: r = cos(v); break;
    }
    y[i] = r;
}

struct BinParams {
    uint n;
    uint rank;
    uint op;
    uint pad;
    uint shape[MAX_RANK];
    uint sa[MAX_RANK];
    uint sb[MAX_RANK];
};

// op codes must match BinaryOp order in mps.rs
inline float binop(uint op, float a, float b) {
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: return a / b;
        case 4: return pow(a, b);
        case 5: return max(a, b);
        case 6: return min(a, b);
        case 7: return a == b ? 1.0f : 0.0f;
        case 8: return a > b ? 1.0f : 0.0f;
        default: return a < b ? 1.0f : 0.0f;
    }
}

kernel void binary_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                       device float* out [[buffer(2)]], constant BinParams& p [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    uint rem = i, oa = 0, ob = 0;
    for (int d = int(p.rank) - 1; d >= 0; --d) {
        uint c = rem % p.shape[d];
        rem /= p.shape[d];
        oa += c * p.sa[d];
        ob += c * p.sb[d];
    }
    out[i] = binop(p.op, a[oa], b[ob]);
}

struct CopyParams {
    uint n;
    uint rank;
    uint base;
    uint pad;
    uint shape[MAX_RANK];
    uint strides[MAX_RANK];
};

// out[i] = x[base + strided offset of i]  (dtype agnostic: copies 32-bit words)
// Used for expand (stride 0), permute and narrow.
kernel void strided_copy(device const uint* x [[buffer(0)]], device uint* out [[buffer(1)]],
                         constant CopyParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    uint rem = i, o = p.base;
    for (int d = int(p.rank) - 1; d >= 0; --d) {
        uint c = rem % p.shape[d];
        rem /= p.shape[d];
        o += c * p.strides[d];
    }
    out[i] = x[o];
}

struct CatParams {
    uint n;      // elements of this input
    uint blk_in;  // input block (axis_len * inner)
    uint blk_out; // output block
    uint offset;  // offset of this input inside an output block
};

kernel void cat_copy(device const uint* x [[buffer(0)]], device uint* out [[buffer(1)]],
                     constant CatParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    uint o = i / p.blk_in, j = i % p.blk_in;
    out[o * p.blk_out + p.offset + j] = x[i];
}

kernel void where_f32(device const float* c [[buffer(0)]], device const float* a [[buffer(1)]],
                      device const float* b [[buffer(2)]], device float* out [[buffer(3)]],
                      constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = c[i] != 0.0f ? a[i] : b[i];
}

// ---------------------------------------------------------------- reductions

struct RedParams {
    uint outer;
    uint len;
    uint inner;
    uint op; // 0 sum, 1 max, 2 min, 3 argmax
};

inline float red_init(uint op) {
    return op == 0 ? 0.0f : (op == 2 ? INFINITY : -INFINITY);
}

inline float red_combine(uint op, float a, float b) {
    return op == 0 ? a + b : (op == 2 ? min(a, b) : max(a, b));
}

inline float simd_combine(uint op, float v) {
    return op == 0 ? simd_sum(v) : (op == 2 ? simd_min(v) : simd_max(v));
}

// One thread per output element (used when inner > 1, and for argmax).
kernel void reduce_strided(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                           constant RedParams& p [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    if (t >= p.outer * p.inner) return;
    uint o = t / p.inner, i = t % p.inner;
    uint base = o * p.len * p.inner + i;
    if (p.op == 3) {
        float best = -INFINITY;
        uint bi = 0;
        for (uint l = 0; l < p.len; ++l) {
            float v = x[base + l * p.inner];
            if (v > best) { best = v; bi = l; }
        }
        ((device int*)out)[t] = int(bi);
        return;
    }
    float acc = red_init(p.op);
    for (uint l = 0; l < p.len; ++l) acc = red_combine(p.op, acc, x[base + l * p.inner]);
    out[t] = acc;
}

// Threadgroup-wide reduction helpers. `buf` must hold 32 floats.
inline float tg_reduce(uint op, float v, threadgroup float* buf, uint lane, uint sg, uint nsg) {
    v = simd_combine(op, v);
    if (lane == 0) buf[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? buf[lane] : red_init(op);
    r = simd_combine(op, r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

#define ROW_ARGS                                                      \
    uint row [[threadgroup_position_in_grid]],                        \
    uint tid [[thread_position_in_threadgroup]],                      \
    uint tg [[threads_per_threadgroup]],                              \
    uint lane [[thread_index_in_simdgroup]],                          \
    uint sg [[simdgroup_index_in_threadgroup]]

// One threadgroup per row (inner == 1).
kernel void reduce_rows(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                        constant RedParams& p [[buffer(2)]], ROW_ARGS) {
    threadgroup float buf[32];
    uint nsg = (tg + 31) / 32;
    device const float* xr = x + row * p.len;
    float acc = red_init(p.op);
    for (uint l = tid; l < p.len; l += tg) acc = red_combine(p.op, acc, xr[l]);
    acc = tg_reduce(p.op, acc, buf, lane, sg, nsg);
    if (tid == 0) out[row] = acc;
}

// ---------------------------------------------------------------- fused row ops

kernel void softmax_rows(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                         constant uint& d [[buffer(2)]], constant uint& log_mode [[buffer(3)]], ROW_ARGS) {
    threadgroup float buf[32];
    uint nsg = (tg + 31) / 32;
    device const float* xr = x + row * d;
    device float* yr = y + row * d;
    float m = -INFINITY;
    for (uint l = tid; l < d; l += tg) m = max(m, xr[l]);
    m = tg_reduce(1, m, buf, lane, sg, nsg);
    float s = 0.0f;
    for (uint l = tid; l < d; l += tg) s += exp(xr[l] - m);
    s = tg_reduce(0, s, buf, lane, sg, nsg);
    if (log_mode != 0) {
        float lse = m + log(s);
        for (uint l = tid; l < d; l += tg) yr[l] = xr[l] - lse;
    } else {
        float inv = 1.0f / s;
        for (uint l = tid; l < d; l += tg) yr[l] = exp(xr[l] - m) * inv;
    }
}

kernel void softmax_backward_rows(device const float* y [[buffer(0)]], device const float* g [[buffer(1)]],
                                  device float* out [[buffer(2)]], constant uint& d [[buffer(3)]],
                                  constant uint& log_mode [[buffer(4)]], ROW_ARGS) {
    threadgroup float buf[32];
    uint nsg = (tg + 31) / 32;
    uint off = row * d;
    float s = 0.0f;
    if (log_mode != 0) {
        for (uint l = tid; l < d; l += tg) s += g[off + l];
    } else {
        for (uint l = tid; l < d; l += tg) s += g[off + l] * y[off + l];
    }
    s = tg_reduce(0, s, buf, lane, sg, nsg);
    for (uint l = tid; l < d; l += tg) {
        float yv = y[off + l], gv = g[off + l];
        out[off + l] = log_mode != 0 ? gv - exp(yv) * s : yv * (gv - s);
    }
}

kernel void layer_norm_rows(device const float* x [[buffer(0)]], device float* xhat [[buffer(1)]],
                            device float* rstd [[buffer(2)]], constant uint& d [[buffer(3)]],
                            constant float& eps [[buffer(4)]], ROW_ARGS) {
    threadgroup float buf[32];
    uint nsg = (tg + 31) / 32;
    uint off = row * d;
    float s = 0.0f;
    for (uint l = tid; l < d; l += tg) s += x[off + l];
    float mean = tg_reduce(0, s, buf, lane, sg, nsg) / float(d);
    float v = 0.0f;
    for (uint l = tid; l < d; l += tg) { float c = x[off + l] - mean; v += c * c; }
    float r = 1.0f / sqrt(tg_reduce(0, v, buf, lane, sg, nsg) / float(d) + eps);
    for (uint l = tid; l < d; l += tg) xhat[off + l] = (x[off + l] - mean) * r;
    if (tid == 0) rstd[row] = r;
}

kernel void layer_norm_backward_rows(device const float* g [[buffer(0)]], device const float* xhat [[buffer(1)]],
                                     device const float* rstd [[buffer(2)]], device float* out [[buffer(3)]],
                                     constant uint& d [[buffer(4)]], ROW_ARGS) {
    threadgroup float buf[32];
    uint nsg = (tg + 31) / 32;
    uint off = row * d;
    float a = 0.0f, b = 0.0f;
    for (uint l = tid; l < d; l += tg) { a += g[off + l]; b += g[off + l] * xhat[off + l]; }
    float mg = tg_reduce(0, a, buf, lane, sg, nsg) / float(d);
    float mgx = tg_reduce(0, b, buf, lane, sg, nsg) / float(d);
    float r = rstd[row];
    for (uint l = tid; l < d; l += tg) out[off + l] = r * (g[off + l] - mg - xhat[off + l] * mgx);
}

struct AdamParams {
    uint n;
    float lr, beta1, beta2, eps, wd;
    uint decoupled;
    float ib1, ib2; // 1 / bias corrections
};

kernel void adam_step(device const float* p [[buffer(0)]], device const float* g [[buffer(1)]],
                      device const float* m [[buffer(2)]], device const float* v [[buffer(3)]],
                      device float* np [[buffer(4)]], device float* nm [[buffer(5)]], device float* nv [[buffer(6)]],
                      constant AdamParams& h [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i >= h.n) return;
    float gi = g[i];
    if (h.wd != 0.0f && h.decoupled == 0) gi += h.wd * p[i];
    float mi = h.beta1 * m[i] + (1.0f - h.beta1) * gi;
    float vi = h.beta2 * v[i] + (1.0f - h.beta2) * gi * gi;
    float pi = p[i] - h.lr * (mi * h.ib1) / (sqrt(vi * h.ib2) + h.eps);
    if (h.wd != 0.0f && h.decoupled != 0) pi -= h.lr * h.wd * p[i];
    np[i] = pi;
    nm[i] = mi;
    nv[i] = vi;
}

// ---------------------------------------------------------------- indexing

kernel void index_select(device const float* t [[buffer(0)]], device const int* idx [[buffer(1)]],
                         device float* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                         constant uint& cols [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint r = i / cols, c = i % cols;
    out[i] = t[uint(idx[r]) * cols + c];
}

kernel void index_add(device const float* src [[buffer(0)]], device const int* idx [[buffer(1)]],
                      device atomic_float* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                      constant uint& cols [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint r = i / cols, c = i % cols;
    atomic_fetch_add_explicit(&out[uint(idx[r]) * cols + c], src[i], memory_order_relaxed);
}

struct GatherParams {
    uint n;   // number of index elements
    uint len; // size of the gathered axis in x
    uint idx_len;
    uint inner;
};

kernel void gather(device const float* x [[buffer(0)]], device const int* idx [[buffer(1)]],
                   device float* out [[buffer(2)]], constant GatherParams& p [[buffer(3)]],
                   uint t [[thread_position_in_grid]]) {
    if (t >= p.n) return;
    uint i = t % p.inner, o = t / (p.idx_len * p.inner);
    out[t] = x[(o * p.len + uint(idx[t])) * p.inner + i];
}

kernel void scatter_add(device const float* src [[buffer(0)]], device const int* idx [[buffer(1)]],
                        device atomic_float* out [[buffer(2)]], constant GatherParams& p [[buffer(3)]],
                        uint t [[thread_position_in_grid]]) {
    if (t >= p.n) return;
    uint i = t % p.inner, o = t / (p.idx_len * p.inner);
    atomic_fetch_add_explicit(&out[(o * p.len + uint(idx[t])) * p.inner + i], src[t], memory_order_relaxed);
}

// ---------------------------------------------------------------- convolution

struct ConvParams {
    uint n, c, h, w, kh, kw, stride, padding, oh, ow;
};

// cols [N, C*KH*KW, OH*OW]
kernel void im2col(device const float* x [[buffer(0)]], device float* cols [[buffer(1)]],
                   constant ConvParams& g [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    uint l = g.oh * g.ow;
    uint total = g.n * g.c * g.kh * g.kw * l;
    if (t >= total) return;
    uint ox = t % g.ow;
    uint oy = (t / g.ow) % g.oh;
    uint r = t / l;
    uint kj = r % g.kw;
    uint ki = (r / g.kw) % g.kh;
    uint c = (r / (g.kw * g.kh)) % g.c;
    uint n = r / (g.c * g.kh * g.kw);
    int iy = int(oy * g.stride + ki) - int(g.padding);
    int ix = int(ox * g.stride + kj) - int(g.padding);
    float v = 0.0f;
    if (iy >= 0 && iy < int(g.h) && ix >= 0 && ix < int(g.w)) {
        v = x[((n * g.c + c) * g.h + uint(iy)) * g.w + uint(ix)];
    }
    cols[t] = v;
}

// one thread per input pixel: gathers every column entry that read it
kernel void col2im(device const float* cols [[buffer(0)]], device float* x [[buffer(1)]],
                   constant ConvParams& g [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    uint total = g.n * g.c * g.h * g.w;
    if (t >= total) return;
    uint px = t % g.w;
    uint py = (t / g.w) % g.h;
    uint c = (t / (g.w * g.h)) % g.c;
    uint n = t / (g.w * g.h * g.c);
    uint l = g.oh * g.ow;
    float acc = 0.0f;
    for (uint ki = 0; ki < g.kh; ++ki) {
        int ty = int(py + g.padding) - int(ki);
        if (ty < 0 || ty % int(g.stride) != 0) continue;
        uint oy = uint(ty) / g.stride;
        if (oy >= g.oh) continue;
        for (uint kj = 0; kj < g.kw; ++kj) {
            int tx = int(px + g.padding) - int(kj);
            if (tx < 0 || tx % int(g.stride) != 0) continue;
            uint ox = uint(tx) / g.stride;
            if (ox >= g.ow) continue;
            uint r = n * (g.c * g.kh * g.kw) + (c * g.kh + ki) * g.kw + kj;
            acc += cols[r * l + oy * g.ow + ox];
        }
    }
    x[t] = acc;
}

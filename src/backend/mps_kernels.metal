// Metal compute kernels for the MPS backend.
// Compiled at runtime (newLibraryWithSource) with safe math, so results
// match the CPU backend closely. Parameter structs mirror `#[repr(C)]`
// structs in mps.rs; all sizes are 32-bit.
//
// Float kernels are templates over the storage type (float, half, bf16_t)
// and always compute in f32: `ld` widens a stored value, `st<T>` rounds back.
// Names: <kernel>_<f32|f16|bf16>.

#include <metal_stdlib>
using namespace metal;

#define MAX_RANK 8

// explicit instantiation under a host-visible name (pattern from MLX)
#define instantiate(name, func, ...) \
    template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;
#define instantiate_floats(base, func)              \
    instantiate(base "_f32", func, float)           \
    instantiate(base "_f16", func, half)            \
    instantiate(base "_bf16", func, bf16_t)

// bfloat16 stored as raw bits (no dependency on Metal 3.1's `bfloat`)
struct bf16_t {
    ushort bits;
};

inline float ld(float v) { return v; }
inline float ld(half v) { return float(v); }
inline float ld(bf16_t v) { return as_type<float>(uint(v.bits) << 16); }
inline float ld(int v) { return float(v); }
inline float ld(char v) { return float(v); }

template <typename T> inline T st(float f);
template <> inline float st<float>(float f) { return f; }
template <> inline half st<half>(float f) { return half(f); }
template <> inline int st<int>(float f) { return int(f); }
template <> inline char st<char>(float f) { return char(clamp(f, -128.0f, 127.0f)); }
template <> inline bf16_t st<bf16_t>(float f) {
    uint u = as_type<uint>(f);
    bf16_t r;
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        r.bits = 0x7fc0; // NaN
    } else {
        u += 0x7fffu + ((u >> 16) & 1u); // round to nearest even
        r.bits = ushort(u >> 16);
    }
    return r;
}

// ---------------------------------------------------------------- memory

template <typename W>
kernel void fill_t(device W* out [[buffer(0)]],
                   constant uint& n [[buffer(1)]],
                   constant uint& bits [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = W(bits);
}
instantiate("fill_u32", fill_t, uint)
instantiate("fill_u16", fill_t, ushort)
instantiate("fill_u8", fill_t, uchar)

template <typename S, typename D>
kernel void cast_t(device const S* x [[buffer(0)]], device D* y [[buffer(1)]],
                   constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = st<D>(ld(x[i]));
}
instantiate("cast_f32_f16", cast_t, float, half)
instantiate("cast_f32_bf16", cast_t, float, bf16_t)
instantiate("cast_f32_i32", cast_t, float, int)
instantiate("cast_f16_f32", cast_t, half, float)
instantiate("cast_f16_bf16", cast_t, half, bf16_t)
instantiate("cast_f16_i32", cast_t, half, int)
instantiate("cast_bf16_f32", cast_t, bf16_t, float)
instantiate("cast_bf16_f16", cast_t, bf16_t, half)
instantiate("cast_bf16_i32", cast_t, bf16_t, int)
instantiate("cast_i32_f32", cast_t, int, float)
instantiate("cast_i32_f16", cast_t, int, half)
instantiate("cast_i32_bf16", cast_t, int, bf16_t)
instantiate("cast_i8_f32", cast_t, char, float)
instantiate("cast_i8_f16", cast_t, char, half)
instantiate("cast_i8_bf16", cast_t, char, bf16_t)
instantiate("cast_i8_i32", cast_t, char, int)
instantiate("cast_f32_i8", cast_t, float, char)
instantiate("cast_f16_i8", cast_t, half, char)
instantiate("cast_bf16_i8", cast_t, bf16_t, char)
instantiate("cast_i32_i8", cast_t, int, char)

// ---------------------------------------------------------------- elementwise

// Metal's tanh returns NaN for arguments above ~45 (GPT-2 has activations in
// the thousands); tanh(±15) is already ±1 in f32.
inline float tanh_f(float x) {
    return tanh(clamp(x, -15.0f, 15.0f));
}

inline float gelu_f(float x) {
    const float S = 0.7978845608f;
    return 0.5f * x * (1.0f + tanh_f(S * (x + 0.044715f * x * x * x)));
}

inline float gelu_grad_f(float x) {
    const float S = 0.7978845608f;
    float u = S * (x + 0.044715f * x * x * x);
    float t = tanh_f(u);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * S * (1.0f + 3.0f * 0.044715f * x * x);
}

// op codes must match UnaryOp order in mps.rs
template <typename T>
kernel void unary_t(device const T* x [[buffer(0)]], device T* y [[buffer(1)]],
                    constant uint& n [[buffer(2)]], constant uint& op [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float v = ld(x[i]);
    float r;
    switch (op) {
        case 0: r = -v; break;
        case 1: r = exp(v); break;
        case 2: r = log(v); break;
        case 3: r = sqrt(v); break;
        case 4: r = 1.0f / v; break;
        case 5: r = fabs(v); break;
        case 6: r = tanh_f(v); break;
        case 7: r = 1.0f / (1.0f + exp(-v)); break;
        case 8: r = max(v, 0.0f); break;
        case 9: r = gelu_f(v); break;
        case 10: r = gelu_grad_f(v); break;
        case 11: r = sin(v); break;
        default: r = cos(v); break;
    }
    y[i] = st<T>(r);
}
instantiate_floats("unary", unary_t)

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

template <typename T>
kernel void binary_t(device const T* a [[buffer(0)]], device const T* b [[buffer(1)]],
                     device T* out [[buffer(2)]], constant BinParams& p [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    uint rem = i, oa = 0, ob = 0;
    for (int d = int(p.rank) - 1; d >= 0; --d) {
        uint c = rem % p.shape[d];
        rem /= p.shape[d];
        oa += c * p.sa[d];
        ob += c * p.sb[d];
    }
    out[i] = st<T>(binop(p.op, ld(a[oa]), ld(b[ob])));
}
instantiate_floats("binary", binary_t)

struct CopyParams {
    uint n;
    uint rank;
    uint base;
    uint pad;
    uint shape[MAX_RANK];
    uint strides[MAX_RANK];
};

// out[i] = x[base + strided offset of i]  (dtype agnostic: copies 32- or
// 16-bit words). Used for expand (stride 0), permute and narrow.
template <typename W>
kernel void strided_copy_t(device const W* x [[buffer(0)]], device W* out [[buffer(1)]],
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
instantiate("strided_copy_u32", strided_copy_t, uint)
instantiate("strided_copy_u16", strided_copy_t, ushort)
instantiate("strided_copy_u8", strided_copy_t, uchar)

struct CatParams {
    uint n;      // elements of this input
    uint blk_in;  // input block (axis_len * inner)
    uint blk_out; // output block
    uint offset;  // offset of this input inside an output block
};

template <typename W>
kernel void cat_copy_t(device const W* x [[buffer(0)]], device W* out [[buffer(1)]],
                       constant CatParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    uint o = i / p.blk_in, j = i % p.blk_in;
    out[o * p.blk_out + p.offset + j] = x[i];
}
instantiate("cat_copy_u32", cat_copy_t, uint)
instantiate("cat_copy_u16", cat_copy_t, ushort)
instantiate("cat_copy_u8", cat_copy_t, uchar)

template <typename T>
kernel void where_t(device const T* c [[buffer(0)]], device const T* a [[buffer(1)]],
                    device const T* b [[buffer(2)]], device T* out [[buffer(3)]],
                    constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = ld(c[i]) != 0.0f ? a[i] : b[i];
}
instantiate_floats("where", where_t)

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
template <typename T>
kernel void reduce_strided_t(device const T* x [[buffer(0)]], device T* out [[buffer(1)]],
                             constant RedParams& p [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    if (t >= p.outer * p.inner) return;
    uint o = t / p.inner, i = t % p.inner;
    uint base = o * p.len * p.inner + i;
    if (p.op == 3) {
        float best = -INFINITY;
        uint bi = 0;
        for (uint l = 0; l < p.len; ++l) {
            float v = ld(x[base + l * p.inner]);
            if (v > best) { best = v; bi = l; }
        }
        ((device int*)out)[t] = int(bi);
        return;
    }
    float acc = red_init(p.op);
    for (uint l = 0; l < p.len; ++l) acc = red_combine(p.op, acc, ld(x[base + l * p.inner]));
    out[t] = st<T>(acc);
}
instantiate_floats("reduce_strided", reduce_strided_t)

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
template <typename T>
kernel void reduce_rows_t(device const T* x [[buffer(0)]], device T* out [[buffer(1)]],
                          constant RedParams& p [[buffer(2)]], ROW_ARGS) {
    threadgroup float buf[32];
    uint nsg = (tg + 31) / 32;
    device const T* xr = x + row * p.len;
    float acc = red_init(p.op);
    for (uint l = tid; l < p.len; l += tg) acc = red_combine(p.op, acc, ld(xr[l]));
    acc = tg_reduce(p.op, acc, buf, lane, sg, nsg);
    if (tid == 0) out[row] = st<T>(acc);
}
instantiate_floats("reduce_rows", reduce_rows_t)

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

template <typename T>
kernel void index_select_t(device const T* t [[buffer(0)]], device const int* idx [[buffer(1)]],
                           device T* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                           constant uint& cols [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    uint r = i / cols, c = i % cols;
    out[i] = t[uint(idx[r]) * cols + c];
}
instantiate_floats("index_select", index_select_t)

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
template <typename T>
kernel void im2col_t(device const T* x [[buffer(0)]], device T* cols [[buffer(1)]],
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
    T v = st<T>(0.0f);
    if (iy >= 0 && iy < int(g.h) && ix >= 0 && ix < int(g.w)) {
        v = x[((n * g.c + c) * g.h + uint(iy)) * g.w + uint(ix)];
    }
    cols[t] = v;
}
instantiate_floats("im2col", im2col_t)

// one thread per input pixel: gathers every column entry that read it
template <typename T>
kernel void col2im_t(device const T* cols [[buffer(0)]], device T* x [[buffer(1)]],
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
            acc += ld(cols[r * l + oy * g.ow + ox]);
        }
    }
    x[t] = st<T>(acc);
}
instantiate_floats("col2im", col2im_t)

// ---------------------------------------------------------------- int8 weights

struct Int8Params {
    uint m, k, n;
};

// out[i, j] = scale[j] * sum_l x[i, l] * q[l, j], for small m (decoding).
// Split-k: a threadgroup of 256 threads covers 32 columns of one row; its 8
// simdgroups each sum every 8th l (lanes read neighbouring weight bytes) and
// the partial sums are added through threadgroup memory.
template <typename T>
kernel void matmul_int8_t(device const T* x [[buffer(0)]], device const char* q [[buffer(1)]],
                          device const float* sc [[buffer(2)]], device T* out [[buffer(3)]],
                          constant Int8Params& p [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float part[8][32];
    uint tiles = (p.n + 31) / 32;
    uint i = tgid / tiles, lane = tid % 32, s = tid / 32;
    uint j = (tgid % tiles) * 32 + lane;
    float acc = 0.0f;
    if (j < p.n) {
        device const T* xr = x + i * p.k;
        for (uint l = s; l < p.k; l += 8) acc += ld(xr[l]) * float(q[l * p.n + j]);
    }
    part[s][lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s == 0 && j < p.n) {
        float t = 0.0f;
        for (uint r = 0; r < 8; ++r) t += part[r][lane];
        out[i * p.n + j] = st<T>(t * sc[j]);
    }
}
instantiate_floats("matmul_int8", matmul_int8_t)

// ---------------------------------------------------------------- int4 weights

struct Int4Params {
    uint m, k, n, group;
};

// Row l, column j of packed int4 weights: byte (l / 2) * n + j, low nibble
// for even l, high for odd l, 4-bit two's complement.
inline float int4_at(device const uchar* q, uint l, uint j, uint n) {
    uchar b = q[(l >> 1) * n + j];
    int v = (l & 1) ? int(b >> 4) : int(b & 15);
    return float(v >= 8 ? v - 16 : v);
}

// out[i, j] = sum_l x[i, l] * w[l, j] * scale[l / group, j], for small m.
// Split-k like matmul_int8_t: each of the 8 simdgroups takes every 8th
// pair of rows (one packed byte) for 32 columns.
template <typename T>
kernel void matmul_int4_t(device const T* x [[buffer(0)]], device const uchar* q [[buffer(1)]],
                          device const float* sc [[buffer(2)]], device T* out [[buffer(3)]],
                          constant Int4Params& p [[buffer(4)]], uint tgid [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float part[8][32];
    uint tiles = (p.n + 31) / 32;
    uint i = tgid / tiles, lane = tid % 32, s = tid / 32;
    uint j = (tgid % tiles) * 32 + lane;
    float acc = 0.0f;
    if (j < p.n) {
        device const T* xr = x + i * p.k;
        for (uint r = s; r < p.k / 2; r += 8) {
            uchar b = q[r * p.n + j];
            int lo = int(b & 15), hi = int(b >> 4);
            uint l = 2 * r;
            float s0 = sc[(l / p.group) * p.n + j];
            acc += (ld(xr[l]) * float(lo >= 8 ? lo - 16 : lo) + ld(xr[l + 1]) * float(hi >= 8 ? hi - 16 : hi)) * s0;
        }
    }
    part[s][lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s == 0 && j < p.n) {
        float t = 0.0f;
        for (uint r = 0; r < 8; ++r) t += part[r][lane];
        out[i * p.n + j] = st<T>(t);
    }
}
instantiate_floats("matmul_int4", matmul_int4_t)

kernel void dequant_int4(device const uchar* q [[buffer(0)]], device const float* sc [[buffer(1)]],
                         device float* out [[buffer(2)]], constant Int4Params& p [[buffer(3)]],
                         uint t [[thread_position_in_grid]]) {
    if (t >= p.k * p.n) return;
    uint l = t / p.n, j = t % p.n;
    out[t] = int4_at(q, l, j, p.n) * sc[(l / p.group) * p.n + j];
}

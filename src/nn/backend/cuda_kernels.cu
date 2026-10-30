// CUDA kernels for the CUDA backend; a one-to-one port of mps_kernels.metal.
// Compiled at runtime with NVRTC (precise math, no fast-math, no headers).
// Parameter structs are passed by value and mirror `#[repr(C)]` structs in
// cuda.rs. Kernel names must match backend/kernel_names.rs.
//
// Float kernels are templates over the storage type (float, f16_t, bf16_t)
// and always compute in f32: `ld` widens a stored value, `st<T>` rounds back.

#define MAX_RANK 8
#define INF __int_as_float(0x7f800000)

typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned long long u64; // NVRTC has no <cstddef>

// half types as raw bits (no cuda_fp16.h / cuda_bf16.h needed)
struct f16_t { u16 bits; };
struct bf16_t { u16 bits; };

__device__ __forceinline__ float ld(float v) { return v; }
__device__ __forceinline__ float ld(int v) { return (float)v; }
__device__ __forceinline__ float ld(signed char v) { return (float)v; }
__device__ __forceinline__ float ld(f16_t v) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(v.bits));
    return f;
}
__device__ __forceinline__ float ld(bf16_t v) { return __int_as_float(((u32)v.bits) << 16); }

template <typename T> __device__ __forceinline__ T st(float f);
template <> __device__ __forceinline__ float st<float>(float f) { return f; }
template <> __device__ __forceinline__ int st<int>(float f) { return (int)f; }
template <> __device__ __forceinline__ signed char st<signed char>(float f) { return (signed char)fminf(fmaxf(f, -128.0f), 127.0f); }
template <> __device__ __forceinline__ f16_t st<f16_t>(float f) {
    f16_t r;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r.bits) : "f"(f));
    return r;
}
template <> __device__ __forceinline__ bf16_t st<bf16_t>(float f) {
    u32 u = (u32)__float_as_int(f);
    bf16_t r;
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        r.bits = 0x7fc0; // NaN
    } else {
        u += 0x7fffu + ((u >> 16) & 1u); // round to nearest even
        r.bits = (u16)(u >> 16);
    }
    return r;
}

__device__ __forceinline__ u32 gid() { return blockIdx.x * blockDim.x + threadIdx.x; }

// MAKE(suffix, type) for each float storage type
#define FOR_FLOATS(MAKE) MAKE(f32, float) MAKE(f16, f16_t) MAKE(bf16, bf16_t)

// ---------------------------------------------------------------- memory

template <typename W> __device__ void fill_impl(W* out, u32 n, u32 bits) {
    u32 i = gid();
    if (i < n) out[i] = (W)bits;
}
extern "C" __global__ void fill_u32(u32* out, u32 n, u32 bits) { fill_impl<u32>(out, n, bits); }
extern "C" __global__ void fill_u16(u16* out, u32 n, u32 bits) { fill_impl<u16>(out, n, bits); }
extern "C" __global__ void fill_u8(unsigned char* out, u32 n, u32 bits) { fill_impl<unsigned char>(out, n, bits); }

template <typename S, typename D> __device__ void cast_impl(const S* x, D* y, u32 n) {
    u32 i = gid();
    if (i < n) y[i] = st<D>(ld(x[i]));
}
#define CAST(sn, S, dn, D) \
    extern "C" __global__ void cast_##sn##_##dn(const S* x, D* y, u32 n) { cast_impl<S, D>(x, y, n); }
CAST(f32, float, f16, f16_t)
CAST(f32, float, bf16, bf16_t)
CAST(f32, float, i32, int)
CAST(f16, f16_t, f32, float)
CAST(f16, f16_t, bf16, bf16_t)
CAST(f16, f16_t, i32, int)
CAST(bf16, bf16_t, f32, float)
CAST(bf16, bf16_t, f16, f16_t)
CAST(bf16, bf16_t, i32, int)
CAST(i32, int, f32, float)
CAST(i32, int, f16, f16_t)
CAST(i32, int, bf16, bf16_t)
CAST(i8, signed char, f32, float)
CAST(i8, signed char, f16, f16_t)
CAST(i8, signed char, bf16, bf16_t)
CAST(i8, signed char, i32, int)
CAST(f32, float, i8, signed char)
CAST(f16, f16_t, i8, signed char)
CAST(bf16, bf16_t, i8, signed char)
CAST(i32, int, i8, signed char)

// ---------------------------------------------------------------- elementwise

__device__ __forceinline__ float gelu_f(float x) {
    const float S = 0.7978845608f;
    return 0.5f * x * (1.0f + tanhf(S * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ float gelu_grad_f(float x) {
    const float S = 0.7978845608f;
    float u = S * (x + 0.044715f * x * x * x);
    float t = tanhf(u);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * S * (1.0f + 3.0f * 0.044715f * x * x);
}

// op codes must match unary_code() in cuda.rs
template <typename T> __device__ void unary_impl(const T* x, T* y, u32 n, u32 op) {
    u32 i = gid();
    if (i >= n) return;
    float v = ld(x[i]);
    float r;
    switch (op) {
        case 0: r = -v; break;
        case 1: r = expf(v); break;
        case 2: r = logf(v); break;
        case 3: r = sqrtf(v); break;
        case 4: r = 1.0f / v; break;
        case 5: r = fabsf(v); break;
        case 6: r = tanhf(v); break;
        case 7: r = 1.0f / (1.0f + expf(-v)); break;
        case 8: r = fmaxf(v, 0.0f); break;
        case 9: r = gelu_f(v); break;
        case 10: r = gelu_grad_f(v); break;
        case 11: r = sinf(v); break;
        default: r = cosf(v); break;
    }
    y[i] = st<T>(r);
}
#define MAKE_UNARY(suf, T) \
    extern "C" __global__ void unary_##suf(const T* x, T* y, u32 n, u32 op) { unary_impl<T>(x, y, n, op); }
FOR_FLOATS(MAKE_UNARY)

struct BinParams {
    u32 n;
    u32 rank;
    u32 op;
    u32 pad;
    u32 shape[MAX_RANK];
    u32 sa[MAX_RANK];
    u32 sb[MAX_RANK];
};

// op codes must match binary_code() in cuda.rs
__device__ __forceinline__ float binop(u32 op, float a, float b) {
    switch (op) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: return a / b;
        case 4: return powf(a, b);
        case 5: return fmaxf(a, b);
        case 6: return fminf(a, b);
        case 7: return a == b ? 1.0f : 0.0f;
        case 8: return a > b ? 1.0f : 0.0f;
        default: return a < b ? 1.0f : 0.0f;
    }
}

template <typename T> __device__ void binary_impl(const T* a, const T* b, T* out, const BinParams& p) {
    u32 i = gid();
    if (i >= p.n) return;
    u32 rem = i, oa = 0, ob = 0;
    for (int d = (int)p.rank - 1; d >= 0; --d) {
        u32 c = rem % p.shape[d];
        rem /= p.shape[d];
        oa += c * p.sa[d];
        ob += c * p.sb[d];
    }
    out[i] = st<T>(binop(p.op, ld(a[oa]), ld(b[ob])));
}
#define MAKE_BINARY(suf, T) \
    extern "C" __global__ void binary_##suf(const T* a, const T* b, T* out, BinParams p) { binary_impl<T>(a, b, out, p); }
FOR_FLOATS(MAKE_BINARY)

struct CopyParams {
    u32 n;
    u32 rank;
    u32 base;
    u32 pad;
    u32 shape[MAX_RANK];
    u32 strides[MAX_RANK];
};

// out[i] = x[base + strided offset of i]  (dtype agnostic: 32- or 16-bit words)
template <typename W> __device__ void strided_copy_impl(const W* x, W* out, const CopyParams& p) {
    u32 i = gid();
    if (i >= p.n) return;
    u32 rem = i, o = p.base;
    for (int d = (int)p.rank - 1; d >= 0; --d) {
        u32 c = rem % p.shape[d];
        rem /= p.shape[d];
        o += c * p.strides[d];
    }
    out[i] = x[o];
}
extern "C" __global__ void strided_copy_u32(const u32* x, u32* out, CopyParams p) { strided_copy_impl<u32>(x, out, p); }
extern "C" __global__ void strided_copy_u16(const u16* x, u16* out, CopyParams p) { strided_copy_impl<u16>(x, out, p); }
extern "C" __global__ void strided_copy_u8(const unsigned char* x, unsigned char* out, CopyParams p) {
    strided_copy_impl<unsigned char>(x, out, p);
}

struct CatParams {
    u32 n;
    u32 blk_in;
    u32 blk_out;
    u32 offset;
};

template <typename W> __device__ void cat_copy_impl(const W* x, W* out, const CatParams& p) {
    u32 i = gid();
    if (i >= p.n) return;
    u32 o = i / p.blk_in, j = i % p.blk_in;
    out[o * p.blk_out + p.offset + j] = x[i];
}
extern "C" __global__ void cat_copy_u32(const u32* x, u32* out, CatParams p) { cat_copy_impl<u32>(x, out, p); }
extern "C" __global__ void cat_copy_u16(const u16* x, u16* out, CatParams p) { cat_copy_impl<u16>(x, out, p); }
extern "C" __global__ void cat_copy_u8(const unsigned char* x, unsigned char* out, CatParams p) {
    cat_copy_impl<unsigned char>(x, out, p);
}

template <typename T> __device__ void where_impl(const T* c, const T* a, const T* b, T* out, u32 n) {
    u32 i = gid();
    if (i < n) out[i] = ld(c[i]) != 0.0f ? a[i] : b[i];
}
#define MAKE_WHERE(suf, T) \
    extern "C" __global__ void where_##suf(const T* c, const T* a, const T* b, T* out, u32 n) { where_impl<T>(c, a, b, out, n); }
FOR_FLOATS(MAKE_WHERE)

// ---------------------------------------------------------------- reductions

struct RedParams {
    u32 outer;
    u32 len;
    u32 inner;
    u32 op; // 0 sum, 1 max, 2 min, 3 argmax
};

__device__ __forceinline__ float red_init(u32 op) {
    return op == 0 ? 0.0f : (op == 2 ? INF : -INF);
}

__device__ __forceinline__ float red_combine(u32 op, float a, float b) {
    return op == 0 ? a + b : (op == 2 ? fminf(a, b) : fmaxf(a, b));
}

__device__ __forceinline__ float warp_reduce(u32 op, float v) {
    for (int o = 16; o > 0; o >>= 1) v = red_combine(op, v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Block-wide reduction; blockDim.x must be a multiple of 32. Result is
// broadcast to every thread. `buf` is 32 floats of shared memory.
__device__ float block_reduce(u32 op, float v, float* buf) {
    u32 lane = threadIdx.x & 31u, wid = threadIdx.x >> 5;
    u32 nw = (blockDim.x + 31u) >> 5;
    v = warp_reduce(op, v);
    if (lane == 0) buf[wid] = v;
    __syncthreads();
    float r = lane < nw ? buf[lane] : red_init(op);
    r = warp_reduce(op, r);
    __syncthreads();
    return r;
}

// One thread per output element (used when inner > 1, and for argmax).
template <typename T> __device__ void reduce_strided_impl(const T* x, T* out, const RedParams& p) {
    u32 t = gid();
    if (t >= p.outer * p.inner) return;
    u32 o = t / p.inner, i = t % p.inner;
    u32 base = o * p.len * p.inner + i;
    if (p.op == 3) {
        float best = -INF;
        u32 bi = 0;
        for (u32 l = 0; l < p.len; ++l) {
            float v = ld(x[base + l * p.inner]);
            if (v > best) { best = v; bi = l; }
        }
        ((int*)out)[t] = (int)bi;
        return;
    }
    float acc = red_init(p.op);
    for (u32 l = 0; l < p.len; ++l) acc = red_combine(p.op, acc, ld(x[base + l * p.inner]));
    out[t] = st<T>(acc);
}
#define MAKE_REDUCE_STRIDED(suf, T) \
    extern "C" __global__ void reduce_strided_##suf(const T* x, T* out, RedParams p) { reduce_strided_impl<T>(x, out, p); }
FOR_FLOATS(MAKE_REDUCE_STRIDED)

// One block per row (inner == 1).
template <typename T> __device__ void reduce_rows_impl(const T* x, T* out, const RedParams& p) {
    __shared__ float buf[32];
    u32 row = blockIdx.x;
    const T* xr = x + (u64)row * p.len;
    float acc = red_init(p.op);
    for (u32 l = threadIdx.x; l < p.len; l += blockDim.x) acc = red_combine(p.op, acc, ld(xr[l]));
    acc = block_reduce(p.op, acc, buf);
    if (threadIdx.x == 0) out[row] = st<T>(acc);
}
#define MAKE_REDUCE_ROWS(suf, T) \
    extern "C" __global__ void reduce_rows_##suf(const T* x, T* out, RedParams p) { reduce_rows_impl<T>(x, out, p); }
FOR_FLOATS(MAKE_REDUCE_ROWS)

// ---------------------------------------------------------------- fused row ops (f32)

extern "C" __global__ void softmax_rows(const float* x, float* y, u32 d, u32 log_mode) {
    __shared__ float buf[32];
    u64 off = (u64)blockIdx.x * d;
    const float* xr = x + off;
    float* yr = y + off;
    float m = -INF;
    for (u32 l = threadIdx.x; l < d; l += blockDim.x) m = fmaxf(m, xr[l]);
    m = block_reduce(1, m, buf);
    float s = 0.0f;
    for (u32 l = threadIdx.x; l < d; l += blockDim.x) s += expf(xr[l] - m);
    s = block_reduce(0, s, buf);
    if (log_mode != 0) {
        float lse = m + logf(s);
        for (u32 l = threadIdx.x; l < d; l += blockDim.x) yr[l] = xr[l] - lse;
    } else {
        float inv = 1.0f / s;
        for (u32 l = threadIdx.x; l < d; l += blockDim.x) yr[l] = expf(xr[l] - m) * inv;
    }
}

extern "C" __global__ void softmax_backward_rows(const float* y, const float* g, float* out, u32 d, u32 log_mode) {
    __shared__ float buf[32];
    u64 off = (u64)blockIdx.x * d;
    float s = 0.0f;
    if (log_mode != 0) {
        for (u32 l = threadIdx.x; l < d; l += blockDim.x) s += g[off + l];
    } else {
        for (u32 l = threadIdx.x; l < d; l += blockDim.x) s += g[off + l] * y[off + l];
    }
    s = block_reduce(0, s, buf);
    for (u32 l = threadIdx.x; l < d; l += blockDim.x) {
        float yv = y[off + l], gv = g[off + l];
        out[off + l] = log_mode != 0 ? gv - expf(yv) * s : yv * (gv - s);
    }
}

extern "C" __global__ void layer_norm_rows(const float* x, float* xhat, float* rstd, u32 d, float eps) {
    __shared__ float buf[32];
    u64 off = (u64)blockIdx.x * d;
    float s = 0.0f;
    for (u32 l = threadIdx.x; l < d; l += blockDim.x) s += x[off + l];
    float mean = block_reduce(0, s, buf) / (float)d;
    float v = 0.0f;
    for (u32 l = threadIdx.x; l < d; l += blockDim.x) {
        float c = x[off + l] - mean;
        v += c * c;
    }
    float r = 1.0f / sqrtf(block_reduce(0, v, buf) / (float)d + eps);
    for (u32 l = threadIdx.x; l < d; l += blockDim.x) xhat[off + l] = (x[off + l] - mean) * r;
    if (threadIdx.x == 0) rstd[blockIdx.x] = r;
}

extern "C" __global__ void layer_norm_backward_rows(const float* g, const float* xhat, const float* rstd, float* out, u32 d) {
    __shared__ float buf[32];
    u64 off = (u64)blockIdx.x * d;
    float a = 0.0f, b = 0.0f;
    for (u32 l = threadIdx.x; l < d; l += blockDim.x) {
        a += g[off + l];
        b += g[off + l] * xhat[off + l];
    }
    float mg = block_reduce(0, a, buf) / (float)d;
    float mgx = block_reduce(0, b, buf) / (float)d;
    float r = rstd[blockIdx.x];
    for (u32 l = threadIdx.x; l < d; l += blockDim.x) out[off + l] = r * (g[off + l] - mg - xhat[off + l] * mgx);
}

struct AdamParams {
    u32 n;
    float lr, beta1, beta2, eps, wd;
    u32 decoupled;
    float ib1, ib2; // 1 / bias corrections
};

extern "C" __global__ void adam_step(const float* p, const float* g, const float* m, const float* v,
                                     float* np, float* nm, float* nv, AdamParams h) {
    u32 i = gid();
    if (i >= h.n) return;
    float gi = g[i];
    if (h.wd != 0.0f && h.decoupled == 0) gi += h.wd * p[i];
    float mi = h.beta1 * m[i] + (1.0f - h.beta1) * gi;
    float vi = h.beta2 * v[i] + (1.0f - h.beta2) * gi * gi;
    float pi = p[i] - h.lr * (mi * h.ib1) / (sqrtf(vi * h.ib2) + h.eps);
    if (h.wd != 0.0f && h.decoupled != 0) pi -= h.lr * h.wd * p[i];
    np[i] = pi;
    nm[i] = mi;
    nv[i] = vi;
}

// ---------------------------------------------------------------- indexing

template <typename T> __device__ void index_select_impl(const T* t, const int* idx, T* out, u32 n, u32 cols) {
    u32 i = gid();
    if (i >= n) return;
    u32 r = i / cols, c = i % cols;
    out[i] = t[(u64)idx[r] * cols + c];
}
#define MAKE_INDEX_SELECT(suf, T)                                                                \
    extern "C" __global__ void index_select_##suf(const T* t, const int* idx, T* out, u32 n, u32 cols) { \
        index_select_impl<T>(t, idx, out, n, cols);                                              \
    }
FOR_FLOATS(MAKE_INDEX_SELECT)

extern "C" __global__ void index_add(const float* src, const int* idx, float* out, u32 n, u32 cols) {
    u32 i = gid();
    if (i >= n) return;
    u32 r = i / cols, c = i % cols;
    atomicAdd(&out[(u64)idx[r] * cols + c], src[i]);
}

struct GatherParams {
    u32 n;
    u32 len;
    u32 idx_len;
    u32 inner;
};

extern "C" __global__ void gather(const float* x, const int* idx, float* out, GatherParams p) {
    u32 t = gid();
    if (t >= p.n) return;
    u32 i = t % p.inner, o = t / (p.idx_len * p.inner);
    out[t] = x[((u64)o * p.len + (u32)idx[t]) * p.inner + i];
}

extern "C" __global__ void scatter_add(const float* src, const int* idx, float* out, GatherParams p) {
    u32 t = gid();
    if (t >= p.n) return;
    u32 i = t % p.inner, o = t / (p.idx_len * p.inner);
    atomicAdd(&out[((u64)o * p.len + (u32)idx[t]) * p.inner + i], src[t]);
}

// ---------------------------------------------------------------- convolution

struct ConvParams {
    u32 n, c, h, w, kh, kw, stride, padding, oh, ow;
};

template <typename T> __device__ void im2col_impl(const T* x, T* cols, const ConvParams& g) {
    u32 t = gid();
    u32 l = g.oh * g.ow;
    u32 total = g.n * g.c * g.kh * g.kw * l;
    if (t >= total) return;
    u32 ox = t % g.ow;
    u32 oy = (t / g.ow) % g.oh;
    u32 r = t / l;
    u32 kj = r % g.kw;
    u32 ki = (r / g.kw) % g.kh;
    u32 c = (r / (g.kw * g.kh)) % g.c;
    u32 n = r / (g.c * g.kh * g.kw);
    int iy = (int)(oy * g.stride + ki) - (int)g.padding;
    int ix = (int)(ox * g.stride + kj) - (int)g.padding;
    T v = st<T>(0.0f);
    if (iy >= 0 && iy < (int)g.h && ix >= 0 && ix < (int)g.w) {
        v = x[((n * g.c + c) * g.h + (u32)iy) * g.w + (u32)ix];
    }
    cols[t] = v;
}
#define MAKE_IM2COL(suf, T) \
    extern "C" __global__ void im2col_##suf(const T* x, T* cols, ConvParams g) { im2col_impl<T>(x, cols, g); }
FOR_FLOATS(MAKE_IM2COL)

// one thread per input pixel: gathers every column entry that read it
template <typename T> __device__ void col2im_impl(const T* cols, T* x, const ConvParams& g) {
    u32 t = gid();
    u32 total = g.n * g.c * g.h * g.w;
    if (t >= total) return;
    u32 px = t % g.w;
    u32 py = (t / g.w) % g.h;
    u32 c = (t / (g.w * g.h)) % g.c;
    u32 n = t / (g.w * g.h * g.c);
    u32 l = g.oh * g.ow;
    float acc = 0.0f;
    for (u32 ki = 0; ki < g.kh; ++ki) {
        int ty = (int)(py + g.padding) - (int)ki;
        if (ty < 0 || ty % (int)g.stride != 0) continue;
        u32 oy = (u32)ty / g.stride;
        if (oy >= g.oh) continue;
        for (u32 kj = 0; kj < g.kw; ++kj) {
            int tx = (int)(px + g.padding) - (int)kj;
            if (tx < 0 || tx % (int)g.stride != 0) continue;
            u32 ox = (u32)tx / g.stride;
            if (ox >= g.ow) continue;
            u32 r = n * (g.c * g.kh * g.kw) + (c * g.kh + ki) * g.kw + kj;
            acc += ld(cols[(u64)r * l + oy * g.ow + ox]);
        }
    }
    x[t] = st<T>(acc);
}
#define MAKE_COL2IM(suf, T) \
    extern "C" __global__ void col2im_##suf(const T* cols, T* x, ConvParams g) { col2im_impl<T>(cols, x, g); }
FOR_FLOATS(MAKE_COL2IM)

// ---------------------------------------------------------------- int8 weights

struct Int8Params {
    u32 m, k, n;
};

// out[i, j] = scale[j] * sum_l x[i, l] * q[l, j]; one thread per output
// (coalesced weight reads across j; suits small m / decoding).
template <typename T> __device__ void matmul_int8_impl(const T* x, const signed char* q, const float* sc, T* out, const Int8Params& p) {
    u32 t = gid();
    if (t >= p.m * p.n) return;
    u32 i = t / p.n, j = t % p.n;
    const T* xr = x + (u64)i * p.k;
    float acc = 0.0f;
    for (u32 l = 0; l < p.k; ++l) acc += ld(xr[l]) * (float)q[(u64)l * p.n + j];
    out[t] = st<T>(acc * sc[j]);
}
#define MAKE_MATMUL_INT8(suf, T)                                                                                \
    extern "C" __global__ void matmul_int8_##suf(const T* x, const signed char* q, const float* sc, T* out, Int8Params p) { \
        matmul_int8_impl<T>(x, q, sc, out, p);                                                                  \
    }
FOR_FLOATS(MAKE_MATMUL_INT8)

// ---------------------------------------------------------------- int4 weights

struct Int4Params {
    u32 m, k, n, group;
};

// Row l, column j of packed int4 weights: byte (l / 2) * n + j, low nibble
// for even l, high for odd l, 4-bit two's complement.
__device__ __forceinline__ float int4_at(const unsigned char* q, u32 l, u32 j, u32 n) {
    unsigned char b = q[(u64)(l >> 1) * n + j];
    int v = (l & 1) ? (int)(b >> 4) : (int)(b & 15);
    return (float)(v >= 8 ? v - 16 : v);
}

// out[i, j] = sum_g scale[g, j] * sum_{l in group g} x[i, l] * w[l, j]; one thread per output.
template <typename T> __device__ void matmul_int4_impl(const T* x, const unsigned char* q, const float* sc, T* out, const Int4Params& p) {
    u32 t = gid();
    if (t >= p.m * p.n) return;
    u32 i = t / p.n, j = t % p.n;
    const T* xr = x + (u64)i * p.k;
    float acc = 0.0f;
    for (u32 g0 = 0; g0 < p.k; g0 += p.group) {
        float part = 0.0f;
        for (u32 l = g0; l < g0 + p.group; ++l) part += ld(xr[l]) * int4_at(q, l, j, p.n);
        acc += part * sc[(u64)(g0 / p.group) * p.n + j];
    }
    out[t] = st<T>(acc);
}
#define MAKE_MATMUL_INT4(suf, T)                                                                                    \
    extern "C" __global__ void matmul_int4_##suf(const T* x, const unsigned char* q, const float* sc, T* out, Int4Params p) { \
        matmul_int4_impl<T>(x, q, sc, out, p);                                                                      \
    }
FOR_FLOATS(MAKE_MATMUL_INT4)

extern "C" __global__ void dequant_int4(const unsigned char* q, const float* sc, float* out, Int4Params p) {
    u32 t = gid();
    if (t >= p.k * p.n) return;
    u32 l = t / p.n, j = t % p.n;
    out[t] = int4_at(q, l, j, p.n) * sc[(u64)(l / p.group) * p.n + j];
}

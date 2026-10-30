// CUDA kernels for the CUDA backend; a one-to-one port of mps_kernels.metal.
// Compiled at runtime with NVRTC (precise math, no fast-math). Parameter
// structs are passed by value and mirror `#[repr(C)]` structs in cuda.rs.

#define MAX_RANK 8
#define INF __int_as_float(0x7f800000)

typedef unsigned int u32;
typedef unsigned long long u64; // NVRTC has no <cstddef>

__device__ __forceinline__ u32 gid() { return blockIdx.x * blockDim.x + threadIdx.x; }

// ---------------------------------------------------------------- memory

extern "C" __global__ void fill_u32(u32* out, u32 n, u32 bits) {
    u32 i = gid();
    if (i < n) out[i] = bits;
}

extern "C" __global__ void cast_f32_i32(const float* x, int* y, u32 n) {
    u32 i = gid();
    if (i < n) y[i] = (int)x[i];
}

extern "C" __global__ void cast_i32_f32(const int* x, float* y, u32 n) {
    u32 i = gid();
    if (i < n) y[i] = (float)x[i];
}

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
extern "C" __global__ void unary_f32(const float* x, float* y, u32 n, u32 op) {
    u32 i = gid();
    if (i >= n) return;
    float v = x[i];
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
    y[i] = r;
}

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

extern "C" __global__ void binary_f32(const float* a, const float* b, float* out, BinParams p) {
    u32 i = gid();
    if (i >= p.n) return;
    u32 rem = i, oa = 0, ob = 0;
    for (int d = (int)p.rank - 1; d >= 0; --d) {
        u32 c = rem % p.shape[d];
        rem /= p.shape[d];
        oa += c * p.sa[d];
        ob += c * p.sb[d];
    }
    out[i] = binop(p.op, a[oa], b[ob]);
}

struct CopyParams {
    u32 n;
    u32 rank;
    u32 base;
    u32 pad;
    u32 shape[MAX_RANK];
    u32 strides[MAX_RANK];
};

// out[i] = x[base + strided offset of i]  (dtype agnostic: copies 32-bit words)
extern "C" __global__ void strided_copy(const u32* x, u32* out, CopyParams p) {
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

struct CatParams {
    u32 n;
    u32 blk_in;
    u32 blk_out;
    u32 offset;
};

extern "C" __global__ void cat_copy(const u32* x, u32* out, CatParams p) {
    u32 i = gid();
    if (i >= p.n) return;
    u32 o = i / p.blk_in, j = i % p.blk_in;
    out[o * p.blk_out + p.offset + j] = x[i];
}

extern "C" __global__ void where_f32(const float* c, const float* a, const float* b, float* out, u32 n) {
    u32 i = gid();
    if (i < n) out[i] = c[i] != 0.0f ? a[i] : b[i];
}

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
extern "C" __global__ void reduce_strided(const float* x, float* out, RedParams p) {
    u32 t = gid();
    if (t >= p.outer * p.inner) return;
    u32 o = t / p.inner, i = t % p.inner;
    u32 base = o * p.len * p.inner + i;
    if (p.op == 3) {
        float best = -INF;
        u32 bi = 0;
        for (u32 l = 0; l < p.len; ++l) {
            float v = x[base + l * p.inner];
            if (v > best) { best = v; bi = l; }
        }
        ((int*)out)[t] = (int)bi;
        return;
    }
    float acc = red_init(p.op);
    for (u32 l = 0; l < p.len; ++l) acc = red_combine(p.op, acc, x[base + l * p.inner]);
    out[t] = acc;
}

// One block per row (inner == 1).
extern "C" __global__ void reduce_rows(const float* x, float* out, RedParams p) {
    __shared__ float buf[32];
    u32 row = blockIdx.x;
    const float* xr = x + (u64)row * p.len;
    float acc = red_init(p.op);
    for (u32 l = threadIdx.x; l < p.len; l += blockDim.x) acc = red_combine(p.op, acc, xr[l]);
    acc = block_reduce(p.op, acc, buf);
    if (threadIdx.x == 0) out[row] = acc;
}

// ---------------------------------------------------------------- fused row ops

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

extern "C" __global__ void index_select(const float* t, const int* idx, float* out, u32 n, u32 cols) {
    u32 i = gid();
    if (i >= n) return;
    u32 r = i / cols, c = i % cols;
    out[i] = t[(u64)idx[r] * cols + c];
}

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

extern "C" __global__ void im2col(const float* x, float* cols, ConvParams g) {
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
    float v = 0.0f;
    if (iy >= 0 && iy < (int)g.h && ix >= 0 && ix < (int)g.w) {
        v = x[((n * g.c + c) * g.h + (u32)iy) * g.w + (u32)ix];
    }
    cols[t] = v;
}

extern "C" __global__ void col2im(const float* cols, float* x, ConvParams g) {
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
            acc += cols[(u64)r * l + oy * g.ow + ox];
        }
    }
    x[t] = acc;
}

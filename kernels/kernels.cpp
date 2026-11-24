// faster-chat-llm GPU kernels (HIP, gfx1201 / RDNA4, wave32).
//
// Every launch is asynchronous on the stream it is given. The Rust side owns all
// scheduling, memory planning and synchronisation; this file is math only.
//
// Conventions
//   B        rows in the current batch (a row = one token position of one sequence)
//   D        hidden size (5120)
//   Q8 mat   weights as stored in the GGUF, de-interleaved: qs[rows][cols] int8 and the f16
//            block scales grouped by 16-row tile, d[rows/16][cols/32][16], so a matrix-core
//            tile reads its scales with one load. Same values as block_q8_0, no requantisation.
//   q8 act   activations quantised per 32-element block exactly like llama.cpp's
//            quantize_q8_1: d = amax/127, q = roundf(x/d), d stored rounded to f16.
//            Stored block-major, xq[blk][row][32] and xd[blk][row] with a row stride `xs` chosen
//            per step: the 16 rows a matrix-core tile needs for one block are contiguous.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <string.h>

// gfx12 compiles wave32 by default; the launchers check warpSize once at init (pk_wave_size).

#define PK_BMAX 16
#define PK_MAX_GROUPS 4

// A group is a run of consecutive batch rows that belong to one sequence.
struct PkGroups {
    int32_t n;
    int32_t start[PK_MAX_GROUPS];    // first batch row of the group
    int32_t len[PK_MAX_GROUPS];      // rows in the group
    int32_t slot[PK_MAX_GROUPS];     // sequence slot (selects state / history buffers)
    int32_t cur[PK_MAX_GROUPS];      // ping-pong half written this step (0/1); the other is "prev"
    int32_t hist_off[PK_MAX_GROUPS]; // conv history = prev mix rows [hist_off, hist_off+3)
    int32_t nacc[PK_MAX_GROUPS];     // GDN rows of prev step to replay into the state first
    int32_t commit[PK_MAX_GROUPS];   // 1: state written back includes this step's rows
    int32_t pos0[PK_MAX_GROUPS];     // position of the group's first row
    int32_t kv_dir[PK_MAX_GROUPS];   // +1 / -1: KV slot = kv_base + kv_dir * position
    int64_t kv_base[PK_MAX_GROUPS];
};

// ---------------------------------------------------------------- helpers

static __device__ __forceinline__ float wave_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}
static __device__ __forceinline__ float wave_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
    return v;
}
static __device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }
static __device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
static __device__ __forceinline__ float softplus_f(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

// Quantise element e of batch row b. Must be called by all 32 lanes of a wave, lane i holding
// element (e & ~31) + i of the same row.
static __device__ __forceinline__ void quant_block(float x, int8_t * xq, float * xd, int xs, int64_t b, int e) {
    const float amax = wave_max(fabsf(x));
    const float d = amax / 127.0f;
    const int64_t cell = (int64_t) (e >> 5) * xs + b;
    xq[cell * 32 + (e & 31)] = amax == 0.0f ? (int8_t) 0 : (int8_t) roundf(x / d);
    if ((e & 31) == 0) xd[cell] = (float) (_Float16) d;
}

// Sum `v` over the whole workgroup (nwaves waves). `red` is shared scratch of >= nwaves floats.
static __device__ __forceinline__ float block_sum(float v, float * red, int nwaves) {
    const float w = wave_sum(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = w;
    __syncthreads();
    float s = 0.0f;
    for (int i = 0; i < nwaves; i++) s += red[i];
    __syncthreads();
    return s;
}
static __device__ __forceinline__ float block_max(float v, float * red, int nwaves) {
    const float w = wave_max(v);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = w;
    __syncthreads();
    float s = red[0];
    for (int i = 1; i < nwaves; i++) s = fmaxf(s, red[i]);
    __syncthreads();
    return s;
}

// ---------------------------------------------------------------- load-time

struct PkColRanges { int32_t n; int32_t start[4]; int32_t len[4]; }; // in 32-element blocks

// raw: n_rows rows of block_q8_0 {f16 d; int8 qs[32]} with nb_src blocks each.
// Copies the selected column blocks into the SoA matrix at row offset dst_row0.
__global__ void k_unpack_q8(const uint8_t * __restrict__ raw, int8_t * __restrict__ qs, uint16_t * __restrict__ d,
                            int64_t n_rows, int nb_src, int nb_dst, int64_t dst_row0, PkColRanges cr) {
    const int64_t row = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    const uint8_t * src_row = raw + row * (int64_t) nb_src * 34;
    const int64_t dr = dst_row0 + row;
    int8_t *   q_row = qs + dr * (int64_t) nb_dst * 32;
    uint16_t * d_row = d + (dr / 16) * (int64_t) nb_dst * 16 + (dr % 16); // [tile][blk][16 rows]
    int ob = 0;
    for (int r = 0; r < cr.n; r++) {
        for (int b = 0; b < cr.len[r]; b++, ob++) {
            const uint8_t * s = src_row + (int64_t) (cr.start[r] + b) * 34;
            d_row[(int64_t) ob * 16] = (uint16_t) s[0] | ((uint16_t) s[1] << 8);
            for (int i = 0; i < 32; i++) q_row[ob * 32 + i] = (int8_t) s[2 + i];
        }
    }
}

// ---------------------------------------------------------------- embedding

// h[b][:] = dequantised row tokens[b] of a Q8 matrix.
__global__ void k_embed(const int8_t * __restrict__ qs, const _Float16 * __restrict__ d,
                        const int32_t * __restrict__ tokens, float * __restrict__ h, int D) {
    const int b = blockIdx.x;
    const int64_t tok = tokens[b];
    for (int e = threadIdx.x; e < D; e += blockDim.x) {
        h[(int64_t) b * D + e] = (float) qs[tok * D + e] * (float) d[((tok / 16) * (D / 32) + e / 32) * 16 + tok % 16];
    }
}

// ---------------------------------------------------------------- residual add + RMS norm + quantise

// grid (B), block 256. h += p0 + p1 (each optional), then x = rmsnorm(h) * w.
// Writes q8 activations starting at element e_off of each row and, if xf != null, the f32
// normalised row.
__global__ void k_norm_quant(float * __restrict__ h, const float * __restrict__ p0, const float * __restrict__ p1,
                             const float * __restrict__ w, float eps, int D,
                             int8_t * __restrict__ xq, float * __restrict__ xd, int xs, int e_off,
                             float * __restrict__ xf) {
    __shared__ float red[8];
    const int b = blockIdx.x, t = threadIdx.x;
    float * hr = h + (int64_t) b * D;
    float ss = 0.0f;
    for (int e = t; e < D; e += 256) {
        float v = hr[e];
        if (p0) v += p0[(int64_t) b * D + e];
        if (p1) v += p1[(int64_t) b * D + e];
        if (p0) hr[e] = v;
        ss += v * v;
    }
    ss = block_sum(ss, red, 8);
    const float scale = 1.0f / sqrtf(ss / (float) D + eps);
    for (int e = t; e < D; e += 256) {
        const float v = hr[e] * scale * w[e];
        if (xf) xf[(int64_t) b * D + e] = v;
        if (xq) quant_block(v, xq, xd, xs, b, e_off + e);
    }
}

// ---------------------------------------------------------------- Q8_0 x q8 GEMM, up to 16 batch rows

typedef int      v2i __attribute__((ext_vector_type(2)));
typedef int      v8i __attribute__((ext_vector_type(8)));
typedef float    v8f __attribute__((ext_vector_type(8)));
typedef _Float16 v8h __attribute__((ext_vector_type(8)));

// y[b][r] = sum_blk d_w[r][blk] * d_x[b][blk] * dot_i8(W[r][blk], x[b][blk]).
// A workgroup of NW waves owns a 16-row weight tile and J*16 batch rows. Each wave takes every
// NW-th 32-column block: it loads the weights once and runs them against J activation tiles
// through the int8 matrix core (V_WMMA_I32_16X16X16_IU8, twice per block).
// grid (ceil(N / (16 J)), rows / 16), block 32 * NW.
// Matrix-core register layout on gfx12, wave32:
//   A[m][k]: lane = m + 16*(k/8), byte k%8     B[k][n]: lane = n + 16*(k/8), byte k%8
//   C[m][n]: lane = n + 16*(m/8), element m%8
// A batch row's result depends only on that row's activations, never on the batch size or the
// row's place in it: a token computes the same bits alone, in a verify batch, or beside
// another sequence. That is what makes speculative decode exactly lossless.
// Waves per weight tile at decode batch sizes. Swept 2..32 on gfx1201: more waves in flight
// stream the weights closer to memory bandwidth (32: 29.3 ms per token step, 4: 33.2 ms).
#ifndef PK_GEMM_WAVES
#define PK_GEMM_WAVES 32
#endif
template <int J, int NW>
__global__ void k_gemm(const int8_t * __restrict__ W, const _Float16 * __restrict__ Wd,
                       const int8_t * __restrict__ xq, const float * __restrict__ xd,
                       int xs, int b_off, float * __restrict__ y, int cols, int N, int y_stride) {
    // Partial sums of the waves; a single-wave workgroup needs none, and must not reserve any:
    // shared memory per workgroup caps how many waves a compute unit can keep in flight.
    __shared__ float red[NW > 1 ? NW : 1][NW > 1 ? J : 1][NW > 1 ? 32 : 1][8];
    const int64_t tile = blockIdx.y;
    const int64_t b0 = (int64_t) blockIdx.x * 16 * J;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, lo = lane & 15, hi = lane >> 4;
    const int nb = cols / 32;
    const int8_t *   wrow = W + (tile * 16 + lo) * cols + hi * 8;
    const _Float16 * dt = Wd + tile * nb * 16 + hi * 8;
    const int64_t xb = b_off + b0 + lo; // batch row of this lane in activation tile 0
    v8f acc[J];
    for (int j = 0; j < J; j++) acc[j] = (v8f) {0, 0, 0, 0, 0, 0, 0, 0};
    for (int blk = wave; blk < nb; blk += NW) {
        // Issue every load of the block before the first use, so they overlap instead of
        // each being waited for in turn.
        v2i a0, a1, x0[J], x1[J];
        float dx[J];
        v8h dh;
        memcpy(&a0, wrow + blk * 32, 8);
        memcpy(&a1, wrow + blk * 32 + 16, 8);
        memcpy(&dh, dt + blk * 16, 16);
#pragma unroll
        for (int j = 0; j < J; j++) {
            const int64_t cell = (int64_t) blk * xs + xb + j * 16;
            memcpy(&x0[j], xq + cell * 32 + hi * 8, 8);
            memcpy(&x1[j], xq + cell * 32 + hi * 8 + 16, 8);
            dx[j] = xd[cell];
        }
        const v8f dw = __builtin_convertvector(dh, v8f);
#pragma unroll
        for (int j = 0; j < J; j++) {
            v8i c = {0, 0, 0, 0, 0, 0, 0, 0};
            c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a0, true, x0[j], c, false);
            c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a1, true, x1[j], c, false);
            acc[j] += __builtin_convertvector(c, v8f) * (dw * dx[j]);
        }
    }
    if (NW > 1) {
        for (int j = 0; j < J; j++)
            for (int e = 0; e < 8; e++) red[wave][j][lane][e] = acc[j][e];
        __syncthreads();
        if (wave != 0) return;
        for (int j = 0; j < J; j++)
            for (int e = 0; e < 8; e++)
                for (int w = 1; w < NW; w++) acc[j][e] += red[w][j][lane][e];
    }
    for (int j = 0; j < J; j++) {
        const int64_t b = b0 + j * 16 + lo;
        if (b >= N) continue;
        for (int e = 0; e < 8; e++) y[b * y_stride + tile * 16 + hi * 8 + e] = acc[j][e];
    }
}

// ---------------------------------------------------------------- FFN activation

// a = silu(gate) * up, quantised. y holds [up (F) | gate (F)] per row. grid (F/256, B), block 256.
__global__ void k_ffn_act(const float * __restrict__ y, int F, int8_t * __restrict__ xq, float * __restrict__ xd, int xs) {
    const int b = blockIdx.y, t = threadIdx.x;
    const int e = blockIdx.x * 256 + t;
    const float * yr = y + (int64_t) b * 2 * F;
    const float a = silu_f(yr[F + e]) * yr[e];
    quant_block(a, xq, xd, xs, b, e);
}

// ---------------------------------------------------------------- Gated DeltaNet

// Layout of one GDN input row (output of the fused "in" GEMV), local to a rank:
//   [ q (Hk*128) | k (Hk*128) | v (Hv*128) | z (Hv*128) | beta (Hv) | alpha (Hv) ]
// CH = 2*Hk*128 + Hv*128 conv channels. v head lv uses k/q head lv % Hk.
//
// Per sequence slot, two ping-pong halves each of:
//   mix  [(3 + R)][CH]            rows 0..2 = conv history, 3.. = this step's raw qkv
//   qkv  [R][CH + 2*Hv]           conv+silu(+l2norm for q,k) output, then g[Hv], beta[Hv]
// A large committed batch (prefill) has no use for per-row history, so it writes "this step"
// into scratch instead (mix_s, qkv_s non-null) and the caller keeps only the last 3 raw rows:
//   mix_s [3 per group + B][CH]   group g starts at row start[g] + 3*g
//   qkv_s [B][CH + 2*Hv]          indexed by batch row

// grid (CH/128 + 1, B), block 128.
__global__ void k_gdn_conv(const float * __restrict__ yin, int NIN,
                           float * __restrict__ mix, float * __restrict__ qkv, int R,
                           const float * __restrict__ conv_w, const float * __restrict__ dt_bias,
                           const float * __restrict__ ssm_a, int Hk, int Hv, float eps,
                           float * __restrict__ mix_s, float * __restrict__ qkv_s, PkGroups G) {
    __shared__ float red[4];
    const int b = blockIdx.y, t = threadIdx.x, chunk = blockIdx.x;
    const int CH = (2 * Hk + Hv) * 128;
    int g = 0;
    for (int i = 1; i < G.n; i++) if (b >= G.start[i]) g = i;
    const int i = b - G.start[g];
    const int64_t mix_half = (int64_t) (3 + R) * CH;
    const int64_t qkv_row = CH + 2 * Hv;
    const float * mix_prev = mix + ((int64_t) G.slot[g] * 2 + (1 - G.cur[g])) * mix_half;
    float *       mix_cur  = mix_s ? mix_s + ((int64_t) G.start[g] + 3 * g) * CH
                                   : mix + ((int64_t) G.slot[g] * 2 + G.cur[g]) * mix_half;
    float *       out = qkv_s ? qkv_s + (int64_t) b * qkv_row
                              : qkv + (((int64_t) G.slot[g] * 2 + G.cur[g]) * R + i) * qkv_row;
    const float * yr = yin + (int64_t) b * NIN;

    if (chunk == CH / 128) {
        // gate terms: beta = sigmoid(beta_raw), g = softplus(alpha + dt_bias) * a
        if (t < Hv) {
            const float beta = sigmoid_f(yr[CH + Hv * 128 + t]);
            const float gg = softplus_f(yr[CH + Hv * 128 + Hv + t] + dt_bias[t]) * ssm_a[t];
            out[CH + t] = gg;
            out[CH + Hv + t] = beta;
        }
        return;
    }
    const int c = chunk * 128 + t;
    // x[j] = input at time (i - 3 + j); rows before this step come from history.
    float x[4];
    for (int j = 0; j < 4; j++) {
        const int tt = i - 3 + j; // index relative to this step's first row
        if (tt >= 0) {
            x[j] = yin[(int64_t) (G.start[g] + tt) * NIN + c];
        } else {
            x[j] = mix_prev[(int64_t) (G.hist_off[g] + 3 + tt) * CH + c];
        }
    }
    // persist: history rows (only once, by row 0) and this row's raw input
    if (i == 0) {
        for (int j = 0; j < 3; j++) mix_cur[(int64_t) j * CH + c] = x[j];
    }
    mix_cur[(int64_t) (3 + i) * CH + c] = x[3];

    float s = 0.0f;
    for (int j = 0; j < 4; j++) s += x[j] * conv_w[(int64_t) c * 4 + j];
    float v = silu_f(s);
    if (chunk < 2 * Hk) {
        // l2 norm over the head: rms_norm(x, eps/128) / sqrt(128)
        const float ss = block_sum(v * v, red, 4);
        v = v * (1.0f / sqrtf(ss / 128.0f + eps / 128.0f)) * (1.0f / sqrtf(128.0f));
    }
    out[c] = v;
}

// One workgroup per (v head, group). grid (Hv, G.n), block 256: two threads per state column,
// each holding 64 of its 128 elements in registers for the whole token loop (the loops over
// them are fully unrolled, so the state never touches memory between tokens; a whole column
// per thread does not fit the register file). The pair sits in adjacent lanes and combines
// with one shuffle. k and q of the current token are staged in shared memory.
// state[slot][Hv][col][i] f32.
__global__ void k_gdn_step(float * __restrict__ state, const float * __restrict__ qkv, int R,
                           const float * __restrict__ yin, int NIN,
                           const float * __restrict__ norm_w, int Hk, int Hv, float eps,
                           int8_t * __restrict__ xq, float * __restrict__ xd, int xs,
                           const float * __restrict__ qkv_s, PkGroups G) {
    __shared__ float red[4];
    __shared__ float sk[128], sq[128], so[128];
    const int lv = blockIdx.x, g = blockIdx.y, tid = threadIdx.x;
    const int col = tid >> 1, half = tid & 1;
    const int lk = lv % Hk;
    const int CH = (2 * Hk + Hv) * 128;
    const int64_t qkv_row = CH + 2 * Hv;
    const float * rows_prev = qkv + ((int64_t) G.slot[g] * 2 + (1 - G.cur[g])) * R * qkv_row;
    const float * rows_cur  = qkv_s ? qkv_s + (int64_t) G.start[g] * qkv_row
                                    : qkv + ((int64_t) G.slot[g] * 2 + G.cur[g]) * R * qkv_row;
    float * st = state + (((int64_t) G.slot[g] * Hv + lv) * 128 + col) * 128 + half * 64;
    const float * skh = sk + half * 64;
    const float * sqh = sq + half * 64;
    const float scale = 1.0f / sqrtf(128.0f);
    const int n_rep = G.nacc[g], n_new = G.len[g];
    const bool commit = G.commit[g] != 0;

    float S[64];
#pragma unroll
    for (int i = 0; i < 64; i++) S[i] = st[i];

    for (int step = 0; step < n_rep + n_new; step++) {
        const bool is_new = step >= n_rep;
        const float * row = is_new ? rows_cur + (int64_t) (step - n_rep) * qkv_row : rows_prev + (int64_t) step * qkv_row;
        if (tid < 128) {
            sq[tid] = row[lk * 128 + tid];
            sk[tid] = row[(Hk + lk) * 128 + tid];
        }
        const float v = row[(2 * Hk + lv) * 128 + col];
        const float gval = expf(row[CH + lv]);
        const float beta = row[CH + Hv + lv];
        __syncthreads();

        // kv = S . k over the column: four interleaved partial sums per thread, then the pair
        float k4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int i = 0; i < 64; i++) k4[i & 3] += S[i] * skh[i];
        float kv = (k4[0] + k4[1]) + (k4[2] + k4[3]);
        kv += __shfl_xor(kv, 1, 32);
        const float delta = (v - gval * kv) * beta;
        float a4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int i = 0; i < 64; i++) {
            S[i] = gval * S[i] + skh[i] * delta;
            a4[i & 3] += S[i] * sqh[i];
        }
        if (!commit && step == n_rep - 1) {
#pragma unroll
            for (int i = 0; i < 64; i++) st[i] = S[i];
        }
        if (is_new) {
            float attn = (a4[0] + a4[1]) + (a4[2] + a4[3]);
            attn += __shfl_xor(attn, 1, 32);
            if (half == 0) so[col] = attn * scale;
            __syncthreads();
            // output: threads 0..127 take one column each, so a wave is one 32-element block
            const int b = G.start[g] + (step - n_rep);
            float o = 0.0f;
            if (tid < 128) {
                o = so[tid];
                const float w = wave_sum(o * o);
                if ((tid & 31) == 0) red[tid >> 5] = w;
            }
            __syncthreads();
            if (tid < 128) {
                const float ss = (red[0] + red[1]) + (red[2] + red[3]);
                const float z = yin[(int64_t) b * NIN + CH + lv * 128 + tid];
                const float y = o * (1.0f / sqrtf(ss / 128.0f + eps)) * norm_w[tid] * silu_f(z);
                quant_block(y, xq, xd, xs, b, lv * 128 + tid);
            }
        }
        __syncthreads(); // everyone is done with the staged token before the next overwrites it
    }
    if (commit) {
#pragma unroll
        for (int i = 0; i < 64; i++) st[i] = S[i];
    }
}

// ---------------------------------------------------------------- gated full attention

// Fused "in" GEMV row layout, local to a rank:
//   [ per q head: q (256) | gate (256) ] x Hq, then k (Hkv*256), then v (Hkv*256)
// KV cache, f16, cap a multiple of 16 slots:
//   K [slot][Hkv][256]
//   V [slot/16][Hkv][256][slot%16]   (16 slots of one dim are contiguous: a matrix-core operand)

// grid (Hq + 2*Hkv - u0, B), block 256. q -> qr[B][Hq][256] f16 (normed + RoPE); k, v -> cache.
// u0 = Hq with kv_off = 0 runs the K/V units only, on a yin that holds just [k | v].
__global__ void k_attn_prep(const float * __restrict__ yin, int NIN, int Hq, int Hkv, int u0, int kv_off,
                            const float * __restrict__ q_norm_w, const float * __restrict__ k_norm_w, float eps,
                            float theta_scale, int n_rot,
                            _Float16 * __restrict__ qr, _Float16 * __restrict__ Kc, _Float16 * __restrict__ Vc,
                            PkGroups G) {
    __shared__ float red[8];
    __shared__ float sx[256];
    const int u = blockIdx.x + u0, b = blockIdx.y, t = threadIdx.x;
    int g = 0;
    for (int i = 1; i < G.n; i++) if (b >= G.start[i]) g = i;
    const int pos = G.pos0[g] + (b - G.start[g]);
    const int64_t slot = G.kv_base[g] + (int64_t) G.kv_dir[g] * pos;
    const float * yr = yin + (int64_t) b * NIN;

    if (u >= Hq + Hkv) { // V: store as f16
        const int hv = u - Hq - Hkv;
        Vc[(((slot >> 4) * Hkv + hv) * 256 + t) * 16 + (slot & 15)] = (_Float16) yr[kv_off + Hkv * 256 + hv * 256 + t];
        return;
    }
    const bool is_q = u < Hq;
    const float x = is_q ? yr[u * 512 + t] : yr[kv_off + (u - Hq) * 256 + t];
    const float ss = block_sum(x * x, red, 8);
    const float xn = x * (1.0f / sqrtf(ss / 256.0f + eps)) * (is_q ? q_norm_w[t] : k_norm_w[t]);
    sx[t] = xn;
    __syncthreads();
    float o = xn;
    if (t < n_rot) {
        const int half = n_rot / 2;
        const int pair = t < half ? t : t - half;
        const float theta = (float) pos * powf(theta_scale, (float) pair);
        const float c = cosf(theta), s = sinf(theta);
        const float x0 = sx[pair], x1 = sx[pair + half];
        o = t < half ? x0 * c - x1 * s : x0 * s + x1 * c;
    }
    if (is_q) qr[((int64_t) b * Hq + u) * 256 + t] = (_Float16) o;
    else      Kc[(slot * Hkv + (u - Hq)) * 256 + t] = (_Float16) o;
}

// Attention runs on the f16 matrix core, 16 query rows x 16 cache slots at a time, and is cut
// into fixed segments of the cache so long contexts spread over many waves:
//   k_attn_seg    one wave per (query head, row tile, segment): online softmax over its slots,
//                 leaves a partial {O[16][256], m[16], l[16]} (unnormalised output, max, sum).
//   k_attn_merge  folds a row's partials, applies the output gate, quantises.
// Segment boundaries are absolute slot numbers (seg_tiles 16-slot tiles each), and a slot a
// row may not see contributes an exact zero, so with a fixed seg_tiles a row's result does not
// depend on which other rows share its batch. Decode uses a fixed segment size; a prefill
// batch already fills the card with row tiles and runs each as one segment.
#define PK_ATTN_PART (16 * 256 + 32)        // floats per partial
#define PK_NEG_BIG (-1.0e30f)

struct PkTile { int g, i0, nr, seg_lo, seg_hi, st_lo, st_hi; };

// Row tile `ty` of the batch: tiles are per group, 16 consecutive rows each.
static __device__ __forceinline__ PkTile attn_tile(const PkGroups & G, int ty, int seg_tiles) {
    PkTile T;
    int g = 0, first = 0;
    while (g < G.n - 1 && ty >= first + (G.len[g] + 15) / 16) { first += (G.len[g] + 15) / 16; g++; }
    T.g = g;
    T.i0 = (ty - first) * 16;
    T.nr = min(16, G.len[g] - T.i0);
    const int64_t p_max = (int64_t) G.pos0[g] + T.i0 + T.nr - 1;
    const int64_t s_lo = G.kv_dir[g] > 0 ? 0 : G.kv_base[g] - p_max;
    const int64_t s_hi = G.kv_dir[g] > 0 ? p_max : G.kv_base[g];
    T.st_lo = (int) (s_lo >> 4);
    T.st_hi = (int) (s_hi >> 4);
    T.seg_lo = T.st_lo / seg_tiles;
    T.seg_hi = T.st_hi / seg_tiles;
    return T;
}

// grid (Hq, row tiles, max segments), block 32.
// Scores are computed transposed, C[slot][row] = K . q, because that register layout is exactly
// the A operand the second product needs: O[row][dim] += P[row][slot] * V[slot][dim].
__global__ void k_attn_seg(const _Float16 * __restrict__ qr, int Hq, int Hkv,
                           const _Float16 * __restrict__ Kc, const _Float16 * __restrict__ Vc, int seg_tiles,
                           float * __restrict__ part, int S, int seg_abs, PkGroups G) {
    const int hq = blockIdx.x, ty = blockIdx.y;
    const PkTile T = attn_tile(G, ty, seg_tiles);
    // Two ways to cover the cache:
    //   seg_abs < 0   every segment in this launch (grid z), each leaving its own partial;
    //   seg_abs >= 0  this launch does that one segment for every row tile and continues the
    //                 tile's single partial in place. Launched segment by segment, all waves
    //                 walk the same stretch of the cache together, so it is read from memory
    //                 once and served from the card's cache to everyone else.
    const bool in_place = seg_abs >= 0;
    const int seg = in_place ? seg_abs : T.seg_lo + blockIdx.z;
    if (seg < T.seg_lo || seg > T.seg_hi) return;
    const int lane = threadIdx.x, lo = lane & 15, hi = lane >> 4;
    const int g = T.g, hkv = hq / (Hq / Hkv);
    const int dir = G.kv_dir[g];
    const int64_t base = G.kv_base[g];
    // lane lo is query row lo of the tile; rows past the tile's end see nothing
    const int64_t p_first = (int64_t) G.pos0[g] + T.i0;
    const int64_t p_row = lo < T.nr ? p_first + lo : -1;
    const _Float16 * q = qr + (((int64_t) G.start[g] + T.i0 + lo) * Hq + hq) * 256 + hi * 8;
    const float kq_scale = 1.0f / sqrtf(256.0f);

    float * out = part + (((int64_t) ty * Hq + hq) * S + (in_place ? 0 : blockIdx.z)) * PK_ATTN_PART;
    v8f O[16];
    float m = PK_NEG_BIG, l = 0.0f;
    if (in_place && seg > T.seg_lo) {
        for (int t = 0; t < 16; t++)
            for (int e = 0; e < 8; e++) O[t][e] = out[(hi * 8 + e) * 256 + t * 16 + lo];
        m = out[16 * 256 + lo];
        l = out[16 * 256 + 16 + lo];
    } else {
        for (int t = 0; t < 16; t++) O[t] = (v8f) {0, 0, 0, 0, 0, 0, 0, 0};
    }

    const int st_first = max(T.st_lo, seg * seg_tiles);
    const int st_last = (int) min((int64_t) T.st_hi, (int64_t) seg * seg_tiles + seg_tiles - 1);
    for (int st = st_first; st <= st_last; st++) {
        // scores: lane lo = slot (A) / row (B, C)
        const _Float16 * k = Kc + (((int64_t) st * 16 + lo) * Hkv + hkv) * 256 + hi * 8;
        // Loads go out four operand pairs at a time so they overlap; all sixteen at once would
        // not fit the register file next to O.
        v8f c = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int c4 = 0; c4 < 16; c4 += 4) {
            v8h ka[4], qb[4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                memcpy(&ka[i], k + (c4 + i) * 16, 16);
                memcpy(&qb[i], q + (c4 + i) * 16, 16);
            }
#pragma unroll
            for (int i = 0; i < 4; i++) c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ka[i], qb[i], c);
        }
        // element e of this lane is (slot st*16 + hi*8 + e, row lo).
        // Most tiles lie wholly before the tile's first row and need no masking at all; the
        // masked path computes the same values wherever a slot is visible.
        const int64_t tile_last_pos = dir > 0 ? (int64_t) st * 16 + 15 : base - (int64_t) st * 16;
        const bool all_visible = tile_last_pos <= p_first;
        float mx = PK_NEG_BIG;
        bool ok[8];
        for (int e = 0; e < 8; e++) {
            if (all_visible) {
                ok[e] = true;
                c[e] = c[e] * kq_scale;
            } else {
                const int64_t slot = (int64_t) st * 16 + hi * 8 + e;
                const int64_t pos = dir > 0 ? slot : base - slot;
                ok[e] = pos >= 0 && pos <= p_row;
                c[e] = ok[e] ? c[e] * kq_scale : PK_NEG_BIG;
            }
            mx = fmaxf(mx, c[e]);
        }
        mx = fmaxf(mx, __shfl_xor(mx, 16, 32));
        const float m_new = fmaxf(m, mx);
        const float alpha = expf(m - m_new);
        v8f p;
        float psum = 0.0f;
        for (int e = 0; e < 8; e++) {
            p[e] = ok[e] ? expf(c[e] - m_new) : 0.0f;
            psum += p[e];
        }
        psum += __shfl_xor(psum, 16, 32);
        l = l * alpha + psum;
        m = m_new;
        // The running max settles quickly; while no row's max moved, O needs no rescale
        // (alpha is exactly 1). O is laid out [dim lane][row element], so otherwise gather
        // each row's factor.
        if (wave_max(alpha != 1.0f ? 1.0f : 0.0f) != 0.0f) {
            v8f av;
            for (int e = 0; e < 8; e++) av[e] = __shfl(alpha, hi * 8 + e, 32);
#pragma unroll
            for (int t = 0; t < 16; t++) O[t] *= av;
        }
        const v8h pa = __builtin_convertvector(p, v8h);
        const _Float16 * v = Vc + ((((int64_t) st * Hkv + hkv) * 256 + lo) * 16) + hi * 8;
#pragma unroll
        for (int t4 = 0; t4 < 16; t4 += 4) {
            v8h vb[4];
#pragma unroll
            for (int i = 0; i < 4; i++) memcpy(&vb[i], v + (t4 + i) * 16 * 16, 16);
#pragma unroll
            for (int i = 0; i < 4; i++) O[t4 + i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(pa, vb[i], O[t4 + i]);
        }
    }
    for (int t = 0; t < 16; t++)
        for (int e = 0; e < 8; e++) out[(hi * 8 + e) * 256 + t * 16 + lo] = O[t][e];
    if (hi == 0) {
        out[16 * 256 + lo] = m;
        out[16 * 256 + 16 + lo] = l;
    }
}

// grid (Hq, B), block 256 (thread = output dim). out = sum_seg O / sum_seg l, times
// sigmoid(gate), quantised.
__global__ void k_attn_merge(const float * __restrict__ part, int S, const float * __restrict__ yin, int NIN,
                             int Hq, int seg_tiles, int8_t * __restrict__ xq, float * __restrict__ xd, int xs,
                             PkGroups G) {
    const int hq = blockIdx.x, b = blockIdx.y, t = threadIdx.x;
    int g = 0, ty = 0;
    for (int i = 1; i < G.n; i++) {
        if (b >= G.start[i]) { ty += (G.len[i - 1] + 15) / 16; g = i; }
    }
    const int i = b - G.start[g];
    ty += i / 16;
    const int r = i & 15;
    const PkTile T = attn_tile(G, ty, seg_tiles);
    const float * p0 = part + ((int64_t) ty * Hq + hq) * S * PK_ATTN_PART;
    // Partials are cold memory: read them eight segments at a time so the loads overlap.
    const int nz = T.seg_hi - T.seg_lo + 1;
    float M = PK_NEG_BIG;
    for (int z0 = 0; z0 < nz; z0 += 8) {
        float mz[8];
#pragma unroll
        for (int i = 0; i < 8; i++) mz[i] = z0 + i < nz ? p0[(int64_t) (z0 + i) * PK_ATTN_PART + 16 * 256 + r] : PK_NEG_BIG;
#pragma unroll
        for (int i = 0; i < 8; i++) M = fmaxf(M, mz[i]);
    }
    float num = 0.0f, den = 0.0f;
    for (int z0 = 0; z0 < nz; z0 += 8) {
        float mz[8], lz[8], oz[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const bool in = z0 + i < nz;
            const float * pz = p0 + (int64_t) (in ? z0 + i : 0) * PK_ATTN_PART;
            mz[i] = in ? pz[16 * 256 + r] : PK_NEG_BIG;
            lz[i] = pz[16 * 256 + 16 + r];
            oz[i] = pz[r * 256 + t];
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float w = z0 + i < nz ? expf(mz[i] - M) : 0.0f;
            num += oz[i] * w;
            den += lz[i] * w;
        }
    }
    const float gate = sigmoid_f(yin[(int64_t) b * NIN + hq * 512 + 256 + t]);
    const float o = (num / den) * gate;
    quant_block(o, xq, xd, xs, b, hq * 256 + t);
}

// ---------------------------------------------------------------- cross-card exchange

// Two cards trade a buffer through coherent pinned host memory, with no host involvement.
// Cross-device memory must be accessed with atomic (system-scope) loads and stores: a plain
// load is served from the card's own cache and never sees the other card's writes.
static __device__ __forceinline__ int64_t xld_i(const int64_t * p) { return __atomic_load_n(p, __ATOMIC_SEQ_CST); }
static __device__ __forceinline__ void    xst_i(int64_t * p, int64_t v) { __atomic_store_n(p, v, __ATOMIC_SEQ_CST); }
static __device__ __forceinline__ float   xld_f(const float * p) {
    const int v = __atomic_load_n((const int *) p, __ATOMIC_RELAXED);
    float f; memcpy(&f, &v, 4); return f;
}
static __device__ __forceinline__ void    xst_f(float * p, float f) {
    int v; memcpy(&v, &f, 4); __atomic_store_n((int *) p, v, __ATOMIC_RELAXED);
}

#define PK_XCHG_BLOCKS 20
#define PK_XCHG_SPIN_MAX 4000000 // ~2 s; a timeout is reported, never waited out

// grid (PK_XCHG_BLOCKS), block 256. Each workgroup owns a fixed slice of the n elements:
// it publishes its slice of `own` to host_out and raises its flag to seq, then waits for the
// peer's same-numbered workgroup and pulls its slice of the peer's data into `peer`.
// Both cards run this with the same seq, so neither can wait on something not yet promised.
__global__ void k_xchg(const float * __restrict__ own, float * host_out, int64_t * my_flags,
                       const float * host_in, const int64_t * peer_flags, float * __restrict__ peer,
                       int n, int64_t seq, int * err) {
    const int t = threadIdx.x, blk = blockIdx.x;
    const int first = blk * 256 + t, stride = gridDim.x * 256;
    for (int i = first; i < n; i += stride) xst_f(host_out + i, own[i]);
    __syncthreads();
    if (t == 0) {
        xst_i(my_flags + blk, seq);
        int spins = 0;
        while (xld_i(peer_flags + blk) < seq && spins < PK_XCHG_SPIN_MAX) spins++;
        if (spins == PK_XCHG_SPIN_MAX) atomicAdd(err, 1);
    }
    __syncthreads();
    for (int i = first; i < n; i += stride) peer[i] = xld_f(host_in + i);
}

// ---------------------------------------------------------------- argmax

// grid (B), block 256. out[b] = {max logit bits, index} over n logits; lowest index wins ties.
__global__ void k_argmax(const float * __restrict__ logits, int64_t n, int64_t stride, int32_t * __restrict__ out) {
    __shared__ float bv[256];
    __shared__ int   bi[256];
    const int b = blockIdx.x, t = threadIdx.x;
    const float * l = logits + (int64_t) b * stride;
    float best = -INFINITY;
    int   idx = 0x7fffffff;
    for (int64_t j = t; j < n; j += 256) {
        if (l[j] > best) { best = l[j]; idx = (int) j; }
    }
    bv[t] = best; bi[t] = idx;
    __syncthreads();
    if (t == 0) {
        for (int i = 1; i < 256; i++) {
            if (bv[i] > best || (bv[i] == best && bi[i] < idx)) { best = bv[i]; idx = bi[i]; }
        }
        memcpy(&out[b * 2], &best, 4);
        out[b * 2 + 1] = idx;
    }
}

// Merge the two ranks' argmax results on the card. best = {logit bits, local index} per row.
// Both ranks run this on the same pairs and so agree on the token without asking the host.
// grid (1), block B. Writes the winning token to tokens[b] (next step's input) and hist[b].
__global__ void k_pick(const int32_t * __restrict__ own, const int32_t * __restrict__ peer, int own_v0, int peer_v0,
                       int32_t * __restrict__ tokens, int32_t * __restrict__ hist) {
    const int b = threadIdx.x;
    float lo, lp;
    memcpy(&lo, &own[b * 2], 4);
    memcpy(&lp, &peer[b * 2], 4);
    const bool take_peer = lp > lo || (lp == lo && peer_v0 < own_v0);
    const int tok = take_peer ? peer_v0 + peer[b * 2 + 1] : own_v0 + own[b * 2 + 1];
    tokens[b] = tok;
    hist[b] = tok;
}

// ================================================================ C ABI launchers

__global__ void k_wave(int * out) { if (threadIdx.x == 0) out[0] = warpSize; }

extern "C" {

void pk_wave_size(hipStream_t s, int * out) { hipLaunchKernelGGL(k_wave, dim3(1), dim3(32), 0, s, out); }

void pk_unpack_q8(hipStream_t s, const uint8_t * raw, int8_t * qs, uint16_t * d, int64_t n_rows, int nb_src,
                  int nb_dst, int64_t dst_row0, const PkColRanges * cr) {
    hipLaunchKernelGGL(k_unpack_q8, dim3((uint32_t) ((n_rows + 63) / 64)), dim3(64), 0, s, raw, qs, d, n_rows, nb_src, nb_dst,
                       dst_row0, *cr);
}

void pk_embed(hipStream_t s, const int8_t * qs, const void * d, const int32_t * tokens, float * h, int D, int B) {
    hipLaunchKernelGGL(k_embed, dim3(B), dim3(256), 0, s, qs, (const _Float16 *) d, tokens, h, D);
}

void pk_norm_quant(hipStream_t s, float * h, const float * p0, const float * p1, const float * w, float eps, int D,
                   int B, int8_t * xq, float * xd, int xs, int e_off, float * xf) {
    hipLaunchKernelGGL(k_norm_quant, dim3(B), dim3(256), 0, s, h, p0, p1, w, eps, D, xq, xd, xs, e_off, xf);
}

// Scores batch rows [b_off, b_off + N). The activation buffers must be readable for a whole
// number of 16*J-row tiles from b_off (contents past N are ignored).
void pk_gemm(hipStream_t s, const int8_t * W, const void * Wd, const int8_t * xq, const float * xd, int xs, int b_off,
             float * y, int64_t rows, int cols, int N, int y_stride) {
    const uint32_t tiles = (uint32_t) (rows / 16);
    // One pass over the weights costs the same whatever N is, so the kernel is sized to the
    // batch. Up to a tile or two of rows, many waves share each weight tile (few rows are
    // latency-bound otherwise). Beyond that, one wave per (weight tile, group of J activation
    // tiles) and no shared memory: the batch supplies the parallelism, and shared memory would
    // cap how many waves a compute unit keeps in flight. Measured per step at TP2:
    // 24 rows 56 ms (2x16) vs 75 (4x1); 100 rows 117 ms (4x1) vs 193 (8x4); 200 rows 192 vs 290.
#define PK_GEMM_LAUNCH(J, NW) \
    hipLaunchKernelGGL((k_gemm<J, NW>), dim3((N + 16 * J - 1) / (16 * J), tiles), dim3(32 * NW), 0, s, W, (const _Float16 *) Wd, xq, xd, xs, b_off, y, cols, N, y_stride)
    if (N <= 16)       PK_GEMM_LAUNCH(1, PK_GEMM_WAVES);
    else if (N <= 32)  PK_GEMM_LAUNCH(2, 16);
    else if (N <= 512) PK_GEMM_LAUNCH(4, 1);
    else               PK_GEMM_LAUNCH(8, 1);
#undef PK_GEMM_LAUNCH
}

void pk_ffn_act(hipStream_t s, const float * y, int F, int B, int8_t * xq, float * xd, int xs) {
    hipLaunchKernelGGL(k_ffn_act, dim3(F / 256, B), dim3(256), 0, s, y, F, xq, xd, xs);
}

void pk_gdn_conv(hipStream_t s, const float * yin, int NIN, float * mix, float * qkv, int R, const float * conv_w,
                 const float * dt_bias, const float * ssm_a, int Hk, int Hv, float eps, float * mix_s, float * qkv_s,
                 int B, const PkGroups * G) {
    hipLaunchKernelGGL(k_gdn_conv, dim3((2 * Hk + Hv) + 1, B), dim3(128), 0, s, yin, NIN, mix, qkv, R, conv_w, dt_bias,
                       ssm_a, Hk, Hv, eps, mix_s, qkv_s, *G);
}

void pk_gdn_step(hipStream_t s, float * state, const float * qkv, int R, const float * yin, int NIN,
                 const float * norm_w, int Hk, int Hv, float eps, int8_t * xq, float * xd, int xs,
                 const float * qkv_s, const PkGroups * G) {
    hipLaunchKernelGGL(k_gdn_step, dim3(Hv, G->n), dim3(256), 0, s, state, qkv, R, yin, NIN, norm_w, Hk, Hv, eps, xq, xd,
                       xs, qkv_s, *G);
}

void pk_attn_prep(hipStream_t s, const float * yin, int NIN, int Hq, int Hkv, int u0, int kv_off,
                  const float * q_norm_w, const float * k_norm_w, float eps, float theta_scale, int n_rot, void * qr,
                  void * Kc, void * Vc, int B, const PkGroups * G) {
    hipLaunchKernelGGL(k_attn_prep, dim3(Hq + 2 * Hkv - u0, B), dim3(256), 0, s, yin, NIN, Hq, Hkv, u0, kv_off, q_norm_w, k_norm_w, eps,
                       theta_scale, n_rot, (_Float16 *) qr, (_Float16 *) Kc, (_Float16 *) Vc, *G);
}

// Decode-sized batch: one launch covers every segment (S of them per row tile, seg_tiles
// slot tiles each). part must hold tiles * Hq * S partials of PK_ATTN_PART floats; qr must be
// readable up to a whole number of 16-row tiles per group.
void pk_attn(hipStream_t s, const void * qr, const float * yin, int NIN, int Hq, int Hkv, const void * Kc,
             const void * Vc, int seg_tiles, float * part, int tiles, int S, int8_t * xq, float * xd, int xs, int B,
             const PkGroups * G) {
    hipLaunchKernelGGL(k_attn_seg, dim3(Hq, tiles, S), dim3(32), 0, s, (const _Float16 *) qr, Hq, Hkv, (const _Float16 *) Kc,
                       (const _Float16 *) Vc, seg_tiles, part, S, -1, *G);
    hipLaunchKernelGGL(k_attn_merge, dim3(Hq, B), dim3(256), 0, s, part, S, yin, NIN, Hq, seg_tiles, xq, xd, xs, *G);
}

// Large batch: segments [seg_first, seg_last] one launch each, every row tile carrying a single
// partial through them. part must hold tiles * Hq partials.
void pk_attn_swept(hipStream_t s, const void * qr, const float * yin, int NIN, int Hq, int Hkv, const void * Kc,
                   const void * Vc, int seg_tiles, int seg_first, int seg_last, float * part, int tiles, int8_t * xq,
                   float * xd, int xs, int B, const PkGroups * G) {
    for (int seg = seg_first; seg <= seg_last; seg++) {
        hipLaunchKernelGGL(k_attn_seg, dim3(Hq, tiles, 1), dim3(32), 0, s, (const _Float16 *) qr, Hq, Hkv, (const _Float16 *) Kc,
                           (const _Float16 *) Vc, seg_tiles, part, 1, seg, *G);
    }
    // one partial per row tile: merge it as a single segment spanning everything
    hipLaunchKernelGGL(k_attn_merge, dim3(Hq, B), dim3(256), 0, s, part, 1, yin, NIN, Hq, 0x7fffffff, xq, xd, xs, *G);
}

void pk_xchg(hipStream_t s, const float * own, float * host_out, int64_t * my_flags, const float * host_in,
             const int64_t * peer_flags, float * peer, int n, int64_t seq, int * err) {
    hipLaunchKernelGGL(k_xchg, dim3(PK_XCHG_BLOCKS), dim3(256), 0, s, own, host_out, my_flags, host_in, peer_flags, peer,
                       n, seq, err);
}

void pk_pick(hipStream_t s, const int32_t * own, const int32_t * peer, int own_v0, int peer_v0, int32_t * tokens,
             int32_t * hist, int B) {
    hipLaunchKernelGGL(k_pick, dim3(1), dim3(B), 0, s, own, peer, own_v0, peer_v0, tokens, hist);
}

void pk_argmax(hipStream_t s, const float * logits, int64_t n, int64_t stride, int B, int32_t * out) {
    hipLaunchKernelGGL(k_argmax, dim3(B), dim3(256), 0, s, logits, n, stride, out);
}

} // extern "C"

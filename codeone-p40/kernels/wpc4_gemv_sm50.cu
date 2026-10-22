/*
 * ==========================================
 * AUTHOR: M. SZUL
 * AI MODEL: Claude Opus 5
 * TIMESTAMP: 2026-08-25 15:40:00
 * REASON FOR CREATION: Step 3 of GPU_MILESTONE_M2000M.md. The 05:22 sweep proved the
 *   WPC v4 decode is bit-exact on the card for all 253 tensors, but it wrote decoded
 *   f32 back to global memory -- which a real forward pass cannot afford. The token
 *   embedding is 197.1 MiB packed and 1483.8 MiB expanded, so the expanded form does
 *   not even fit beside the model on a 4 GB card. This kernel removes the round trip:
 *   weights are decoded inside the multiply and never exist in memory as f32.
 * MECHANICS: One thread block per output row. Threads stride over the row's packed
 *   byte positions; each position yields TWO weights (low nibble = code j, high nibble
 *   = code j+64, the layout wpc-format chose for SIMD) which are reconstructed with the
 *   affine FMA `w = zero_point + code * scale` and multiplied straight into a private
 *   accumulator against the matching activations. A shared-memory tree reduction sums
 *   the partial products into y[row]. The block header is re-read per position rather
 *   than staged in shared memory: 64 threads read the same 4 bytes, which the cache
 *   broadcasts, and keeping it stateless avoids a barrier in the inner loop.
 * SYSTEM PART: WPC / GPU offload lane.
 * ARCHITECTURE FUNCTION: The matrix-vector product of a GPU-resident WPC runtime --
 *   the dominant operation of single-token decoding. Reads packed weights at 4.25
 *   bits/value, so the memory traffic per token is 7.5x smaller than an f32 model and
 *   ~2x smaller than f16, which is the whole point of keeping WPC on the card.
 * DEPENDENCIES/LINKS: Mirrors QuantBlockV4 in wpc-format/src/lib.rs (BLOCK_SIZE_V4 128,
 *   PACKED_BYTES_V4 64, SIZE 68). Shares its f16 decode with wpc4_decode_sm50.cu so the
 *   two kernels cannot drift. Compiled ahead of time to PTX and loaded by
 *   run_wpc4_gemv.c through the CUDA Driver API.
 * TECH STACK: CUDA C compiled to PTX for sm_50 with the relocated CUDA 12.0 toolchain.
 *   PTX rather than a fatbin because the relocated toolchain ships no libcudart and the
 *   host side uses the Driver API only.
 * LOCAL WORKSPACE: gpu/wpc4-decode/ inside the wpc-engine checkout; built and run in
 *   WSL Ubuntu against /home/aions/qwen3-4b-wpc4/model_v4.wpc
 * GIT COMMIT: PENDING
 * GITHUB METADATA: jpytka666-jpg/wpc-engine, branch feature/gpu-wpc4-decode-sm50
 * ==========================================
 */

#define WPC4_BLOCK_VALUES 128u
#define WPC4_PACKED_BYTES 64u
#define WPC4_BLOCK_BYTES  68u

#define GEMV_THREADS 256

/* IEEE 754 binary16 -> binary32, exact, integer-only. Identical to the copy in
 * wpc4_decode_sm50.cu; kept spelled out for the same reason -- Maxwell has no native
 * half arithmetic, and an explicit decode is provably correct on sm_50 and mirrorable
 * by the host reference. */
__device__ __forceinline__ float wpc4_half_to_float(unsigned short h)
{
    unsigned int sign = ((unsigned int)(h >> 15)) << 31;
    unsigned int exp  = (unsigned int)((h >> 10) & 0x1Fu);
    unsigned int man  = (unsigned int)(h & 0x3FFu);
    unsigned int bits;

    if (exp == 0u) {
        if (man == 0u) {
            bits = sign;
        } else {
            int shift = 0;
            unsigned int m = man;
            while ((m & 0x400u) == 0u) {
                m <<= 1;
                shift++;
            }
            m &= 0x3FFu;
            bits = sign | ((unsigned int)(127 - 15 - shift) << 23) | (m << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23) | (man << 13);
    }

    return __int_as_float((int)bits);
}

/*
 * y = W * x, with W held packed in WPC v4 and never expanded.
 *
 * blocks         : the weight matrix, row-major, n_rows * blocks_per_row * 68 bytes
 * x              : blocks_per_row * 128 activations
 * y              : n_rows results
 * blocks_per_row : in_features / 128
 * n_rows         : out_features
 *
 * Every row of every tensor in Qwen3-4B v4 is a whole number of 128-value blocks
 * (in_features is 2560, 4096 or 9728), so a row never straddles a block boundary and
 * the indexing below needs no special case. The host asserts this before launching.
 */
extern "C" __global__ void wpc4_gemv(const unsigned char * __restrict__ blocks,
                                     const float * __restrict__ x,
                                     float * __restrict__ y,
                                     unsigned int blocks_per_row,
                                     unsigned int n_rows)
{
    __shared__ float partial[GEMV_THREADS];

    unsigned int row = blockIdx.x;
    if (row >= n_rows) {
        return;
    }

    const unsigned char *row_base =
        blocks + (unsigned long long)row
               * (unsigned long long)blocks_per_row
               * (unsigned long long)WPC4_BLOCK_BYTES;

    unsigned int positions = blocks_per_row * WPC4_PACKED_BYTES;
    float acc = 0.0f;

    for (unsigned int p = threadIdx.x; p < positions; p += GEMV_THREADS) {
        unsigned int b = p / WPC4_PACKED_BYTES;
        unsigned int j = p % WPC4_PACKED_BYTES;

        const unsigned char *blk = row_base + (unsigned long long)b * WPC4_BLOCK_BYTES;

        /* Byte-wise read: the 68-byte stride leaves headers unaligned in general. */
        unsigned short zp_bits = (unsigned short)blk[0]
                               | (unsigned short)((unsigned short)blk[1] << 8);
        unsigned short sc_bits = (unsigned short)blk[2]
                               | (unsigned short)((unsigned short)blk[3] << 8);
        float zero_point = wpc4_half_to_float(zp_bits);
        float scale      = wpc4_half_to_float(sc_bits);

        unsigned char byte = blk[4u + j];
        float w_lo = zero_point + (float)(unsigned int)(byte & 0x0Fu) * scale;
        float w_hi = zero_point + (float)(unsigned int)(byte >> 4)    * scale;

        const float *xb = x + (unsigned long long)b * WPC4_BLOCK_VALUES;
        acc += w_lo * xb[j];
        acc += w_hi * xb[j + WPC4_PACKED_BYTES];
    }

    partial[threadIdx.x] = acc;
    __syncthreads();

    for (unsigned int s = GEMV_THREADS / 2u; s > 0u; s >>= 1) {
        if (threadIdx.x < s) {
            partial[threadIdx.x] += partial[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0u) {
        y[row] = partial[0];
    }
}

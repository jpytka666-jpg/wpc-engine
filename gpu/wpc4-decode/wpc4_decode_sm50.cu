/*
 * ==========================================
 * AUTHOR: M. SZUL
 * AI MODEL: Claude Opus 5
 * TIMESTAMP: 2026-08-25 04:05:00
 * REASON FOR CREATION: Step 2 of the GPU_MILESTONE_M2000M.md plan -- "Port the WPC
 *   4-bit decode path to CUDA". The 2026-08-25 02:30 probe proved a synthetic nibble
 *   unpack runs correctly on the physical Quadro M2000M. This kernel does the same
 *   job on the REAL WPC v4 on-disk block layout of the Qwen3-4B model, so the decode
 *   used by inference -- not a stand-in for it -- is the thing executing on the card.
 * MECHANICS: One thread per packed byte. Each thread reads its block's 4-byte header
 *   (f16 zero_point, f16 scale), reads packed byte j, and emits TWO weights:
 *   codes[j] from the low nibble and codes[j + 64] from the high nibble, each
 *   reconstructed with the single affine FMA `w = zero_point + code * scale`. The
 *   f16 -> f32 conversion is written out in integer arithmetic rather than taken from
 *   cuda_fp16.h: Maxwell has no native half arithmetic, and an explicit IEEE 754
 *   binary16 decode is both provably correct on sm_50 and character-for-character
 *   mirrorable by the host reference, which is what makes an exact comparison mean
 *   something.
 * SYSTEM PART: WPC / GPU offload lane.
 * ARCHITECTURE FUNCTION: The decode half of a GPU-resident WPC runtime. Weights stay
 *   packed in VRAM at 4.25 bits/value and are expanded on demand by the card, so the
 *   PCIe transfer measured at 9.16 ms vs 0.30 ms of compute is paid ONCE at load
 *   rather than per token.
 * DEPENDENCIES/LINKS: Mirrors QuantBlockV4 in wpc-format/src/lib.rs -- BLOCK_SIZE_V4
 *   128, PACKED_BYTES_V4 64, SIZE 68, and the "byte j holds code j and code j+64"
 *   layout. If that struct changes, this kernel is wrong and the host comparison will
 *   say so. Compiled ahead of time to PTX and loaded by run_wpc4_decode.c through the
 *   CUDA Driver API.
 * TECH STACK: CUDA C compiled to PTX for sm_50 with the relocated CUDA 12.0 toolchain.
 *   PTX rather than a fatbin because the relocated toolchain ships no libcudart and
 *   the host side uses the Driver API only.
 * LOCAL WORKSPACE: gpu/wpc4-decode/ inside the wpc-engine checkout; built and run in
 *   WSL Ubuntu against /home/aions/qwen3-4b-wpc4/model_v4.wpc
 * GIT COMMIT: 85be400
 * GITHUB METADATA: jpytka666-jpg/wpc-engine, branch feature/gpu-wpc4-decode-sm50 (pushed
 *   2026-08-25); no PR opened yet
 * ==========================================
 */

/* v4 on-disk geometry. These are the numbers from wpc-format/src/lib.rs, spelled
 * here so the kernel is readable on its own; the host asserts them against the file
 * size before launching, so a divergence is caught rather than silently decoded. */
#define WPC4_BLOCK_VALUES 128u
#define WPC4_PACKED_BYTES 64u
#define WPC4_BLOCK_BYTES  68u

/* IEEE 754 binary16 -> binary32, exact, integer-only.
 *
 * Handles the three cases the format can actually contain: normal, zero/subnormal,
 * and inf/nan. Subnormals matter -- a scale of (max-min)/15 on a block of nearly
 * identical weights lands there, and flushing it to zero would quietly destroy that
 * block's values instead of failing loudly. */
__device__ __forceinline__ float wpc4_half_to_float(unsigned short h)
{
    unsigned int sign = ((unsigned int)(h >> 15)) << 31;
    unsigned int exp  = (unsigned int)((h >> 10) & 0x1Fu);
    unsigned int man  = (unsigned int)(h & 0x3FFu);
    unsigned int bits;

    if (exp == 0u) {
        if (man == 0u) {
            bits = sign;                      /* +-0 */
        } else {
            /* Subnormal: shift the mantissa left until the implicit bit appears,
             * then re-bias as a normal float32. */
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
        bits = sign | 0x7F800000u | (man << 13);  /* inf / nan */
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23) | (man << 13);
    }

    return __int_as_float((int)bits);
}

/*
 * Decode `n_blocks` consecutive WPC v4 blocks into f32.
 *
 * blocks : raw on-disk bytes, n_blocks * 68, already resident in VRAM
 * out    : n_blocks * 128 floats
 *
 * Thread t handles byte j = t % 64 of block b = t / 64, and writes out[b*128 + j]
 * and out[b*128 + j + 64]. Both output runs are contiguous, which is the same
 * property the AVX2 path exploits -- the layout was chosen for the decoder, and it
 * pays off identically here.
 */
extern "C" __global__ void wpc4_decode(const unsigned char * __restrict__ blocks,
                                       float * __restrict__ out,
                                       unsigned long long n_blocks)
{
    unsigned long long t =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x
        + (unsigned long long)threadIdx.x;

    unsigned long long total_threads = n_blocks * (unsigned long long)WPC4_PACKED_BYTES;
    if (t >= total_threads) {
        return;
    }

    unsigned long long b = t / (unsigned long long)WPC4_PACKED_BYTES;
    unsigned int       j = (unsigned int)(t % (unsigned long long)WPC4_PACKED_BYTES);

    const unsigned char *blk = blocks + b * (unsigned long long)WPC4_BLOCK_BYTES;

    /* Little-endian f16 pair, read byte-wise: the block stride is 68, so a 68-byte
     * block is not 4-byte aligned in general and a short/int load would fault or
     * silently read the wrong bytes on some architectures. */
    unsigned short zp_bits = (unsigned short)blk[0] | (unsigned short)((unsigned short)blk[1] << 8);
    unsigned short sc_bits = (unsigned short)blk[2] | (unsigned short)((unsigned short)blk[3] << 8);

    float zero_point = wpc4_half_to_float(zp_bits);
    float scale      = wpc4_half_to_float(sc_bits);

    unsigned char byte = blk[4u + j];
    unsigned int  lo   = (unsigned int)(byte & 0x0Fu);
    unsigned int  hi   = (unsigned int)(byte >> 4);

    float *o = out + b * (unsigned long long)WPC4_BLOCK_VALUES;
    o[j]                       = zero_point + (float)lo * scale;
    o[j + WPC4_PACKED_BYTES]   = zero_point + (float)hi * scale;
}

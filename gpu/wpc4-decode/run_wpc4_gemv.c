/*
 * ==========================================
 * AUTHOR: M. SZUL
 * AI MODEL: Claude Opus 5
 * TIMESTAMP: 2026-08-25 15:45:00
 * REASON FOR CREATION: Host half of step 3 of GPU_MILESTONE_M2000M.md. Drives the fused
 *   decode-and-multiply kernel against the real Qwen3-4B v4 weights and answers whether
 *   it is numerically trustworthy. Unlike the decode gate, this one cannot demand
 *   bit-for-bit equality: a GPU tree reduction and a sequential CPU sum add the same
 *   products in different orders, so they legitimately differ in the last bits. The
 *   honest test is therefore a double-precision reference that both single-precision
 *   paths are measured against -- if the GPU is no further from the exact answer than
 *   the CPU is, the port is sound.
 * MECHANICS: Driver API only. Uploads the whole packed model once and keeps it resident,
 *   builds a deterministic activation vector, launches wpc4_gemv over one tensor's rows
 *   straight out of the resident weights, and copies back only the result vector -- a
 *   few kilobytes instead of the hundreds of megabytes the expanded tensor would cost.
 *   It then computes the same product twice on the host, once in float and once in
 *   double, and reports how far each single-precision result sits from the double one.
 *   Timing comes from CUDA events; the reported bandwidth counts packed bytes, which is
 *   the traffic that actually crosses the memory bus.
 * SYSTEM PART: WPC / GPU offload lane.
 * ARCHITECTURE FUNCTION: Gate. Exit code 0 only when the GPU's error against the
 *   double-precision reference is no worse than the host float path's own error by more
 *   than a stated factor. A failure means the fused kernel must not be wired into
 *   inference.
 * DEPENDENCIES/LINKS: cuda.h from the relocated CUDA 12.0 toolchain; libcuda.so.1 from
 *   /usr/lib/wsl/lib; loads the PTX from wpc4_gemv_sm50.cu; consumes model_v4.wpc with
 *   offsets and shapes taken from model_v4.meta.
 * TECH STACK: C11, built with the system gcc. No device syntax on the host side, so
 *   nvcc stays out of the host build entirely.
 * LOCAL WORKSPACE: gpu/wpc4-decode/ inside the wpc-engine checkout; run in WSL Ubuntu.
 * GIT COMMIT: PENDING
 * GITHUB METADATA: jpytka666-jpg/wpc-engine, branch feature/gpu-wpc4-decode-sm50
 * ==========================================
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cuda.h>

#define WPC4_BLOCK_VALUES 128u
#define WPC4_PACKED_BYTES 64u
#define WPC4_BLOCK_BYTES  68u

#define MIB (1024.0 * 1024.0)

/* How much worse than the host float path the GPU is allowed to be, measured against
 * the double-precision reference. Both paths sum the same products in different orders,
 * so neither is exact; the question is whether the GPU is in the same league. */
#define ERROR_BUDGET 4.0

static const char *g_stage = "startup";

#define CHECK(call)                                                            \
    do {                                                                       \
        CUresult _r = (call);                                                  \
        if (_r != CUDA_SUCCESS) {                                              \
            const char *_msg = NULL;                                           \
            cuGetErrorString(_r, &_msg);                                       \
            fprintf(stderr, "FAIL [%s] %s -> %d: %s\n", g_stage, #call,        \
                    (int)_r, _msg ? _msg : "(no message)");                    \
            return 2;                                                          \
        }                                                                      \
    } while (0)

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Host mirror of the kernel's binary16 decode. */
static float half_to_float(uint16_t h)
{
    uint32_t sign = ((uint32_t)(h >> 15)) << 31;
    uint32_t exp  = (uint32_t)((h >> 10) & 0x1Fu);
    uint32_t man  = (uint32_t)(h & 0x3FFu);
    uint32_t bits;
    float out;

    if (exp == 0u) {
        if (man == 0u) {
            bits = sign;
        } else {
            int shift = 0;
            uint32_t m = man;
            while ((m & 0x400u) == 0u) {
                m <<= 1;
                shift++;
            }
            m &= 0x3FFu;
            bits = sign | ((uint32_t)(127 - 15 - shift) << 23) | (m << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23) | (man << 13);
    }

    memcpy(&out, &bits, sizeof(out));
    return out;
}

/* Deterministic activations. A fixed LCG rather than rand() so the vector is identical
 * on every run and on any machine, which makes a failure reproducible. */
static void fill_activations(float *x, uint64_t n)
{
    uint64_t state = 0x2026082515450000ull;
    uint64_t i;
    for (i = 0; i < n; i++) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        /* top 24 bits -> [-1, 1) */
        uint32_t bits = (uint32_t)(state >> 40);
        x[i] = ((float)bits / 8388608.0f) - 1.0f;
    }
}

/* y = W*x on the host. Accumulates in `double` when `exact` is set, in `float`
 * otherwise, so the two calls bracket the GPU result. */
static void reference_gemv(const unsigned char *blocks, const float *x, float *y,
                           uint32_t n_rows, uint32_t blocks_per_row, int exact)
{
    uint32_t row;
    for (row = 0; row < n_rows; row++) {
        const unsigned char *row_base = blocks
            + (uint64_t)row * (uint64_t)blocks_per_row * (uint64_t)WPC4_BLOCK_BYTES;
        double acc_d = 0.0;
        float  acc_f = 0.0f;
        uint32_t b;

        for (b = 0; b < blocks_per_row; b++) {
            const unsigned char *blk = row_base + (uint64_t)b * WPC4_BLOCK_BYTES;
            uint16_t zp_bits = (uint16_t)blk[0] | (uint16_t)((uint16_t)blk[1] << 8);
            uint16_t sc_bits = (uint16_t)blk[2] | (uint16_t)((uint16_t)blk[3] << 8);
            float zero_point = half_to_float(zp_bits);
            float scale      = half_to_float(sc_bits);
            const float *xb  = x + (uint64_t)b * WPC4_BLOCK_VALUES;
            uint32_t j;

            for (j = 0; j < WPC4_PACKED_BYTES; j++) {
                unsigned char byte = blk[4u + j];
                float w_lo = zero_point + (float)(unsigned int)(byte & 0x0Fu) * scale;
                float w_hi = zero_point + (float)(unsigned int)(byte >> 4)    * scale;
                if (exact) {
                    acc_d += (double)w_lo * (double)xb[j];
                    acc_d += (double)w_hi * (double)xb[j + WPC4_PACKED_BYTES];
                } else {
                    acc_f += w_lo * xb[j];
                    acc_f += w_hi * xb[j + WPC4_PACKED_BYTES];
                }
            }
        }
        y[row] = exact ? (float)acc_d : acc_f;
    }
}

/* Largest relative distance between `test` and the double-precision `ref`, with an
 * absolute floor so a near-zero reference does not manufacture an infinite ratio. */
static double max_rel_error(const float *test, const float *ref, uint32_t n,
                            uint32_t *worst_index)
{
    double worst = 0.0;
    uint32_t i;
    *worst_index = 0;
    for (i = 0; i < n; i++) {
        double d = fabs((double)test[i] - (double)ref[i]);
        double scale = fabs((double)ref[i]);
        double rel = d / (scale > 1e-6 ? scale : 1e-6);
        if (rel > worst) {
            worst = rel;
            *worst_index = i;
        }
    }
    return worst;
}

int main(int argc, char **argv)
{
    const char *model_path, *ptx_path, *tensor_name;
    uint64_t tensor_offset, tensor_bytes;
    uint32_t n_rows, in_features, blocks_per_row;

    int fd;
    struct stat st;
    unsigned char *model_map;
    uint64_t model_bytes, expected_bytes;

    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction fn;
    CUdeviceptr d_model = 0, d_x = 0, d_y = 0, d_tensor;
    CUevent ev_start, ev_stop;

    char dev_name[256];
    int cc_major = 0, cc_minor = 0, sm_count = 0;
    size_t vram_free_before = 0, vram_free_after = 0, vram_total = 0;
    double t0, t1, upload_seconds, cpu_double_seconds, cpu_float_seconds;
    float kernel_ms = 0.0f;

    float *x = NULL, *y_gpu = NULL, *y_ref_d = NULL, *y_ref_f = NULL;
    double err_gpu, err_cpu;
    uint32_t worst_gpu = 0, worst_cpu = 0;
    void *params[5];
    int verdict_ok;

    if (argc != 8) {
        fprintf(stderr,
                "usage: %s <model_v4.wpc> <ptx> <tensor_name> <offset_bytes> "
                "<size_bytes> <out_features> <in_features>\n",
                argv[0]);
        return 1;
    }

    model_path    = argv[1];
    ptx_path      = argv[2];
    tensor_name   = argv[3];
    tensor_offset = strtoull(argv[4], NULL, 10);
    tensor_bytes  = strtoull(argv[5], NULL, 10);
    n_rows        = (uint32_t)strtoul(argv[6], NULL, 10);
    in_features   = (uint32_t)strtoul(argv[7], NULL, 10);

    /* ---- geometry, checked rather than trusted ---- */
    if (in_features % WPC4_BLOCK_VALUES != 0) {
        fprintf(stderr, "FAIL: in_features %u is not a multiple of %u; a row would "
                        "straddle a block and this kernel does not handle that.\n",
                in_features, WPC4_BLOCK_VALUES);
        return 2;
    }
    blocks_per_row = in_features / WPC4_BLOCK_VALUES;
    expected_bytes = (uint64_t)n_rows * (uint64_t)blocks_per_row * WPC4_BLOCK_BYTES;
    if (expected_bytes != tensor_bytes) {
        fprintf(stderr, "FAIL: shape %u x %u implies %llu bytes, meta says %llu.\n",
                n_rows, in_features,
                (unsigned long long)expected_bytes, (unsigned long long)tensor_bytes);
        return 2;
    }

    g_stage = "mmap model";
    fd = open(model_path, O_RDONLY);
    if (fd < 0) { perror("open model"); return 2; }
    if (fstat(fd, &st) != 0) { perror("fstat model"); close(fd); return 2; }
    model_bytes = (uint64_t)st.st_size;
    if (tensor_offset + tensor_bytes > model_bytes) {
        fprintf(stderr, "FAIL: tensor range runs past the end of the model.\n");
        close(fd);
        return 2;
    }
    model_map = (unsigned char *)mmap(NULL, (size_t)model_bytes, PROT_READ,
                                      MAP_PRIVATE, fd, 0);
    if (model_map == MAP_FAILED) { perror("mmap model"); close(fd); return 2; }

    printf("=== WPC v4 fused decode + matvec -- real Qwen3-4B weights ===\n");
    printf("tensor          : %s\n", tensor_name);
    printf("shape           : %u x %u  (out x in)\n", n_rows, in_features);
    printf("packed bytes    : %llu (%.1f MiB)\n",
           (unsigned long long)tensor_bytes, (double)tensor_bytes / MIB);
    printf("blocks per row  : %u\n", blocks_per_row);
    printf("if expanded f32 : %.1f MiB -- never materialised by this kernel\n",
           (double)n_rows * in_features * 4.0 / MIB);
    printf("result vector   : %.1f KiB\n", (double)n_rows * 4.0 / 1024.0);
    printf("\n");

    g_stage = "driver init";
    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuDeviceGetName(dev_name, (int)sizeof(dev_name), dev));
    CHECK(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CHECK(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    CHECK(cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    CHECK(cuCtxCreate(&ctx, 0, dev));
    CHECK(cuMemGetInfo(&vram_free_before, &vram_total));

    printf("device          : %s (compute capability %d.%d, %d SMs)\n",
           dev_name, cc_major, cc_minor, sm_count);
    printf("VRAM free before: %.0f MiB\n", (double)vram_free_before / MIB);

    if ((uint64_t)vram_free_before < model_bytes + (uint64_t)in_features * 4 + (uint64_t)n_rows * 4) {
        printf("RESIDENCY: NO -- not enough free VRAM for the packed model plus vectors.\n");
        munmap(model_map, (size_t)model_bytes);
        close(fd);
        return 3;
    }

    g_stage = "upload model";
    CHECK(cuMemAlloc(&d_model, (size_t)model_bytes));
    t0 = now_seconds();
    CHECK(cuMemcpyHtoD(d_model, model_map, (size_t)model_bytes));
    CHECK(cuCtxSynchronize());
    t1 = now_seconds();
    upload_seconds = t1 - t0;
    CHECK(cuMemGetInfo(&vram_free_after, &vram_total));
    printf("VRAM free after : %.0f MiB\n", (double)vram_free_after / MIB);
    printf("model uploaded  : %.3f s (%.2f GB/s), paid once\n",
           upload_seconds, (double)model_bytes / upload_seconds / 1e9);
    printf("\n");

    g_stage = "vectors";
    x       = (float *)malloc((size_t)in_features * sizeof(float));
    y_gpu   = (float *)malloc((size_t)n_rows * sizeof(float));
    y_ref_d = (float *)malloc((size_t)n_rows * sizeof(float));
    y_ref_f = (float *)malloc((size_t)n_rows * sizeof(float));
    if (!x || !y_gpu || !y_ref_d || !y_ref_f) {
        fprintf(stderr, "FAIL: out of host memory\n");
        return 2;
    }
    fill_activations(x, in_features);

    CHECK(cuMemAlloc(&d_x, (size_t)in_features * sizeof(float)));
    CHECK(cuMemAlloc(&d_y, (size_t)n_rows * sizeof(float)));
    CHECK(cuMemcpyHtoD(d_x, x, (size_t)in_features * sizeof(float)));

    g_stage = "load ptx";
    CHECK(cuModuleLoad(&mod, ptx_path));
    CHECK(cuModuleGetFunction(&fn, mod, "wpc4_gemv"));

    g_stage = "launch gemv";
    d_tensor  = d_model + (CUdeviceptr)tensor_offset;
    params[0] = &d_tensor;
    params[1] = &d_x;
    params[2] = &d_y;
    params[3] = &blocks_per_row;
    params[4] = &n_rows;

    CHECK(cuEventCreate(&ev_start, CU_EVENT_DEFAULT));
    CHECK(cuEventCreate(&ev_stop, CU_EVENT_DEFAULT));
    CHECK(cuEventRecord(ev_start, 0));
    CHECK(cuLaunchKernel(fn, n_rows, 1u, 1u, 256u, 1u, 1u, 0u, 0, params, NULL));
    CHECK(cuEventRecord(ev_stop, 0));
    CHECK(cuCtxSynchronize());
    CHECK(cuEventElapsedTime(&kernel_ms, ev_start, ev_stop));
    CHECK(cuMemcpyDtoH(y_gpu, d_y, (size_t)n_rows * sizeof(float)));

    {
        double weights = (double)n_rows * (double)in_features;
        printf("fused kernel    : %.3f ms\n", (double)kernel_ms);
        printf("                  %.1f M weights/s, %.2f GFLOP/s (2 flop per weight)\n",
               weights / ((double)kernel_ms * 1e-3) / 1e6,
               2.0 * weights / ((double)kernel_ms * 1e-3) / 1e9);
        printf("packed traffic  : %.2f GB/s  (f32 weights would need %.2f GB/s for the "
               "same work)\n",
               (double)tensor_bytes / ((double)kernel_ms * 1e-3) / 1e9,
               weights * 4.0 / ((double)kernel_ms * 1e-3) / 1e9);
    }

    g_stage = "host reference";
    t0 = now_seconds();
    reference_gemv(model_map + tensor_offset, x, y_ref_d, n_rows, blocks_per_row, 1);
    t1 = now_seconds();
    cpu_double_seconds = t1 - t0;

    t0 = now_seconds();
    reference_gemv(model_map + tensor_offset, x, y_ref_f, n_rows, blocks_per_row, 0);
    t1 = now_seconds();
    cpu_float_seconds = t1 - t0;

    printf("host reference  : %.3f s double, %.3f s float\n",
           cpu_double_seconds, cpu_float_seconds);
    printf("speed-up vs host: %.0fx (float host path)\n",
           cpu_float_seconds / ((double)kernel_ms * 1e-3));
    printf("\n");

    g_stage = "compare";
    err_gpu = max_rel_error(y_gpu, y_ref_d, n_rows, &worst_gpu);
    err_cpu = max_rel_error(y_ref_f, y_ref_d, n_rows, &worst_cpu);

    printf("=== VERDICT ===\n");
    printf("rows compared   : %u\n", n_rows);
    printf("GPU vs double   : max relative error %.3e (row %u)\n", err_gpu, worst_gpu);
    printf("host float vs double : max relative error %.3e (row %u)\n", err_cpu, worst_cpu);
    printf("sample y[0..3]  : gpu %.6f %.6f %.6f | exact %.6f %.6f %.6f\n",
           (double)y_gpu[0], (double)y_gpu[1], (double)y_gpu[2],
           (double)y_ref_d[0], (double)y_ref_d[1], (double)y_ref_d[2]);

    /* Bit-exactness is not available here and claiming it would be a lie: the GPU sums
     * in a tree, the host sums in sequence. The GPU tree is usually the MORE accurate
     * of the two, so the budget is a sanity bound, not a tight tolerance. */
    verdict_ok = (err_gpu <= ERROR_BUDGET * err_cpu) || (err_gpu < 1e-5);
    printf("RESULT          : %s\n",
           verdict_ok
               ? "PASS -- fused GPU result is as close to exact as the host float path"
               : "FAIL -- fused GPU result drifts further than summation order explains");

    free(x);
    free(y_gpu);
    free(y_ref_d);
    free(y_ref_f);
    cuMemFree(d_y);
    cuMemFree(d_x);
    cuMemFree(d_model);
    cuCtxDestroy(ctx);
    munmap(model_map, (size_t)model_bytes);
    close(fd);

    return verdict_ok ? 0 : 4;
}

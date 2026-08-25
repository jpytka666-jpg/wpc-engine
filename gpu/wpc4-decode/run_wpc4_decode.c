/*
 * ==========================================
 * AUTHOR: M. SZUL
 * AI MODEL: Claude Opus 5
 * TIMESTAMP: 2026-08-25 04:10:00
 * REASON FOR CREATION: Host half of step 2 of GPU_MILESTONE_M2000M.md. Answers two
 *   questions the milestone leaves open and that decide whether a GPU-resident WPC
 *   runtime is possible at all on this card: (1) does the whole 4-bit Qwen3-4B model
 *   actually fit and stay resident in the Quadro M2000M's VRAM, and (2) does the
 *   CUDA decode of the real on-disk v4 blocks reproduce the CPU decode exactly.
 * MECHANICS: Driver API only. Initialises the driver, reports device identity and
 *   free VRAM, mmaps the packed model, uploads it whole to the card and re-reads free
 *   VRAM to prove residency, then launches wpc4_decode over ONE named tensor's block
 *   range straight out of the resident copy -- no re-upload. The result is copied back
 *   and compared bit-for-bit against a host reference that mirrors the kernel's
 *   arithmetic. Kernel time comes from CUDA events, transfer time from the monotonic
 *   clock, so the run also yields the per-load transfer cost the milestone's step 4
 *   asks for.
 * SYSTEM PART: WPC / GPU offload lane.
 * ARCHITECTURE FUNCTION: Gate. Exit code 0 only when the model is resident AND every
 *   decoded value matches the host reference bit-for-bit. Any mismatch means the GPU
 *   decode path must not be wired into inference.
 * DEPENDENCIES/LINKS: cuda.h from the relocated CUDA 12.0 toolchain for declarations;
 *   libcuda.so.1 from /usr/lib/wsl/lib at link and run time; loads the PTX emitted from
 *   wpc4_decode_sm50.cu; consumes model_v4.wpc and the offsets recorded in
 *   model_v4.meta. Deliberately avoids libcudart, which the relocated toolchain lacks.
 * TECH STACK: C11, built with the system gcc. C rather than CUDA C because the host
 *   side contains no device syntax, which keeps nvcc out of the host build entirely --
 *   the same split that made the 2026-08-25 sm_50 probe build cleanly.
 * LOCAL WORKSPACE: gpu/wpc4-decode/ inside the wpc-engine checkout; built and run in
 *   WSL Ubuntu.
 * GIT COMMIT: 85be400
 * GITHUB METADATA: jpytka666-jpg/wpc-engine, branch feature/gpu-wpc4-decode-sm50 (pushed
 *   2026-08-25); no PR opened yet
 * ==========================================
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cuda.h>

/* v4 geometry -- must agree with QuantBlockV4 in wpc-format/src/lib.rs. */
#define WPC4_BLOCK_VALUES 128u
#define WPC4_PACKED_BYTES 64u
#define WPC4_BLOCK_BYTES  68u

#define MIB (1024.0 * 1024.0)

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

/* Host mirror of the kernel's IEEE 754 binary16 decode. Written out rather than
 * taken from a library so that "GPU matches CPU" compares two implementations of
 * the same documented rule instead of one implementation against itself. */
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

/* Host reference decode of one tensor's blocks, mirroring wpc4_decode exactly. */
static void reference_decode(const unsigned char *blocks, float *out, uint64_t n_blocks)
{
    uint64_t b;
    for (b = 0; b < n_blocks; b++) {
        const unsigned char *blk = blocks + b * (uint64_t)WPC4_BLOCK_BYTES;
        uint16_t zp_bits = (uint16_t)blk[0] | (uint16_t)((uint16_t)blk[1] << 8);
        uint16_t sc_bits = (uint16_t)blk[2] | (uint16_t)((uint16_t)blk[3] << 8);
        float zero_point = half_to_float(zp_bits);
        float scale      = half_to_float(sc_bits);
        float *o = out + b * (uint64_t)WPC4_BLOCK_VALUES;
        unsigned int j;
        for (j = 0; j < WPC4_PACKED_BYTES; j++) {
            unsigned char byte = blk[4u + j];
            unsigned int lo = (unsigned int)(byte & 0x0Fu);
            unsigned int hi = (unsigned int)(byte >> 4);
            o[j]                     = zero_point + (float)lo * scale;
            o[j + WPC4_PACKED_BYTES] = zero_point + (float)hi * scale;
        }
    }
}

static int bits_differ(float a, float b)
{
    uint32_t ba, bb;
    memcpy(&ba, &a, sizeof(ba));
    memcpy(&bb, &b, sizeof(bb));
    return ba != bb;
}

int main(int argc, char **argv)
{
    const char *model_path;
    const char *ptx_path;
    const char *tensor_name;
    uint64_t tensor_offset;
    uint64_t tensor_bytes;

    int fd;
    struct stat st;
    unsigned char *model_map;
    uint64_t model_bytes;
    uint64_t n_blocks;
    uint64_t n_values;
    uint64_t out_bytes;

    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction fn;
    CUdeviceptr d_model = 0;
    CUdeviceptr d_out = 0;
    CUevent ev_start, ev_stop;

    char dev_name[256];
    int cc_major = 0, cc_minor = 0, sm_count = 0;
    size_t vram_free_before = 0, vram_free_after = 0, vram_total = 0;
    double t0, t1, upload_seconds, download_seconds;
    float kernel_ms = 0.0f;

    float *gpu_out = NULL;
    float *ref_out = NULL;
    uint64_t mismatches = 0;
    uint64_t first_bad = 0;
    double max_abs_diff = 0.0;
    uint64_t i;

    unsigned int threads_per_block = 256u;
    uint64_t total_threads;
    uint64_t grid_blocks;
    void *params[3];

    if (argc != 6) {
        fprintf(stderr,
                "usage: %s <model_v4.wpc> <wpc4_decode_sm50.ptx> <tensor_name> "
                "<offset_bytes> <size_bytes>\n",
                argv[0]);
        return 1;
    }

    model_path    = argv[1];
    ptx_path      = argv[2];
    tensor_name   = argv[3];
    tensor_offset = strtoull(argv[4], NULL, 10);
    tensor_bytes  = strtoull(argv[5], NULL, 10);

    /* ---- geometry check before anything expensive happens ---- */
    if (tensor_bytes % (uint64_t)WPC4_BLOCK_BYTES != 0) {
        fprintf(stderr,
                "FAIL: tensor size %llu is not a multiple of the v4 block size %u. "
                "Either the meta file or the format assumption is wrong.\n",
                (unsigned long long)tensor_bytes, WPC4_BLOCK_BYTES);
        return 2;
    }
    n_blocks  = tensor_bytes / (uint64_t)WPC4_BLOCK_BYTES;
    n_values  = n_blocks * (uint64_t)WPC4_BLOCK_VALUES;
    out_bytes = n_values * sizeof(float);

    /* ---- map the packed model ---- */
    g_stage = "mmap model";
    fd = open(model_path, O_RDONLY);
    if (fd < 0) {
        perror("open model");
        return 2;
    }
    if (fstat(fd, &st) != 0) {
        perror("fstat model");
        close(fd);
        return 2;
    }
    model_bytes = (uint64_t)st.st_size;
    if (tensor_offset + tensor_bytes > model_bytes) {
        fprintf(stderr, "FAIL: tensor range runs past the end of the model file.\n");
        close(fd);
        return 2;
    }
    model_map = (unsigned char *)mmap(NULL, (size_t)model_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (model_map == MAP_FAILED) {
        perror("mmap model");
        close(fd);
        return 2;
    }

    printf("=== WPC v4 GPU decode -- real Qwen3-4B weights ===\n");
    printf("model file      : %s\n", model_path);
    printf("model bytes     : %llu (%.1f MiB)\n",
           (unsigned long long)model_bytes, (double)model_bytes / MIB);
    printf("tensor          : %s\n", tensor_name);
    printf("tensor offset   : %llu\n", (unsigned long long)tensor_offset);
    printf("tensor bytes    : %llu (%.1f MiB)\n",
           (unsigned long long)tensor_bytes, (double)tensor_bytes / MIB);
    printf("blocks          : %llu\n", (unsigned long long)n_blocks);
    printf("values          : %llu\n", (unsigned long long)n_values);
    printf("decoded f32     : %.1f MiB\n", (double)out_bytes / MIB);
    printf("bits per value  : %.4f\n",
           (double)tensor_bytes * 8.0 / (double)n_values);
    printf("\n");

    /* ---- device ---- */
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
    printf("VRAM total      : %.0f MiB\n", (double)vram_total / MIB);
    printf("VRAM free before: %.0f MiB\n", (double)vram_free_before / MIB);

    if ((uint64_t)vram_free_before < model_bytes) {
        printf("\nRESIDENCY: NO -- %.0f MiB free, model needs %.0f MiB. "
               "The model cannot stay resident on this card.\n",
               (double)vram_free_before / MIB, (double)model_bytes / MIB);
        munmap(model_map, (size_t)model_bytes);
        close(fd);
        return 3;
    }

    /* ---- upload the whole packed model and prove it stays there ---- */
    g_stage = "upload model";
    CHECK(cuMemAlloc(&d_model, (size_t)model_bytes));
    t0 = now_seconds();
    CHECK(cuMemcpyHtoD(d_model, model_map, (size_t)model_bytes));
    CHECK(cuCtxSynchronize());
    t1 = now_seconds();
    upload_seconds = t1 - t0;
    CHECK(cuMemGetInfo(&vram_free_after, &vram_total));

    printf("VRAM free after : %.0f MiB\n", (double)vram_free_after / MIB);
    printf("model uploaded  : %.3f s (%.2f GB/s)\n",
           upload_seconds,
           (double)model_bytes / upload_seconds / 1e9);
    printf("RESIDENCY       : YES -- whole packed model lives in VRAM, "
           "%.0f MiB left for KV cache and activations\n",
           (double)vram_free_after / MIB);
    printf("\n");

    /* ---- decode one tensor straight out of the resident copy ---- */
    g_stage = "load ptx";
    CHECK(cuModuleLoad(&mod, ptx_path));
    CHECK(cuModuleGetFunction(&fn, mod, "wpc4_decode"));

    g_stage = "allocate output";
    CHECK(cuMemAlloc(&d_out, (size_t)out_bytes));

    total_threads = n_blocks * (uint64_t)WPC4_PACKED_BYTES;
    grid_blocks   = (total_threads + threads_per_block - 1) / threads_per_block;

    {
        CUdeviceptr d_tensor = d_model + (CUdeviceptr)tensor_offset;
        params[0] = &d_tensor;
        params[1] = &d_out;
        params[2] = &n_blocks;

        g_stage = "launch decode";
        CHECK(cuEventCreate(&ev_start, CU_EVENT_DEFAULT));
        CHECK(cuEventCreate(&ev_stop, CU_EVENT_DEFAULT));
        CHECK(cuEventRecord(ev_start, 0));
        CHECK(cuLaunchKernel(fn,
                             (unsigned int)grid_blocks, 1u, 1u,
                             threads_per_block, 1u, 1u,
                             0u, 0, params, NULL));
        CHECK(cuEventRecord(ev_stop, 0));
        CHECK(cuCtxSynchronize());
        CHECK(cuEventElapsedTime(&kernel_ms, ev_start, ev_stop));
    }

    printf("grid            : %llu blocks x %u threads\n",
           (unsigned long long)grid_blocks, threads_per_block);
    printf("decode kernel   : %.3f ms (%.1f M values/s)\n",
           (double)kernel_ms,
           (double)n_values / ((double)kernel_ms * 1e-3) / 1e6);

    /* ---- bring it back and check every single value ---- */
    g_stage = "download result";
    gpu_out = (float *)malloc((size_t)out_bytes);
    ref_out = (float *)malloc((size_t)out_bytes);
    if (!gpu_out || !ref_out) {
        fprintf(stderr, "FAIL: out of host memory for %.1f MiB x2\n", (double)out_bytes / MIB);
        free(gpu_out);
        free(ref_out);
        return 2;
    }
    t0 = now_seconds();
    CHECK(cuMemcpyDtoH(gpu_out, d_out, (size_t)out_bytes));
    t1 = now_seconds();
    download_seconds = t1 - t0;
    printf("result download : %.3f s (%.2f GB/s)\n",
           download_seconds, (double)out_bytes / download_seconds / 1e9);

    g_stage = "host reference";
    t0 = now_seconds();
    reference_decode(model_map + tensor_offset, ref_out, n_blocks);
    t1 = now_seconds();
    printf("host reference  : %.3f s (%.1f M values/s)\n",
           t1 - t0, (double)n_values / (t1 - t0) / 1e6);
    printf("\n");

    g_stage = "compare";
    for (i = 0; i < n_values; i++) {
        if (bits_differ(gpu_out[i], ref_out[i])) {
            double d = (double)gpu_out[i] - (double)ref_out[i];
            if (d < 0.0) {
                d = -d;
            }
            if (d > max_abs_diff) {
                max_abs_diff = d;
            }
            if (mismatches == 0) {
                first_bad = i;
            }
            mismatches++;
        }
    }

    printf("=== VERDICT ===\n");
    printf("values checked  : %llu\n", (unsigned long long)n_values);
    printf("mismatches      : %llu\n", (unsigned long long)mismatches);
    if (mismatches != 0) {
        printf("first mismatch  : index %llu, gpu %.9g vs cpu %.9g\n",
               (unsigned long long)first_bad,
               (double)gpu_out[first_bad], (double)ref_out[first_bad]);
        printf("max abs diff    : %.9g\n", max_abs_diff);
        printf("RESULT          : FAIL -- GPU decode does not reproduce the CPU decode\n");
    } else {
        printf("sample values   : %.6f %.6f %.6f %.6f\n",
               (double)gpu_out[0], (double)gpu_out[1],
               (double)gpu_out[n_values / 2], (double)gpu_out[n_values - 1]);
        printf("RESULT          : PASS -- bit-for-bit identical to the CPU decode\n");
    }

    free(gpu_out);
    free(ref_out);
    cuMemFree(d_out);
    cuMemFree(d_model);
    cuCtxDestroy(ctx);
    munmap(model_map, (size_t)model_bytes);
    close(fd);

    return mismatches == 0 ? 0 : 4;
}

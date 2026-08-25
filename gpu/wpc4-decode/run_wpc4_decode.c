/*
 * ==========================================
 * AUTHOR: M. SZUL
 * AI MODEL: Claude Opus 5
 * TIMESTAMP: 2026-08-25 05:25:00
 * REASON FOR CREATION: Host half of step 2 of GPU_MILESTONE_M2000M.md. Answers two
 *   questions the milestone leaves open and that decide whether a GPU-resident WPC
 *   runtime is possible at all on this card: (1) does the whole 4-bit Qwen3-4B model
 *   fit and stay resident in the Quadro M2000M's VRAM, and (2) does the CUDA decode of
 *   the real on-disk v4 blocks reproduce the CPU decode exactly.
 * MECHANICS: Driver API only. Initialises the driver, reports device identity and free
 *   VRAM, mmaps the packed model, uploads it whole to the card and re-reads free VRAM
 *   to prove residency, then decodes tensors straight out of the resident copy with no
 *   re-upload. Every decoded value is copied back and compared bit-for-bit against a
 *   host reference implementing the same documented rule. Two modes: one named tensor,
 *   or a manifest of them -- the manifest mode exists because the upload is the
 *   expensive part and is paid once, so sweeping all 253 tensors costs barely more than
 *   sweeping one. Kernel time comes from CUDA events, transfer time from the monotonic
 *   clock, so the run also yields the per-load transfer cost step 4 asks for.
 * SYSTEM PART: WPC / GPU offload lane.
 * ARCHITECTURE FUNCTION: Gate. Exit code 0 only when the model is resident AND every
 *   decoded value of every requested tensor matches the host reference bit-for-bit. Any
 *   mismatch means the GPU decode path must not be wired into inference.
 * DEPENDENCIES/LINKS: cuda.h from the relocated CUDA 12.0 toolchain for declarations;
 *   libcuda.so.1 from /usr/lib/wsl/lib at link and run time; loads the PTX emitted from
 *   wpc4_decode_sm50.cu; consumes model_v4.wpc and offsets taken from model_v4.meta,
 *   passed in as a plain-text manifest so the C side needs no JSON parser.
 * TECH STACK: C11, built with the system gcc. C rather than CUDA C because the host
 *   side contains no device syntax, which keeps nvcc out of the host build entirely --
 *   the same split that made the 2026-08-25 sm_50 probe build cleanly.
 * LOCAL WORKSPACE: gpu/wpc4-decode/ inside the wpc-engine checkout; built and run in
 *   WSL Ubuntu.
 * GIT COMMIT: 85be400 (first version); manifest sweep mode PENDING
 * GITHUB METADATA: jpytka666-jpg/wpc-engine, branch feature/gpu-wpc4-decode-sm50
 *   (pushed 2026-08-25); no PR opened yet
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
#define MAX_NAME 512

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

/* Host mirror of the kernel's IEEE 754 binary16 decode. Written out rather than taken
 * from a library so that "GPU matches CPU" compares two implementations of the same
 * documented rule instead of one implementation against itself. */
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

/* Everything the sweep accumulates, so the summary is measured rather than retold. */
struct totals {
    uint64_t tensors;
    uint64_t values;
    uint64_t mismatches;
    uint64_t failed_tensors;
    double   kernel_ms;
    double   best_rate;      /* M values/s */
    char     best_name[MAX_NAME];
};

/*
 * Decode one tensor out of the resident model and check every value.
 * Returns 0 when the tensor matched, 1 when it did not, negative on a hard error.
 */
static int decode_one(CUfunction fn,
                      CUdeviceptr d_model,
                      CUdeviceptr d_out,
                      const unsigned char *model_map,
                      const char *name,
                      uint64_t offset,
                      uint64_t bytes,
                      float *gpu_out,
                      float *ref_out,
                      int verbose,
                      struct totals *acc)
{
    uint64_t n_blocks, n_values, out_bytes, i;
    uint64_t mismatches = 0, first_bad = 0;
    unsigned int threads_per_block = 256u;
    uint64_t total_threads, grid_blocks;
    CUdeviceptr d_tensor;
    CUevent ev_start, ev_stop;
    float kernel_ms = 0.0f;
    double rate;
    void *params[3];
    CUresult r;

    if (bytes % (uint64_t)WPC4_BLOCK_BYTES != 0) {
        fprintf(stderr, "FAIL %s: size %llu is not a multiple of the v4 block size %u\n",
                name, (unsigned long long)bytes, WPC4_BLOCK_BYTES);
        return -1;
    }

    n_blocks  = bytes / (uint64_t)WPC4_BLOCK_BYTES;
    n_values  = n_blocks * (uint64_t)WPC4_BLOCK_VALUES;
    out_bytes = n_values * sizeof(float);

    total_threads = n_blocks * (uint64_t)WPC4_PACKED_BYTES;
    grid_blocks   = (total_threads + threads_per_block - 1) / threads_per_block;

    d_tensor  = d_model + (CUdeviceptr)offset;
    params[0] = &d_tensor;
    params[1] = &d_out;
    params[2] = &n_blocks;

    r = cuEventCreate(&ev_start, CU_EVENT_DEFAULT);
    if (r != CUDA_SUCCESS) return -1;
    r = cuEventCreate(&ev_stop, CU_EVENT_DEFAULT);
    if (r != CUDA_SUCCESS) return -1;

    if (cuEventRecord(ev_start, 0) != CUDA_SUCCESS) return -1;
    r = cuLaunchKernel(fn, (unsigned int)grid_blocks, 1u, 1u,
                       threads_per_block, 1u, 1u, 0u, 0, params, NULL);
    if (r != CUDA_SUCCESS) {
        const char *msg = NULL;
        cuGetErrorString(r, &msg);
        fprintf(stderr, "FAIL %s: launch -> %s\n", name, msg ? msg : "(no message)");
        return -1;
    }
    if (cuEventRecord(ev_stop, 0) != CUDA_SUCCESS) return -1;
    if (cuCtxSynchronize() != CUDA_SUCCESS) return -1;
    if (cuEventElapsedTime(&kernel_ms, ev_start, ev_stop) != CUDA_SUCCESS) return -1;
    cuEventDestroy(ev_start);
    cuEventDestroy(ev_stop);

    if (cuMemcpyDtoH(gpu_out, d_out, (size_t)out_bytes) != CUDA_SUCCESS) return -1;

    reference_decode(model_map + offset, ref_out, n_blocks);

    for (i = 0; i < n_values; i++) {
        if (bits_differ(gpu_out[i], ref_out[i])) {
            if (mismatches == 0) {
                first_bad = i;
            }
            mismatches++;
        }
    }

    rate = (double)n_values / ((double)kernel_ms * 1e-3) / 1e6;

    acc->tensors++;
    acc->values     += n_values;
    acc->mismatches += mismatches;
    acc->kernel_ms  += (double)kernel_ms;
    if (rate > acc->best_rate) {
        acc->best_rate = rate;
        snprintf(acc->best_name, MAX_NAME, "%s", name);
    }
    if (mismatches != 0) {
        acc->failed_tensors++;
    }

    if (verbose) {
        printf("blocks          : %llu\n", (unsigned long long)n_blocks);
        printf("values          : %llu\n", (unsigned long long)n_values);
        printf("decoded f32     : %.1f MiB\n", (double)out_bytes / MIB);
        printf("bits per value  : %.4f\n", (double)bytes * 8.0 / (double)n_values);
        printf("grid            : %llu blocks x %u threads\n",
               (unsigned long long)grid_blocks, threads_per_block);
        printf("decode kernel   : %.3f ms (%.1f M values/s)\n", (double)kernel_ms, rate);
        printf("values checked  : %llu\n", (unsigned long long)n_values);
        printf("mismatches      : %llu\n", (unsigned long long)mismatches);
        if (mismatches != 0) {
            printf("first mismatch  : index %llu, gpu %.9g vs cpu %.9g\n",
                   (unsigned long long)first_bad,
                   (double)gpu_out[first_bad], (double)ref_out[first_bad]);
        } else {
            printf("sample values   : %.6f %.6f %.6f\n",
                   (double)gpu_out[0], (double)gpu_out[n_values / 2],
                   (double)gpu_out[n_values - 1]);
        }
    } else {
        printf("%-52s %10llu values  %8.3f ms  %s\n",
               name, (unsigned long long)n_values, (double)kernel_ms,
               mismatches == 0 ? "OK" : "MISMATCH");
        if (mismatches != 0) {
            printf("    first mismatch at %llu: gpu %.9g vs cpu %.9g\n",
                   (unsigned long long)first_bad,
                   (double)gpu_out[first_bad], (double)ref_out[first_bad]);
        }
        fflush(stdout);
    }

    return mismatches == 0 ? 0 : 1;
}

int main(int argc, char **argv)
{
    const char *model_path;
    const char *ptx_path;
    const char *manifest_path = NULL;
    const char *tensor_name = NULL;
    uint64_t tensor_offset = 0, tensor_bytes = 0;
    int sweep;

    int fd;
    struct stat st;
    unsigned char *model_map;
    uint64_t model_bytes;
    uint64_t max_tensor_bytes = 0, max_out_bytes;

    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction fn;
    CUdeviceptr d_model = 0, d_out = 0;

    char dev_name[256];
    int cc_major = 0, cc_minor = 0, sm_count = 0;
    size_t vram_free_before = 0, vram_free_after = 0, vram_total = 0;
    double t0, t1, upload_seconds, sweep_seconds;

    float *gpu_out = NULL, *ref_out = NULL;
    struct totals acc;

    FILE *mf = NULL;
    char line[MAX_NAME + 128];

    memset(&acc, 0, sizeof(acc));

    if (argc == 4) {
        sweep         = 1;
        model_path    = argv[1];
        ptx_path      = argv[2];
        manifest_path = argv[3];
    } else if (argc == 6) {
        sweep         = 0;
        model_path    = argv[1];
        ptx_path      = argv[2];
        tensor_name   = argv[3];
        tensor_offset = strtoull(argv[4], NULL, 10);
        tensor_bytes  = strtoull(argv[5], NULL, 10);
    } else {
        fprintf(stderr,
                "usage: %s <model_v4.wpc> <ptx> <manifest>\n"
                "       %s <model_v4.wpc> <ptx> <tensor_name> <offset_bytes> <size_bytes>\n"
                "\n"
                "manifest lines: <offset_bytes> <size_bytes> <tensor_name>\n",
                argv[0], argv[0]);
        return 1;
    }

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
    model_map = (unsigned char *)mmap(NULL, (size_t)model_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (model_map == MAP_FAILED) {
        perror("mmap model");
        close(fd);
        return 2;
    }

    /* ---- work out the biggest tensor so the output buffer is allocated once ---- */
    g_stage = "read manifest";
    if (sweep) {
        mf = fopen(manifest_path, "r");
        if (!mf) {
            perror("open manifest");
            munmap(model_map, (size_t)model_bytes);
            close(fd);
            return 2;
        }
        while (fgets(line, (int)sizeof(line), mf)) {
            unsigned long long off = 0, sz = 0;
            if (sscanf(line, "%llu %llu", &off, &sz) == 2) {
                if ((uint64_t)sz > max_tensor_bytes) {
                    max_tensor_bytes = (uint64_t)sz;
                }
            }
        }
        rewind(mf);
    } else {
        max_tensor_bytes = tensor_bytes;
    }
    if (max_tensor_bytes == 0) {
        fprintf(stderr, "FAIL: no tensors to decode\n");
        if (mf) fclose(mf);
        munmap(model_map, (size_t)model_bytes);
        close(fd);
        return 2;
    }
    max_out_bytes = (max_tensor_bytes / (uint64_t)WPC4_BLOCK_BYTES)
                    * (uint64_t)WPC4_BLOCK_VALUES * sizeof(float);

    printf("=== WPC v4 GPU decode -- real Qwen3-4B weights ===\n");
    printf("model file      : %s\n", model_path);
    printf("model bytes     : %llu (%.1f MiB)\n",
           (unsigned long long)model_bytes, (double)model_bytes / MIB);
    printf("mode            : %s\n", sweep ? "sweep (manifest)" : "single tensor");
    printf("largest tensor  : %.1f MiB packed, %.1f MiB decoded\n",
           (double)max_tensor_bytes / MIB, (double)max_out_bytes / MIB);
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

    if ((uint64_t)vram_free_before < model_bytes + max_out_bytes) {
        printf("\nRESIDENCY: NO -- %.0f MiB free, model plus output needs %.0f MiB.\n",
               (double)vram_free_before / MIB,
               (double)(model_bytes + max_out_bytes) / MIB);
        if (mf) fclose(mf);
        munmap(model_map, (size_t)model_bytes);
        close(fd);
        return 3;
    }

    /* ---- upload the whole packed model, once ---- */
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
    printf("RESIDENCY       : YES -- packed model lives in VRAM, %.0f MiB left\n",
           (double)vram_free_after / MIB);
    printf("\n");

    g_stage = "load ptx";
    CHECK(cuModuleLoad(&mod, ptx_path));
    CHECK(cuModuleGetFunction(&fn, mod, "wpc4_decode"));

    g_stage = "allocate output";
    CHECK(cuMemAlloc(&d_out, (size_t)max_out_bytes));
    gpu_out = (float *)malloc((size_t)max_out_bytes);
    ref_out = (float *)malloc((size_t)max_out_bytes);
    if (!gpu_out || !ref_out) {
        fprintf(stderr, "FAIL: out of host memory for %.1f MiB x2\n",
                (double)max_out_bytes / MIB);
        free(gpu_out);
        free(ref_out);
        if (mf) fclose(mf);
        return 2;
    }

    /* ---- decode ---- */
    g_stage = "decode";
    t0 = now_seconds();
    if (sweep) {
        while (fgets(line, (int)sizeof(line), mf)) {
            unsigned long long off = 0, sz = 0;
            char name[MAX_NAME];
            int got = sscanf(line, "%llu %llu %511s", &off, &sz, name);
            if (got != 3) {
                continue;
            }
            if ((uint64_t)off + (uint64_t)sz > model_bytes) {
                fprintf(stderr, "FAIL %s: range runs past the end of the model\n", name);
                acc.failed_tensors++;
                continue;
            }
            if (decode_one(fn, d_model, d_out, model_map, name,
                           (uint64_t)off, (uint64_t)sz,
                           gpu_out, ref_out, 0, &acc) < 0) {
                acc.failed_tensors++;
            }
        }
        fclose(mf);
    } else {
        printf("tensor          : %s\n", tensor_name);
        printf("tensor offset   : %llu\n", (unsigned long long)tensor_offset);
        printf("tensor bytes    : %llu (%.1f MiB)\n",
               (unsigned long long)tensor_bytes, (double)tensor_bytes / MIB);
        if (tensor_offset + tensor_bytes > model_bytes) {
            fprintf(stderr, "FAIL: tensor range runs past the end of the model file.\n");
            return 2;
        }
        decode_one(fn, d_model, d_out, model_map, tensor_name,
                   tensor_offset, tensor_bytes, gpu_out, ref_out, 1, &acc);
    }
    t1 = now_seconds();
    sweep_seconds = t1 - t0;

    /* ---- verdict ---- */
    printf("\n=== VERDICT ===\n");
    printf("tensors decoded : %llu\n", (unsigned long long)acc.tensors);
    printf("values checked  : %llu\n", (unsigned long long)acc.values);
    printf("mismatches      : %llu\n", (unsigned long long)acc.mismatches);
    printf("failed tensors  : %llu\n", (unsigned long long)acc.failed_tensors);
    printf("kernel time     : %.3f ms total\n", acc.kernel_ms);
    if (acc.kernel_ms > 0.0) {
        printf("mean rate       : %.1f M values/s\n",
               (double)acc.values / (acc.kernel_ms * 1e-3) / 1e6);
        printf("best tensor     : %.1f M values/s (%s)\n", acc.best_rate, acc.best_name);
    }
    printf("wall clock      : %.3f s decode+compare (upload %.3f s excluded)\n",
           sweep_seconds, upload_seconds);
    printf("RESULT          : %s\n",
           (acc.mismatches == 0 && acc.failed_tensors == 0)
               ? "PASS -- bit-for-bit identical to the CPU decode"
               : "FAIL -- GPU decode does not reproduce the CPU decode");

    free(gpu_out);
    free(ref_out);
    cuMemFree(d_out);
    cuMemFree(d_model);
    cuCtxDestroy(ctx);
    munmap(model_map, (size_t)model_bytes);
    close(fd);

    return (acc.mismatches == 0 && acc.failed_tensors == 0) ? 0 : 4;
}

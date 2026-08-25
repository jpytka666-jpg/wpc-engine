# WPC v4 decode on the GPU — Qwen3-4B resident on a Quadro M2000M

Date: 2026-08-25
Closes: step 2 of `GPU_MILESTONE_M2000M.md` ("Port the WPC 4-bit decode path to CUDA")
Branch: `feature/gpu-wpc4-decode-sm50`

## What this run answers

The 2026-08-25 02:30 probe proved a *synthetic* nibble unpack executes correctly on the
physical card. It did not touch the real model. Two questions stayed open, and both of
them decide whether a GPU-resident WPC runtime is possible on this hardware at all:

1. Does the whole 4-bit Qwen3-4B model fit and **stay** in the M2000M's 4 GB of VRAM?
2. Does a CUDA decode of the **real** v4 on-disk blocks reproduce the CPU decode exactly?

Both are now answered by measurement, not by argument.

## Verified — residency

| Measured | Result |
|---|---|
| Device | Quadro M2000M, compute capability 5.0, 5 SMs |
| VRAM total | 4096 MiB |
| VRAM free before upload | 3410 MiB |
| Packed model size | 2 136 832 000 B = 2037.8 MiB |
| VRAM free after upload | **1372 MiB** |
| Upload time | 6.672 s (0.32 GB/s) |

The entire packed model is resident. 1372 MiB remains for KV cache and activations.

This is the configuration the 02:30 probe's transfer-cost finding demands. That probe
measured 9.16 ms of transfer against 0.30 ms of compute — moving data costs ~30× the
arithmetic. A design that streams weights per token would therefore lose to the CPU path
it replaces. Residency removes the per-token transfer entirely: the 6.672 s is paid once
at load.

## Verified — decode correctness

The kernel decodes blocks straight out of the resident copy, with no re-upload. Every
decoded value is compared **bit-for-bit** against a host reference that implements the
same documented rule independently.

| Tensor | Shape | Offset | Values | Kernel | Mismatches |
|---|---|---|---|---|---|
| `model.layers.12.self_attn.o_proj.weight` | 2560 × 4096 | 0 | 10 485 760 | 2.622 ms | **0** |
| `model.layers.4.self_attn.v_proj.weight` | 1024 × 2560 | 5 570 560 | 2 621 440 | 0.332 ms | **0** |
| `model.layers.35.mlp.down_proj.weight` | 2560 × 9728 | 2 123 601 920 | 24 903 680 | 2.514 ms | **0** |

**38 010 880 values checked, 0 mismatches.** The third tensor sits at the far end of the
2 GB resident buffer, which exercises the 64-bit offset arithmetic rather than just the
first megabyte.

### Full sweep — all 253 tensors, same run

The sample above was then replaced by complete coverage. The model is uploaded once, so
sweeping every tensor costs barely more than sweeping one.

| Measured | Result |
|---|---|
| Tensors decoded | **253 of 253** |
| Values checked bit-for-bit | **4 022 272 000** |
| Mismatches | **0** |
| Failed tensors | **0** |
| Total kernel time, whole model | **891.966 ms** |
| Mean rate | 4509.4 M values/s |
| Best tensor | 8949.0 M values/s (`model.layers.6.mlp.down_proj.weight`) |
| Wall clock, decode + compare | 17.796 s (upload excluded) |

Every weight in Qwen3-4B v4 decodes on the card to the same bits the CPU produces. The
entire model's weights expand in **0.89 s of GPU time**.

### The chunking constraint — found by hitting it

The first sweep attempt refused to start:

```text
largest tensor  : 197.1 MiB packed, 1483.8 MiB decoded as f32
RESIDENCY: NO -- 3410 MiB free, model plus output needs 3522 MiB.
```

The token embedding is 197.1 MiB packed and **1483.8 MiB expanded**, because f32 costs
7.5x the packed form. Model plus that one output buffer exceeds the card. This is not a
bug to work around, it is the shape of the problem: **a 4 GB card can hold the packed
model or a big expanded tensor, never both.**

The decoder therefore walks each tensor in chunks bounded by a 128 MiB output buffer
(262 144 blocks). Tensors under the cap take exactly one pass, so nothing else changed.
This is also what a fused decode-and-multiply kernel wants — it never needs the whole
expanded tensor either, which makes the constraint an argument for fusion rather than
against it.

A third, independent implementation (NumPy, using its own float16) decodes block 0 to
`0.005032` for value 0 — the same value the card returned. Agreement across three
separately written decoders, rather than one checked against itself.

## Verified — throughput

| Measured | Result |
|---|---|
| GPU decode, peak | **9905.7 M values/s** (`mlp.down_proj`) |
| GPU decode, first tensor | 3998.7 M values/s |
| CPU host reference | 138.2 M values/s |
| Speed-up over the scalar host reference | ~29× to ~72× |
| Result download | 0.064 s for 40 MiB (0.65 GB/s) |

The host reference is deliberately scalar and unoptimised; it is a correctness oracle,
not the AVX2 production path, so the ratio is not a claim about the real CPU runtime.

## Format confirmed against the model, not assumed

`wpc-format/src/lib.rs` documents v4 as 128 values per block in 68 bytes: `f16`
zero_point, `f16` scale, then 64 packed bytes where **byte j holds code j in the low
nibble and code j+64 in the high nibble** — deliberately not neighbours, so the SIMD path
needs no shuffle. The model file agrees exactly:

- 253 tensors, layer sizes summing to 2 136 832 000 B — the exact file size.
- First tensor: 10 485 760 values in 5 570 560 B = 81 920 blocks × 68 B = 128.0 values per
  block, **4.2500 bits per weight**.

Had the pairs been read as neighbours, the comparison would have failed rather than
producing plausible-looking weights. That is the point of comparing bit-for-bit.

## Reproduce

```
NVCC=/home/aions/cuda-local/rozpakowane/usr/lib/nvidia-cuda-toolkit/bin/nvcc
R=/home/aions/cuda-local/rozpakowane
export PATH=$R/usr/lib/nvidia-cuda-toolkit/bin:$R/usr/bin:$PATH

$NVCC -ccbin ./hostbin -I$R/usr/include -arch=sm_50 -ptx \
      wpc4_decode_sm50.cu -o wpc4_decode_sm50.ptx

gcc -O2 -std=c11 -I$R/usr/include run_wpc4_decode.c \
    -o run_wpc4_decode -L/usr/lib/wsl/lib -lcuda

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
./run_wpc4_decode <model_v4.wpc> wpc4_decode_sm50.ptx <tensor_name> <offset> <size>
```

Build notes that cost time and are worth recording:

- `/usr/bin/nvcc` in the relocated toolchain is a wrapper with a hardcoded absolute path
  to `/usr/lib/nvidia-cuda-toolkit/bin/nvcc`, which does not exist on this machine. Call
  the real binary under the relocated tree directly.
- `cicc` and `ptxas` are found through `PATH`, so both relocated `bin` directories must be
  on it, or nvcc fails with `cicc: not found`.
- `-ccbin` must point at a directory holding `gcc`/`g++` symlinks to GCC 12
  (`hostbin/`), not at the GCC 12 binary itself. The system compiler is GCC 13, which the
  CUDA 12.0 front end rejects.
- `-I` must reach `cuda_runtime.h`; nvcc includes it implicitly even for a `-ptx` build.
- The Driver API is used throughout and only `libcuda.so.1` from `/usr/lib/wsl/lib` is
  linked, because the relocated toolchain ships no `libcudart`.

## What this does NOT claim

- **Qwen3-4B inference is not running on the GPU.** Only the decode step is. Matmul,
  attention, sampling and the KV cache are all still CPU-side.
- Decode correctness is now complete: 253 of 253 tensors, 4 022 272 000 values, 0
  mismatches. That is coverage of the decode step only.
- Decode throughput was measured in isolation. It is not a token rate, and it does not
  account for reading activations or writing results in a real forward pass.

## Next step

1. ~~Sweep all 253 tensors in one process run.~~ **Done** — see the full sweep above.
2. Fuse decode with the matmul so decoded weights never round-trip to global memory. The
   present kernel writes f32 out, which a real forward pass would not do — and the
   embedding tensor proves it cannot, since its expanded form does not fit beside the
   model.
3. Only then measure tokens/s for a real generation, per step 4 of the original plan.

## Artifacts

- Working directory: `/home/aions/gpu-wpc4-decode-2026-08-25`
- Host binary SHA256: `68621d3e0731c0b77061c0019aa4c3430af582a7549e32448d4352b3835a040b`
- PTX SHA256: `f3321aae8c41df620053d21af67dbb7d8cdabf01b3723a2381c2b38cb933f3fd`
- Sweep binary SHA256: `15f9d62e5d577045a7e7c3d0506714cd344eeb2928af51cdbacce29bdec145b2`
- Run logs: `gpu/wpc4-decode/runs/run_2026-08-25_0500.log` (three tensors),
  `gpu/wpc4-decode/runs/sweep_2026-08-25_0522.log` (all 253)
- Tensor manifest: `gpu/wpc4-decode/tensors_all.txt`, generated from `model_v4.meta`
- Model inputs, unmodified: `/home/aions/qwen3-4b-wpc4/model_v4.wpc`, `model_v4.meta`

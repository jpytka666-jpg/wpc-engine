# Noworodek — Training Precision and Quadro M2000M Strategy

## Hardware target

The primary local target is the NVIDIA Quadro M2000M: 4 GB GDDR5, Maxwell generation, 640 CUDA cores, and roughly 80 GB/s memory bandwidth. It is a compute-capability 5.0/SM50 class device. NVIDIA's Maxwell documentation indicates that Maxwell 5.0/5.2 does not provide native FP16 arithmetic instructions; FP16 can still be used as a storage format, but training kernels should not assume Tensor Core or native half-precision matrix-multiply acceleration.

## Critical design decision

**Do not train the first model directly in 4-bit weights.**

Four-bit WPC is a storage/execution format for a later inference/backend experiment, not the numerical representation to use for the first learning algorithm. Training needs stable gradients and update paths; forcing the optimizer to update 4-bit quantized parameters from step one would confound the experiment and make optimization unnecessarily brittle.

Also, 128-bit parameters are not a useful target. The model does not need 128 bits per parameter for ordinary training.

## Initial numerical policy

### Teacher/Student research model

Start with:

- trainable master parameters: `f32` in the correctness reference path;
- optional `f16` storage where it reduces VRAM pressure and kernels remain numerically validated;
- gradients: `f32` accumulation initially;
- optimizer state: `f32` initially;
- checkpoints: exact `f32` reference plus optional compressed copies.

This gives us a trustworthy baseline before introducing numerical compression.

### Quantization roadmap

```text
FP32 reference training
        |
        +--> optional FP16 storage experiment
        |
        v
stable student model
        |
        +--> WPC 8-bit/6-bit/4-bit inference artifacts
        |
        +--> later low-bit training experiments
```

The first model should therefore be trained **above** the final WPC inference bit-width and only later converted into a WPC artifact.

## Full-GPU-residency rule

For the local deployment target, the execution path should keep the active model tensors and working state inside the 4 GB GPU whenever feasible. CPU/RAM may be used for dataset staging, checkpoint storage, and preparation, but the hot inference path must not silently page model tensors through host memory.

The architecture therefore separates:

```text
Host
  -> dataset / trace / checkpoint preparation

GPU
  -> active WeightSet tensors
  -> activations
  -> gradients (during training)
  -> KV state for inference

WPC / disk
  -> cold weight artifacts
```

A runtime check should fail or explicitly enter a documented fallback mode when the requested model/WeightSet does not fit in available VRAM. No silent host offload is allowed for the primary benchmark.

## Practical model scale on 4 GB

Parameter capacity depends on optimizer state, activations, sequence length, and checkpointing. Approximate raw parameter storage alone is:

- 30M params: ~120 MB at FP32, ~60 MB at FP16;
- 100M params: ~400 MB at FP32, ~200 MB at FP16;
- 300M params: ~1.2 GB at FP32, ~600 MB at FP16;
- 1B params: ~4.0 GB at FP32, ~2.0 GB at FP16.

Training requires substantially more memory than raw parameters because gradients, optimizer state, and activations must coexist. The first on-device training target is therefore intentionally small (tens of millions of parameters) and should be scaled only after measured VRAM headroom exists.

## WPC relationship

WPC is treated as a `WeightBackend`. The same external `WeightSet` manifest can have:

- `MemoryWeightBackend` for correctness/reference training;
- file/mmap backend for checkpoint experiments;
- WPC backend for compressed inference and later parameter-pattern experiments.

The student model must not contain WPC-specific conditionals in Transformer math. This keeps the training reference implementation independent from the compression experiment.

## Required benchmarks

Before claiming that the model runs "fully on GPU":

1. report total VRAM allocation;
2. report every active WeightSet allocation;
3. report activation/gradient/optimizer allocation during training;
4. report host-memory use by the hot path;
5. verify no host tensor fallback occurs during the benchmark;
6. verify model output equivalence between reference and optimized backends within stated tolerances.

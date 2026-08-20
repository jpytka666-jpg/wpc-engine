//! Library surface for wpc-eval: exposes the tested fused-SIMD WPC decode
//! kernel so other crates (e.g. wpc-runtime) can reuse it instead of
//! duplicating the AVX2 decode math.

pub mod fused_kernel;

pub mod config;
pub mod gemma4_config;
pub mod gemma4_model;
pub mod model;
pub mod norm;
pub mod qwen3_model;
pub mod qwen3_moe_model;
pub mod rope;
pub mod sampling;
pub mod weights;
pub mod wpc_weights;
pub mod wpc_weights_v2;
pub mod wpc_weights_v3;
pub mod wpc_weights_v4;

pub mod forward_batch;

// Historical CUDA Driver API bridge, recovered source, no model forward offload yet.
// Device discovery is explicit; never run GPU operations on module load.
#[cfg(any(target_os = "linux", windows))]
pub mod gpu;

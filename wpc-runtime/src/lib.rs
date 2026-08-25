pub mod config;
pub mod gemma4_config;
pub mod gemma4_model;
// The CUDA bridge finds the driver at run time -- dlopen on Linux, LoadLibrary on
// Windows -- so it builds on both and simply reports an error where there is no card.
#[cfg(any(target_os = "linux", windows))]
pub mod gpu;
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
pub mod wpc_weights_v5;

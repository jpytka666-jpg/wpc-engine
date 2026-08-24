# Real-model KV bridge

The Gemma4 runtime now exposes an opt-in `KvProbe` hook at the point where
resident K/V states are appended to `Gemma4KvCache`.

The hook is disabled by default and receives borrowed K/V slices only. It does
not mutate the runtime cache and has no model-file write path.

The existing WPC v3 model remains an external read-only input. The local model
artifact is never copied, rewritten, repacked, renamed, or deleted by this
feature.

Real-model validation still requires the original Gemma checkpoint metadata and
1D norm/scalar tensors that the existing `Gemma4Model::load_wpc_v3` path expects.
These are also treated as read-only inputs.

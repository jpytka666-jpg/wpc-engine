// =============================================================================
// wpc-format — zero-dependency data definitions for WPC.
// -----------------------------------------------------------------------------
// This crate is the *single source of truth* for the on-disk binary layout
// consumed by the C++ JIT SIMD decoder. Both the Rust encoder and the
// C++ engine read/write the exact same byte stream, so any layout change
// here is a contract change that must be mirrored in `wpc_inference_engine.cpp`.
//
// On-disk format (all multi-byte fields are little-endian, native IEEE 754):
//
//   CompressedBlock (6 bytes, packed):
//       offset 0 : u8   pattern_id   (0..255)
//       offset 1 : u16  residual_id  (0..65535)        LE
//       offset 3 : f16  base_value   (IEEE 754 half)   LE
//       offset 5 : i8   scale        (-127..127)
//
//   patterns.bin   : 256  * 16 * 4  = 16 384   bytes  (row-major f32 LE)
//   residuals.bin  : 65536* 16 * 2  = 2 097 152 bytes  (row-major f16 LE)
//   <name>.wpc     : N_BLOCKS * 6                   bytes  (CompressedBlock LE)
//
// The packed `#[repr(C, packed)]` attribute is critical: without it, the
// compiler is free to insert padding and silently break the C++ decoder.
// =============================================================================

#![no_std]

// We use `core::` for no_std compatibility. `half` re-exports f16 from
// its own crate, so we keep using that.
use half::f16;

/// The strict 6-byte packed block. Field order is the on-disk byte order.
///
/// This struct is `Copy` because it's `repr(C, packed)` and trivially
/// duplicable. It must NOT be passed by reference for fields other than
/// `pattern_id` (alignment 1) because packed fields can be unaligned.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressedBlock {
    pub pattern_id: u8,
    pub residual_id: u16,
    pub base_value: f16,
    pub scale: i8,
}

impl CompressedBlock {
    /// The on-disk byte size. C++ has an equivalent `static_assert`.
    pub const SIZE: usize = 6;

    /// Manual little-endian serialization. Avoids any alignment UB from
    /// `to_le_bytes()` on packed fields, and is endian-explicit so the
    /// file is portable across hosts.
    #[inline]
    pub fn to_le_bytes(self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        out[0] = self.pattern_id;
        out[1] = (self.residual_id & 0x00FF) as u8;
        out[2] = ((self.residual_id >> 8) & 0x00FF) as u8;
        let bv = self.base_value.to_le_bytes();
        out[3] = bv[0];
        out[4] = bv[1];
        out[5] = self.scale as u8;
        out
    }

    /// Parse a block from a 6-byte little-endian buffer. C++ reads the
    /// same buffer with a single `_mm_loadu_si128` and 4 manual field
    /// extractions; this is the Rust mirror.
    #[inline]
    pub fn from_le_bytes(b: &[u8; Self::SIZE]) -> Self {
        Self {
            pattern_id: b[0],
            residual_id: u16::from_le_bytes([b[1], b[2]]),
            base_value: f16::from_le_bytes([b[3], b[4]]),
            scale: b[5] as i8,
        }
    }
}

// Compile-time size check. If this ever changes, the binary contract
// changes and the C++ decoder MUST be updated. Stable Rust cannot directly
// assert on `core::mem::size_of` in const, but it CAN build an array of
// length N where N is the struct's size — that fails to compile if N != 6.
#[allow(dead_code)]
const _SIZE_CHECK: [(); 6] = [(); core::mem::size_of::<CompressedBlock>()];

// -----------------------------------------------------------------------------
// Dictionary dimensions — these are the C++ array sizes. Changing them
// requires updating `L1_patterns[256][16]` and `L2_residuals[65536][16]`
// in wpc_inference_engine.cpp.
// -----------------------------------------------------------------------------
pub const PATTERN_COUNT: usize = 256;
pub const RESIDUAL_COUNT: usize = 65536;
pub const BLOCK_SIZE: usize = 16;
pub const PATTERN_TABLE_BYTES: usize = PATTERN_COUNT * BLOCK_SIZE * 4; // 16 384
pub const RESIDUAL_TABLE_BYTES: usize = RESIDUAL_COUNT * BLOCK_SIZE * 2; // 2 097 152

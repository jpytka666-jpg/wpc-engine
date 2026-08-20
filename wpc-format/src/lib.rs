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
//       offset 1 : u16  residual_id  (0..RESIDUAL_COUNT-1, fits u16 while <=65536) LE
//       offset 3 : f16  base_value   (IEEE 754 half)   LE
//       offset 5 : i8   scale        (-127..127)
//
//   patterns.bin   : 256  * 16 * 4  = 16 384   bytes  (row-major f32 LE)
//   residuals.bin  : 8192 * 16 * 2  = 262 144  bytes  (row-major f16 LE)
//   <name>.wpc     : N_BLOCKS * 6                   bytes  (CompressedBlock LE)
//
//   RESIDUAL_COUNT was cut from 65536 to 8192: ResidualDict::train() only
//   ever subsampled 8192 points for k-means anyway (see wpc-core/codebook.rs),
//   so requesting k=65536 just padded the dictionary with exact duplicate
//   centroids. 8192 real centroids also makes ResidualDict::nearest()'s
//   linear scan -- the main per-block encode cost -- 8x cheaper.
//
// Scaling contract (INPUT_SCALE = 127.0, see wpc_core::encoder::INPUT_SCALE):
//   Values in patterns.bin are PRE-DIVIDED by INPUT_SCALE, so decode is a
//   single FMA with no further division: weight = pattern[pid] * scale + base.
//   Values in residuals.bin are stored RAW (not pre-divided); decode divides
//   them by INPUT_SCALE: weight += residual[rid] / INPUT_SCALE.
//   Both wpc-compiler (writer) and wpc-eval/fused_kernel (reader) must agree
//   on this; a mismatch here silently produces weights off by ~INPUT_SCALE.
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
// requires updating `L1_patterns[256][16]` and `L2_residuals[RESIDUAL_COUNT][16]`
// in wpc_inference_engine.cpp (not yet written -- wpc-engine/ is empty).
// -----------------------------------------------------------------------------
pub const PATTERN_COUNT: usize = 256;
pub const RESIDUAL_COUNT: usize = 8192;
pub const BLOCK_SIZE: usize = 16;
pub const PATTERN_TABLE_BYTES: usize = PATTERN_COUNT * BLOCK_SIZE * 4; // 16 384
pub const RESIDUAL_TABLE_BYTES: usize = RESIDUAL_COUNT * BLOCK_SIZE * 2; // 262 144

// =============================================================================
// v2 format: simple affine (min/max) quantization, no dictionary
// =============================================================================
// QuantBlockV2 is a simpler alternative to the VQ-codebook scheme above.
// Each block is fully self-describing: min/max are stored per-block, and
// 128 f32 values are quantized to 6-bit unsigned codes (0..63) via:
//   zero_point = min(block)
//   scale = (max(block) - min(block)) / 63.0
//   code[i] = round((value[i] - zero_point) / scale), clamped to [0, 63]
// Decode: value ≈ zero_point + code * scale.
//
// This scheme trades ~25% extra file size (full byte per 6-bit code) for
// simplicity, branch-free decode, and dramatically better reconstruction
// error on real transformer weights (measured ~2.5-2.7% RMSE vs 41-48% for VQ).

pub const BLOCK_SIZE_V2: usize = 128;

/// v2 on-disk block: plain per-block affine (min/max) 6-bit quantization.
/// No dictionary, no VQ, no residual -- each block is fully self-describing.
/// 6-bit codes are stored one per byte (values 0..63, top 2 bits unused)
/// to keep decode branch-free and byte-aligned.
///
/// offset layout (repr(C, packed)):
///   offset 0-1 : f16  zero_point  (LE, stores min(block))
///   offset 2-3 : f16  scale       (LE, stores (max-min)/63)
///   offset 4-131 : u8[128]  codes (values 0..63)
///
/// total size: 4 + 128 = 132 bytes per 128 values.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantBlockV2 {
    pub zero_point: f16,
    pub scale: f16,
    pub codes: [u8; 128],
}

impl QuantBlockV2 {
    pub const SIZE: usize = 132;

    pub fn to_le_bytes(&self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        let zp = self.zero_point.to_le_bytes();
        out[0] = zp[0];
        out[1] = zp[1];
        let sc = self.scale.to_le_bytes();
        out[2] = sc[0];
        out[3] = sc[1];
        out[4..132].copy_from_slice(&self.codes);
        out
    }

    pub fn from_le_bytes(b: &[u8; Self::SIZE]) -> Self {
        let zero_point = f16::from_le_bytes([b[0], b[1]]);
        let scale = f16::from_le_bytes([b[2], b[3]]);
        let mut codes = [0u8; 128];
        codes.copy_from_slice(&b[4..132]);
        QuantBlockV2 { zero_point, scale, codes }
    }
}

#[allow(dead_code)]
const _QUANT_BLOCK_V2_SIZE_CHECK: [(); 132] = [(); QuantBlockV2::SIZE];

// =============================================================================
// v3 format: v2's arithmetic, with the 6-bit codes actually packed
// =============================================================================
// v2 writes each 6-bit code into a whole byte, so two bits per weight reach
// the disk as padding -- 24% of every v2 file. That was a deliberate trade for
// branch-free decode, made before the bottleneck was known.
//
// It is known now. Generation on this workload is memory-bandwidth bound: one
// token costs a full pass over the weights, and the cores idle waiting for
// them (measured 190-350% CPU out of 800% available, ~10 GB/s sustained out of
// ~25 GB/s the memory can do). Bytes not read are time not spent, so that
// padding is not merely disk space -- it is roughly 24% of generation speed,
// given away.
//
// v3 keeps v2's quantization EXACTLY: same zero_point, same scale, same 0..63
// codes, therefore bit-identical reconstruction. Only the layout changes. Four
// consecutive codes are packed into three bytes:
//
//   byte 0 :  c0[5:0]        | c1[1:0] << 6
//   byte 1 : (c1[5:2] >> 2)  | c2[3:0] << 4
//   byte 2 : (c2[5:4] >> 4)  | c3[5:0] << 2
//
// 128 codes / 4 = 32 groups * 3 bytes = 96 bytes, plus the same 4-byte header:
// 100 bytes per 128 values = 6.25 bits/value against v2's 8.25.
//
// Unpacking costs a shift and a mask per code, paid on cores that were idle
// anyway -- spending the surplus to buy back the shortage.

pub const BLOCK_SIZE_V3: usize = 128;
/// 128 codes * 6 bits / 8 = 96 bytes of payload.
pub const PACKED_BYTES_V3: usize = BLOCK_SIZE_V3 * 6 / 8;

/// v3 on-disk block: v2's affine quantization with the codes bit-packed.
///
/// offset layout (repr(C, packed)):
///   offset 0-1  : f16  zero_point (LE, stores min(block))
///   offset 2-3  : f16  scale      (LE, stores (max-min)/63)
///   offset 4-99 : u8[96]  packed 6-bit codes, four per three bytes
///
/// total size: 4 + 96 = 100 bytes per 128 values.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantBlockV3 {
    pub zero_point: f16,
    pub scale: f16,
    pub packed: [u8; PACKED_BYTES_V3],
}

impl QuantBlockV3 {
    pub const SIZE: usize = 4 + PACKED_BYTES_V3;

    /// Pack 128 codes into 96 bytes. Values are masked to 6 bits, which
    /// matches the encoder's own clamp to 0..63 -- a code outside that range
    /// is an encoder bug, and silently truncating keeps decode total.
    #[inline]
    pub fn pack_codes(codes: &[u8; BLOCK_SIZE_V3]) -> [u8; PACKED_BYTES_V3] {
        let mut out = [0u8; PACKED_BYTES_V3];
        for g in 0..BLOCK_SIZE_V3 / 4 {
            let c0 = codes[g * 4] & 0x3F;
            let c1 = codes[g * 4 + 1] & 0x3F;
            let c2 = codes[g * 4 + 2] & 0x3F;
            let c3 = codes[g * 4 + 3] & 0x3F;
            out[g * 3] = c0 | (c1 << 6);
            out[g * 3 + 1] = (c1 >> 2) | (c2 << 4);
            out[g * 3 + 2] = (c2 >> 4) | (c3 << 2);
        }
        out
    }

    /// Exact inverse of `pack_codes`.
    #[inline]
    pub fn unpack_codes(packed: &[u8; PACKED_BYTES_V3]) -> [u8; BLOCK_SIZE_V3] {
        let mut codes = [0u8; BLOCK_SIZE_V3];
        for g in 0..BLOCK_SIZE_V3 / 4 {
            let b0 = packed[g * 3];
            let b1 = packed[g * 3 + 1];
            let b2 = packed[g * 3 + 2];
            codes[g * 4] = b0 & 0x3F;
            codes[g * 4 + 1] = (b0 >> 6) | ((b1 & 0x0F) << 2);
            codes[g * 4 + 2] = (b1 >> 4) | ((b2 & 0x03) << 4);
            codes[g * 4 + 3] = b2 >> 2;
        }
        codes
    }

    pub fn to_le_bytes(&self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        let zp = self.zero_point.to_le_bytes();
        out[0] = zp[0];
        out[1] = zp[1];
        let sc = self.scale.to_le_bytes();
        out[2] = sc[0];
        out[3] = sc[1];
        out[4..Self::SIZE].copy_from_slice(&self.packed);
        out
    }

    pub fn from_le_bytes(b: &[u8; Self::SIZE]) -> Self {
        let zero_point = f16::from_le_bytes([b[0], b[1]]);
        let scale = f16::from_le_bytes([b[2], b[3]]);
        let mut packed = [0u8; PACKED_BYTES_V3];
        packed.copy_from_slice(&b[4..Self::SIZE]);
        QuantBlockV3 { zero_point, scale, packed }
    }

    /// Rebuild a v3 block from a v2 one. The arithmetic is identical, so this
    /// is a pure re-layout: no requantization, no extra error.
    pub fn from_v2(v2: &QuantBlockV2) -> Self {
        let codes = v2.codes;
        QuantBlockV3 {
            zero_point: v2.zero_point,
            scale: v2.scale,
            packed: Self::pack_codes(&codes),
        }
    }
}

#[allow(dead_code)]
const _QUANT_BLOCK_V3_SIZE_CHECK: [(); 100] = [(); QuantBlockV3::SIZE];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_le_bytes() {
        let block = CompressedBlock {
            pattern_id: 200,
            residual_id: 54321,
            base_value: f16::from_f32(-3.25),
            scale: -100,
        };
        let bytes = block.to_le_bytes();
        assert_eq!(CompressedBlock::from_le_bytes(&bytes), block);
    }

    #[test]
    fn negative_scale_round_trips() {
        // scale is i8 in [-127, 127]; make sure the u8 bit-cast in
        // to_le_bytes/from_le_bytes doesn't corrupt negative values.
        for scale in [-127i8, -1, 0, 1, 127] {
            let block = CompressedBlock {
                pattern_id: 0,
                residual_id: 0,
                base_value: f16::from_f32(0.0),
                scale,
            };
            assert_eq!(CompressedBlock::from_le_bytes(&block.to_le_bytes()).scale, scale);
        }
    }

    #[test]
    fn v2_round_trips() {
        let mut codes = [0u8; 128];
        for i in 0..128 {
            codes[i] = (i % 64) as u8;
        }
        let block = QuantBlockV2 {
            zero_point: f16::from_f32(-1.5),
            scale: f16::from_f32(0.025),
            codes,
        };
        let bytes = block.to_le_bytes();
        assert_eq!(QuantBlockV2::from_le_bytes(&bytes), block);
    }
}

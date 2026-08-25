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

// =============================================================================
// v4 format: the same affine scheme at 4 bits, two codes per byte
// =============================================================================
// v3 costs 6.25 bits/weight. A 4B model is 2.93 GB that way, and a 4 GB card
// with ~2839 MiB actually free cannot take it -- it misses by about 100 MB.
// Four bits puts the same model at ~1918 MiB, which fits with room left for
// the KV cache.
//
// The quantization is v2/v3's, with one number changed: codes run 0..15 instead
// of 0..63, so scale = (max - min) / 15. Reconstruction is the same single FMA,
// `w = zero_point + code * scale`.
//
// The layout is chosen for the decoder rather than for the writer. Two codes
// share a byte, but NOT as neighbours: byte j holds code j in its low nibble
// and code j+64 in its high nibble. That is what makes the SIMD path free of
// shuffles -- `_mm256_cvtepu8_epi32` over 8 bytes yields 8 lanes, `& 0x0F`
// gives weights j..j+8 and `>> 4` gives weights j+64..j+72, and BOTH runs of
// activations are contiguous in `x`. Had the pairs been neighbours (j, j+1),
// the codes would come out even-and-odd and every load of `x` would need a
// deinterleave to match.
//
// 128 codes / 2 = 64 bytes, plus the same 4-byte header: 68 bytes per 128
// values = 4.25 bits/value, against v3's 6.25 and v2's 8.25.

pub const BLOCK_SIZE_V4: usize = 128;
/// 128 codes * 4 bits / 8 = 64 bytes of payload.
pub const PACKED_BYTES_V4: usize = BLOCK_SIZE_V4 * 4 / 8;
/// Highest 4-bit code. The encoder's `levels`, spelled once.
pub const MAX_CODE_V4: u8 = 15;

/// v4 on-disk block: affine 4-bit quantization, two codes per byte.
///
/// offset layout (repr(C, packed)):
///   offset 0-1  : f16  zero_point (LE, stores min(block))
///   offset 2-3  : f16  scale      (LE, stores (max-min)/15)
///   offset 4-67 : u8[64]  packed 4-bit codes; byte j is
///                 `codes[j] | (codes[j + 64] << 4)`
///
/// total size: 4 + 64 = 68 bytes per 128 values.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantBlockV4 {
    pub zero_point: f16,
    pub scale: f16,
    pub packed: [u8; PACKED_BYTES_V4],
}

impl QuantBlockV4 {
    pub const SIZE: usize = 4 + PACKED_BYTES_V4;

    /// Pack 128 codes into 64 bytes. Values are masked to 4 bits, matching the
    /// encoder's own clamp to 0..15 -- a code outside that range is an encoder
    /// bug, and truncating keeps decode total.
    #[inline]
    pub fn pack_codes(codes: &[u8; BLOCK_SIZE_V4]) -> [u8; PACKED_BYTES_V4] {
        let mut out = [0u8; PACKED_BYTES_V4];
        for j in 0..PACKED_BYTES_V4 {
            let lo = codes[j] & 0x0F;
            let hi = codes[j + PACKED_BYTES_V4] & 0x0F;
            out[j] = lo | (hi << 4);
        }
        out
    }

    /// Exact inverse of `pack_codes`.
    #[inline]
    pub fn unpack_codes(packed: &[u8; PACKED_BYTES_V4]) -> [u8; BLOCK_SIZE_V4] {
        let mut codes = [0u8; BLOCK_SIZE_V4];
        for j in 0..PACKED_BYTES_V4 {
            codes[j] = packed[j] & 0x0F;
            codes[j + PACKED_BYTES_V4] = packed[j] >> 4;
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
        let mut packed = [0u8; PACKED_BYTES_V4];
        packed.copy_from_slice(&b[4..Self::SIZE]);
        QuantBlockV4 { zero_point, scale, packed }
    }
}

#[allow(dead_code)]
const _QUANT_BLOCK_V4_SIZE_CHECK: [(); 68] = [(); QuantBlockV4::SIZE];

// =============================================================================
// v5 format: the same affine scheme at 2 bits, four codes per byte
// =============================================================================
// v4 answered "does 4 bits still work?" with yes -- the same tokens as v3, at
// 52% of the time. v5 asks the next question rather than assuming the answer:
// two bits is four levels per 128-weight block, and four levels is where an
// affine scheme is expected to stop describing a bell-shaped weight
// distribution at all. This exists to find out where that wall is, and the
// measurement is the point whether it passes or fails.
//
// The quantization is v2/v3/v4's with one number changed again: codes run 0..3,
// so scale = (max - min) / 3. Reconstruction is the same single FMA,
// `w = zero_point + code * scale`.
//
// The layout follows v4's reasoning one level down. Byte j holds four codes
// spaced 32 apart, NOT four neighbours:
//
//   bits 0-1 : code j        bits 4-5 : code j+64
//   bits 2-3 : code j+32     bits 6-7 : code j+96
//
// so `_mm256_cvtepu8_epi32` over 8 bytes yields, after `& 0x03` and shifts of
// 2, 4 and 6, four full YMM registers of codes -- and all four runs of
// activations (`x[j..j+8]`, `x[j+32..j+40]`, `x[j+64..]`, `x[j+96..]`) are
// contiguous. Neighbouring codes would come out interleaved 4-ways and every
// load of `x` would need a deinterleave.
//
// 128 codes / 4 = 32 bytes, plus the same 4-byte header: 36 bytes per 128
// values = 2.25 bits/value, against v4's 4.25 and v3's 6.25.

pub const BLOCK_SIZE_V5: usize = 128;
/// 128 codes * 2 bits / 8 = 32 bytes of payload.
pub const PACKED_BYTES_V5: usize = BLOCK_SIZE_V5 * 2 / 8;
/// Highest 2-bit code. The encoder's `levels`, spelled once.
pub const MAX_CODE_V5: u8 = 3;

/// v5 on-disk block: affine 2-bit quantization, four codes per byte.
///
/// offset layout (repr(C, packed)):
///   offset 0-1  : f16  zero_point (LE, stores min(block))
///   offset 2-3  : f16  scale      (LE, stores (max-min)/3)
///   offset 4-35 : u8[32]  packed 2-bit codes; byte j is
///                 `codes[j] | codes[j+32] << 2 | codes[j+64] << 4 | codes[j+96] << 6`
///
/// total size: 4 + 32 = 36 bytes per 128 values.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantBlockV5 {
    pub zero_point: f16,
    pub scale: f16,
    pub packed: [u8; PACKED_BYTES_V5],
}

impl QuantBlockV5 {
    pub const SIZE: usize = 4 + PACKED_BYTES_V5;

    /// Pack 128 codes into 32 bytes. Values are masked to 2 bits, matching the
    /// encoder's own clamp to 0..3 -- a code outside that range is an encoder
    /// bug, and truncating keeps decode total.
    #[inline]
    pub fn pack_codes(codes: &[u8; BLOCK_SIZE_V5]) -> [u8; PACKED_BYTES_V5] {
        let mut out = [0u8; PACKED_BYTES_V5];
        for j in 0..PACKED_BYTES_V5 {
            let c0 = codes[j] & 0x03;
            let c1 = codes[j + PACKED_BYTES_V5] & 0x03;
            let c2 = codes[j + 2 * PACKED_BYTES_V5] & 0x03;
            let c3 = codes[j + 3 * PACKED_BYTES_V5] & 0x03;
            out[j] = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
        }
        out
    }

    /// Exact inverse of `pack_codes`.
    #[inline]
    pub fn unpack_codes(packed: &[u8; PACKED_BYTES_V5]) -> [u8; BLOCK_SIZE_V5] {
        let mut codes = [0u8; BLOCK_SIZE_V5];
        for j in 0..PACKED_BYTES_V5 {
            let b = packed[j];
            codes[j] = b & 0x03;
            codes[j + PACKED_BYTES_V5] = (b >> 2) & 0x03;
            codes[j + 2 * PACKED_BYTES_V5] = (b >> 4) & 0x03;
            codes[j + 3 * PACKED_BYTES_V5] = b >> 6;
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
        let mut packed = [0u8; PACKED_BYTES_V5];
        packed.copy_from_slice(&b[4..Self::SIZE]);
        QuantBlockV5 { zero_point, scale, packed }
    }
}

#[allow(dead_code)]
const _QUANT_BLOCK_V5_SIZE_CHECK: [(); 36] = [(); QuantBlockV5::SIZE];

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

    #[test]
    fn v4_block_is_sixty_eight_bytes() {
        assert_eq!(QuantBlockV4::SIZE, 68);
        assert_eq!(core::mem::size_of::<QuantBlockV4>(), 68);
        // 68 bytes / 128 values = 4.25 bits per value.
        assert_eq!(PACKED_BYTES_V4, 64);
    }

    #[test]
    fn v4_round_trips_through_le_bytes() {
        let mut codes = [0u8; BLOCK_SIZE_V4];
        for i in 0..BLOCK_SIZE_V4 {
            codes[i] = (i % 16) as u8;
        }
        let block = QuantBlockV4 {
            zero_point: f16::from_f32(-1.5),
            scale: f16::from_f32(0.025),
            packed: QuantBlockV4::pack_codes(&codes),
        };
        let bytes = block.to_le_bytes();
        assert_eq!(QuantBlockV4::from_le_bytes(&bytes), block);
    }

    #[test]
    fn v4_packing_round_trips_every_code_in_every_nibble() {
        // Both nibbles of every byte, for every legal code value: a swapped
        // shift or a missing mask cannot hide in a half the test never uses.
        for v in 0u8..=MAX_CODE_V4 {
            let uniform = [v; BLOCK_SIZE_V4];
            let p = QuantBlockV4::pack_codes(&uniform);
            assert_eq!(QuantBlockV4::unpack_codes(&p), uniform, "code value {v} failed");
        }

        // And a pattern where the low half and the high half differ, so the
        // two nibbles cannot be confused for one another.
        let mut codes = [0u8; BLOCK_SIZE_V4];
        for i in 0..BLOCK_SIZE_V4 {
            codes[i] = ((i * 5 + i / 16) % 16) as u8;
        }
        let packed = QuantBlockV4::pack_codes(&codes);
        assert_eq!(QuantBlockV4::unpack_codes(&packed), codes);

        // The documented byte layout itself, not just the round trip.
        for j in 0..PACKED_BYTES_V4 {
            assert_eq!(packed[j] & 0x0F, codes[j]);
            assert_eq!(packed[j] >> 4, codes[j + PACKED_BYTES_V4]);
        }
    }

    #[test]
    fn v5_block_is_thirty_six_bytes() {
        assert_eq!(QuantBlockV5::SIZE, 36);
        assert_eq!(core::mem::size_of::<QuantBlockV5>(), 36);
        // 36 bytes / 128 values = 2.25 bits per value.
        assert_eq!(PACKED_BYTES_V5, 32);
    }

    #[test]
    fn v5_round_trips_through_le_bytes() {
        let mut codes = [0u8; BLOCK_SIZE_V5];
        for i in 0..BLOCK_SIZE_V5 {
            codes[i] = (i % 4) as u8;
        }
        let block = QuantBlockV5 {
            zero_point: f16::from_f32(-1.5),
            scale: f16::from_f32(0.025),
            packed: QuantBlockV5::pack_codes(&codes),
        };
        let bytes = block.to_le_bytes();
        assert_eq!(bytes.len(), 36);
        assert_eq!(QuantBlockV5::from_le_bytes(&bytes), block);
    }

    #[test]
    fn v5_packing_round_trips_every_code_in_every_slot() {
        // All four 2-bit slots of every byte, for every legal code value: a
        // shift off by two cannot hide in a slot the test never uses.
        for v in 0u8..=MAX_CODE_V5 {
            let uniform = [v; BLOCK_SIZE_V5];
            let p = QuantBlockV5::pack_codes(&uniform);
            assert_eq!(QuantBlockV5::unpack_codes(&p), uniform, "code value {v} failed");
        }

        // A pattern where all four quarters of the block differ, so no two
        // slots can be confused for one another.
        let mut codes = [0u8; BLOCK_SIZE_V5];
        for i in 0..BLOCK_SIZE_V5 {
            codes[i] = ((i * 5 + i / 32) % 4) as u8;
        }
        let packed = QuantBlockV5::pack_codes(&codes);
        assert_eq!(QuantBlockV5::unpack_codes(&packed), codes);

        // The documented byte layout itself, not just the round trip.
        for j in 0..PACKED_BYTES_V5 {
            assert_eq!(packed[j] & 0x03, codes[j]);
            assert_eq!((packed[j] >> 2) & 0x03, codes[j + PACKED_BYTES_V5]);
            assert_eq!((packed[j] >> 4) & 0x03, codes[j + 2 * PACKED_BYTES_V5]);
            assert_eq!(packed[j] >> 6, codes[j + 3 * PACKED_BYTES_V5]);
        }
    }

    /// Every one of the 256 possible payload bytes decoded and re-encoded. With
    /// only four values per slot the whole space is small enough to check
    /// exhaustively, so nothing is left to a chosen pattern.
    #[test]
    fn v5_every_possible_byte_survives_pack_unpack() {
        for b in 0u16..=255 {
            let packed = [b as u8; PACKED_BYTES_V5];
            let codes = QuantBlockV5::unpack_codes(&packed);
            for &c in codes.iter() {
                assert!(c <= MAX_CODE_V5, "byte {b} unpacked to out-of-range code {c}");
            }
            assert_eq!(QuantBlockV5::pack_codes(&codes), packed, "byte {b} failed");
        }
    }
}

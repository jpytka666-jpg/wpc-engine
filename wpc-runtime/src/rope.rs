//! Rotary position embeddings, HF `rotate_half` convention (non-interleaved
//! half-split), matching `Qwen2RotaryEmbedding` / `apply_rotary_pos_emb`.
//!
//! For a head vector `x` of length `d` (even), split into `x1 = x[0..d/2]`,
//! `x2 = x[d/2..d]`. For each `i` in `0..d/2`:
//!   angle_i   = pos * theta^(-2i/d)
//!   out[i]      = x1[i] * cos(angle_i) - x2[i] * sin(angle_i)
//!   out[i+d/2]  = x2[i] * cos(angle_i) + x1[i] * sin(angle_i)

pub fn apply_rope(x: &mut [f32], pos: usize, theta: f64) {
    let half = x.len() / 2;
    apply_rope_partial(x, pos, theta, half);
}

/// Gemma4 "proportional" partial RoPE: identical `rotate_half` math to
/// [`apply_rope`], but only the first `rotary_pairs` of the `d/2` frequency
/// pairs actually rotate (`inv_freq` is conceptually zero for the rest, i.e.
/// those dims pass through unchanged) — matches HF's
/// `_compute_proportional_rope_parameters`, which computes real frequencies
/// for `rope_angles = floor(partial_rotary_factor * head_dim / 2)` pairs and
/// zero-frequency (identity) for the remainder, over the *full* head_dim
/// (not a truncated prefix of the vector). `rotary_pairs == d/2` reproduces
/// full rotation, i.e. plain [`apply_rope`].
pub fn apply_rope_partial(x: &mut [f32], pos: usize, theta: f64, rotary_pairs: usize) {
    let d = x.len();
    debug_assert_eq!(d % 2, 0);
    let half = d / 2;
    debug_assert!(rotary_pairs <= half);
    for i in 0..rotary_pairs {
        let inv_freq = theta.powf(-(2.0 * i as f64) / d as f64);
        let angle = pos as f64 * inv_freq;
        let (s, c) = angle.sin_cos();
        let (s, c) = (s as f32, c as f32);
        let x1 = x[i];
        let x2 = x[i + half];
        x[i] = x1 * c - x2 * s;
        x[i + half] = x2 * c + x1 * s;
    }
    // Pairs [rotary_pairs, half) have zero rotation angle => identity, left untouched.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_at_position_zero_is_identity() {
        let mut x = [1.0f32, 2.0, 3.0, 4.0];
        let orig = x;
        apply_rope(&mut x, 0, 1000000.0);
        // angle = 0 for every freq at pos=0 -> cos=1, sin=0 -> identity
        for i in 0..4 {
            assert!((x[i] - orig[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_preserves_per_pair_norm() {
        // Rotation is norm-preserving on each (x1[i], x2[i]) pair.
        let mut x = [0.5f32, -1.3, 2.7, 0.1];
        let half = x.len() / 2;
        let norms_before: Vec<f32> = (0..half)
            .map(|i| (x[i] * x[i] + x[i + half] * x[i + half]).sqrt())
            .collect();
        apply_rope(&mut x, 5, 1000000.0);
        let norms_after: Vec<f32> = (0..half)
            .map(|i| (x[i] * x[i] + x[i + half] * x[i + half]).sqrt())
            .collect();
        for i in 0..half {
            assert!((norms_before[i] - norms_after[i]).abs() < 1e-4);
        }
    }
}

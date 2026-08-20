//! RMSNorm, exactly matching HF's `Qwen2RMSNorm`:
//! `variance = mean(x^2)`; `x = x * rsqrt(variance + eps)`; `out = weight * x`.

pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), out.len());
    let n = x.len() as f32;
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * scale * weight[i];
    }
}

/// RMSNorm without a learned weight (Gemma4's `v_norm`, `with_scale=False`):
/// `out = x * rsqrt(mean(x^2) + eps)`.
pub fn rms_norm_no_weight(x: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    let n = x.len() as f32;
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * scale;
    }
}

/// Numerically-stable softmax over `x`, written in place.
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_unit_weight_normalizes_rms_to_one() {
        let x = [3.0f32, 4.0, 0.0, 0.0];
        let w = [1.0f32; 4];
        let mut out = [0.0f32; 4];
        rms_norm(&x, &w, 1e-6, &mut out);
        // mean_sq = (9+16)/4 = 6.25, rms = 2.5
        let expected = [3.0 / 2.5, 4.0 / 2.5, 0.0, 0.0];
        for i in 0..4 {
            assert!((out[i] - expected[i]).abs() < 1e-4, "idx {i}: {} vs {}", out[i], expected[i]);
        }
    }

    #[test]
    fn rms_norm_applies_weight_scaling() {
        let x = [1.0f32, 1.0, 1.0, 1.0];
        let w = [2.0f32, 0.5, 1.0, 1.0];
        let mut out = [0.0f32; 4];
        rms_norm(&x, &w, 1e-6, &mut out);
        // mean_sq = 1, rms = 1, so out = x * w
        assert!((out[0] - 2.0).abs() < 1e-5);
        assert!((out[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn softmax_sums_to_one_and_is_monotonic() {
        let mut x = [1.0f32, 2.0, 3.0];
        softmax_inplace(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(x[0] < x[1] && x[1] < x[2]);
    }

    #[test]
    fn softmax_uniform_for_equal_inputs() {
        let mut x = [5.0f32, 5.0, 5.0, 5.0];
        softmax_inplace(&mut x);
        for v in x.iter() {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }
}

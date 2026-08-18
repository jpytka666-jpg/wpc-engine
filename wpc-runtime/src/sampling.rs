/// Greedy argmax over logits. Sufficient for correctness testing against
/// the HF greedy-decode reference.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_the_largest() {
        let logits = [0.1f32, 5.0, -2.0, 4.9];
        assert_eq!(argmax(&logits), 1);
    }
}

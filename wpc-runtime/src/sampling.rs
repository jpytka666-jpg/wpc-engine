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

/// Token ids that must never be emitted, read from the `WPC_BAN_TOKENS`
/// environment variable as a comma-separated list.
///
/// Gemma4 is a multimodal checkpoint: its vocabulary carries image, audio and
/// video markers (258882 is `<image|>`) next to thousands of `<unusedN>` slots.
/// Text-only greedy decoding has no reason to emit any of them, but nothing in
/// the logits prevents it. Returning an empty list when the variable is unset
/// keeps the default decode path byte-identical to the previous behaviour.
pub fn banned_from_env() -> Vec<u32> {
    match std::env::var("WPC_BAN_TOKENS") {
        Ok(s) => s
            .split(',')
            .filter_map(|p| p.trim().parse::<u32>().ok())
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Greedy argmax that skips `banned` ids. With an empty `banned` slice this is
/// exactly `argmax`, so callers can pass the env-derived list unconditionally.
pub fn argmax_banned(logits: &[f32], banned: &[u32]) -> u32 {
    if banned.is_empty() {
        return argmax(logits);
    }
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if banned.contains(&(i as u32)) {
            continue;
        }
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

    #[test]
    fn empty_ban_list_matches_plain_argmax() {
        let logits = [0.1f32, 5.0, -2.0, 4.9];
        assert_eq!(argmax_banned(&logits, &[]), argmax(&logits));
    }

    #[test]
    fn banned_ids_are_skipped() {
        let logits = [0.1f32, 5.0, -2.0, 4.9];
        assert_eq!(argmax_banned(&logits, &[1]), 3);
        assert_eq!(argmax_banned(&logits, &[1, 3]), 0);
    }
}

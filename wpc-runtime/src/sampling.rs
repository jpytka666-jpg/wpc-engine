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

/// Why greedy decoding alone is not enough.
///
/// `argmax` always takes the single most likely continuation. That sounds like
/// the safest possible choice and it is exactly what makes long answers jam:
/// once "programming" is the most likely word after "programming", nothing can
/// ever break the tie differently, and the model repeats it until it runs out
/// of room. Observed on Qwen3-4B, Qwen3-Coder-30B and Gemma alike, which is the
/// tell that it is the decode path and not any one model.
///
/// Two independent cures, both off by default in their neutral setting so the
/// old byte-identical greedy path is still reachable:
///
/// * a repetition penalty, which pushes down the score of anything already said
///   inside a recent window; and
/// * temperature with nucleus (top-p) sampling, which lets the second-best word
///   win sometimes instead of never.
///
/// The scratch buffer is kept across calls because the vocabulary is ~152k
/// entries and reallocating that per token is pure waste.
pub struct Decoder {
    pub repeat_penalty: f32,
    pub repeat_window: usize,
    pub temperature: f32,
    pub top_p: f32,
    rng: u64,
    scratch: Vec<f32>,
    order: Vec<u32>,
}

impl Decoder {
    /// `repeat_penalty` of 1.0, `temperature` of 0.0 and any `top_p` reproduce
    /// plain `argmax_banned` exactly.
    pub fn new(repeat_penalty: f32, repeat_window: usize, temperature: f32, top_p: f32, seed: u64) -> Self {
        Self {
            repeat_penalty,
            repeat_window,
            temperature,
            top_p,
            // A zero state would make xorshift emit zero forever.
            rng: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
            scratch: Vec::new(),
            order: Vec::new(),
        }
    }

    /// True when nothing would be changed and the caller could use `argmax_banned`.
    pub fn is_plain_greedy(&self) -> bool {
        self.repeat_penalty == 1.0 && self.temperature <= 0.0
    }

    fn next_rand(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Choose the next token id.
    ///
    /// `history` is what the model has ALREADY WRITTEN -- not the prompt.
    ///
    /// That distinction is the whole correctness of this function. Feeding it the prompt
    /// as well pushes down exactly the words the person just used, so a model asked to
    /// repeat a figure back cannot reach it. Observed on Qwen3-Coder-30B: given "4096 MB"
    /// and asked to repeat it, it answered "4０９６ MB" -- the right digits in
    /// full-width characters, because the ordinary ones had been penalised out of its
    /// way. The same effect turned "M2000M" into "m2oOa" on the 4B and was blamed on the
    /// model for most of a day.
    ///
    /// Only the tail of `repeat_window` entries is penalised, so an early mention does
    /// not silence a word forever.
    pub fn pick(&mut self, logits: &[f32], banned: &[u32], history: &[u32]) -> u32 {
        if self.is_plain_greedy() && banned.is_empty() {
            return argmax(logits);
        }

        self.scratch.clear();
        self.scratch.extend_from_slice(logits);

        // Penalise what was said recently. Dividing a positive score and
        // multiplying a negative one both move the value toward -inf, which is
        // the formulation the CTRL paper uses and the one every runtime copied.
        if self.repeat_penalty != 1.0 && self.repeat_window > 0 {
            let start = history.len().saturating_sub(self.repeat_window);
            for &id in &history[start..] {
                if let Some(v) = self.scratch.get_mut(id as usize) {
                    *v = if *v > 0.0 { *v / self.repeat_penalty } else { *v * self.repeat_penalty };
                }
            }
        }

        for &id in banned {
            if let Some(v) = self.scratch.get_mut(id as usize) {
                *v = f32::NEG_INFINITY;
            }
        }

        if self.temperature <= 0.0 {
            return argmax(&self.scratch);
        }

        // Softmax at the requested temperature, shifted by the maximum so the
        // exponentials cannot overflow.
        let inv_t = 1.0 / self.temperature;
        let mut max_v = f32::NEG_INFINITY;
        for v in self.scratch.iter_mut() {
            *v *= inv_t;
            if *v > max_v {
                max_v = *v;
            }
        }
        let mut total = 0.0f32;
        for v in self.scratch.iter_mut() {
            *v = (*v - max_v).exp();
            total += *v;
        }
        if !(total > 0.0) {
            return argmax(logits);
        }

        // Nucleus: keep the smallest set of candidates whose probability adds up
        // to `top_p`, and draw from that set alone. The long tail of a 152k
        // vocabulary is where the nonsense lives.
        self.order.clear();
        self.order.extend(0..self.scratch.len() as u32);
        let scratch = &self.scratch;
        self.order.sort_unstable_by(|&a, &b| {
            scratch[b as usize]
                .partial_cmp(&scratch[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let cutoff = total * self.top_p.clamp(0.0, 1.0);
        let mut kept = 0.0f32;
        let mut last = 0usize;
        for (i, &id) in self.order.iter().enumerate() {
            kept += self.scratch[id as usize];
            last = i;
            if kept >= cutoff {
                break;
            }
        }

        let draw = (self.next_rand() >> 11) as f32 / (1u64 << 53) as f32 * kept;
        let mut acc = 0.0f32;
        for &id in &self.order[..=last] {
            acc += self.scratch[id as usize];
            if acc >= draw {
                return id;
            }
        }
        self.order[last]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_with_neutral_settings_is_plain_greedy() {
        let logits = [0.1f32, 5.0, -2.0, 4.9];
        let mut d = Decoder::new(1.0, 64, 0.0, 0.95, 1);
        assert!(d.is_plain_greedy());
        assert_eq!(d.pick(&logits, &[], &[]), argmax(&logits));
    }

    #[test]
    fn repetition_penalty_demotes_what_was_just_said() {
        let logits = [0.1f32, 5.0, -2.0, 4.9];
        let mut d = Decoder::new(2.0, 64, 0.0, 0.95, 1);
        // 1 is the winner until it has just been used; then 3 takes over.
        assert_eq!(d.pick(&logits, &[], &[]), 1);
        assert_eq!(d.pick(&logits, &[], &[1]), 3);
    }

    #[test]
    fn penalty_window_forgets_old_mentions() {
        let logits = [0.1f32, 5.0, -2.0, 4.9];
        let mut d = Decoder::new(2.0, 1, 0.0, 0.95, 1);
        // Only the last entry is inside the window, so token 1 is untouched.
        assert_eq!(d.pick(&logits, &[], &[1, 0]), 1);
    }

    #[test]
    fn banned_ids_are_never_drawn() {
        let logits = [0.1f32, 5.0, -2.0, 4.9];
        let mut d = Decoder::new(1.1, 64, 1.0, 1.0, 7);
        for _ in 0..200 {
            assert_ne!(d.pick(&logits, &[1], &[]), 1);
        }
    }

    #[test]
    fn sampling_stays_inside_the_nucleus() {
        // 0 carries almost all the mass; a tight top_p must never reach 2.
        let logits = [10.0f32, 1.0, -20.0];
        let mut d = Decoder::new(1.0, 64, 1.0, 0.9, 3);
        for _ in 0..200 {
            assert_ne!(d.pick(&logits, &[], &[]), 2);
        }
    }

    #[test]
    fn same_seed_gives_the_same_sequence() {
        let logits = [1.0f32, 1.1, 0.9, 1.05];
        let mut a = Decoder::new(1.0, 64, 1.0, 0.99, 42);
        let mut b = Decoder::new(1.0, 64, 1.0, 0.99, 42);
        for _ in 0..50 {
            assert_eq!(a.pick(&logits, &[], &[]), b.pick(&logits, &[], &[]));
        }
    }

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

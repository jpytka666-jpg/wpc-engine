// ==========================================
// AUTHOR: M. SZUL
// AI MODEL: Claude Opus 5
// TIMESTAMP: 2026-08-26 11:45:13
// REASON FOR CREATION: Fixed-width packing spends the same eleven bits on the 1367th
//   `status` as on a hash seen once. Measured on the AIONS corpus the id stream carries
//   5.98 bits of entropy per id, so 46% of the packed size is width nobody needs. This is
//   M. Szul's repetition-pricing idea applied where it belongs: not suppressing repeats,
//   which would corrupt a lossless encoding, but charging less for what is common.
// MECHANICS: Canonical Huffman over id frequencies. Code lengths are derived once, offline,
//   from a corpus and travel WITH THE BOOK rather than in every message - sender and
//   receiver already share the book, so a message that carried its own table would be
//   paying twice for something both sides have. Decoding needs only the lengths, which is
//   what makes the code canonical.
// SYSTEM PART: cbms-writing - the CBMS writing system
// ARCHITECTURE FUNCTION: The transport layer under vocab. Above it everything is ids;
//   below it ids become a bitstream priced by how ordinary they are.
// DEPENDENCIES/LINKS: consumes id frequencies produced by vocab::Vocabulary::encode
// TECH STACK: Rust 2021, standard library only.
// LOCAL WORKSPACE: C:\temp\aions-cbms-2026-08-26\cbms-writing
// GIT COMMIT: PENDING
// GITHUB METADATA: jpytka666-jpg/wpc-engine, branch feature/cbms-writing
// ==========================================

//! Frequency-weighted coding of the id stream: common ids cost fewer bits.

use std::collections::HashMap;

/// A canonical Huffman code, held as one length per id.
///
/// Lengths are all a decoder needs, which is the point of the canonical form: the codes
/// themselves are reconstructed from the lengths in a fixed order, so the table that
/// ships with the book is one byte per id rather than a code per id.
#[derive(Debug, Clone, Default)]
pub struct Code {
    lengths: Vec<u8>,
    codes: Vec<u32>,
    /// Ids sorted by (length, id) - the canonical order decoding walks.
    order: Vec<u16>,
    /// first_code[len] and first_index[len], indexed by code length.
    first_code: Vec<u32>,
    first_index: Vec<usize>,
    max_len: usize,
}

impl Code {
    /// Build from id frequencies. Ids never seen get no code, which is correct: a code
    /// for something that does not occur is bits spent on nothing.
    pub fn from_frequencies(freq: &HashMap<u16, usize>, vocab_size: usize) -> Code {
        let mut lengths = vec![0u8; vocab_size];
        let present: Vec<(u16, usize)> =
            freq.iter().filter(|(_, &c)| c > 0).map(|(&s, &c)| (s, c)).collect();

        match present.len() {
            0 => return Code::default(),
            // One symbol still needs a bit, or the stream has no length.
            1 => lengths[present[0].0 as usize] = 1,
            _ => {
                for (sym, len) in huffman_lengths(&present) {
                    lengths[sym as usize] = len;
                }
            }
        }
        Code::from_lengths(lengths)
    }

    /// Rebuild from lengths alone - what a receiver does with the table from the book.
    pub fn from_lengths(lengths: Vec<u8>) -> Code {
        let max_len = lengths.iter().copied().max().unwrap_or(0) as usize;
        if max_len == 0 {
            return Code::default();
        }
        let mut order: Vec<u16> = (0..lengths.len() as u16)
            .filter(|&s| lengths[s as usize] > 0)
            .collect();
        // Sort by length then id. Both sides must agree on this exactly.
        order.sort_by_key(|&s| (lengths[s as usize], s));

        let mut count = vec![0usize; max_len + 2];
        for &s in &order {
            count[lengths[s as usize] as usize] += 1;
        }

        let mut first_code = vec![0u32; max_len + 2];
        let mut first_index = vec![0usize; max_len + 2];
        let mut code = 0u32;
        let mut index = 0usize;
        for len in 1..=max_len {
            code = (code + count[len - 1] as u32) << 1;
            first_code[len] = code;
            first_index[len] = index;
            index += count[len];
        }

        let mut codes = vec![0u32; lengths.len()];
        let mut next = first_code.clone();
        for &s in &order {
            let len = lengths[s as usize] as usize;
            codes[s as usize] = next[len];
            next[len] += 1;
        }

        Code { lengths, codes, order, first_code, first_index, max_len }
    }

    pub fn lengths(&self) -> &[u8] {
        &self.lengths
    }

    pub fn is_empty(&self) -> bool {
        self.max_len == 0
    }

    /// Bits this code would spend on a stream, without building it.
    pub fn bit_cost(&self, ids: &[u16]) -> u64 {
        ids.iter()
            .map(|&id| *self.lengths.get(id as usize).unwrap_or(&0) as u64)
            .sum()
    }

    /// Does this table have a code for every id in the stream?
    ///
    /// A table built from one corpus has no code for an id that corpus never used, and
    /// encoding such a stream would drop those ids. The caller has to know BEFORE it
    /// writes a header claiming a count it cannot deliver.
    pub fn covers(&self, ids: &[u16]) -> bool {
        ids.iter()
            .all(|&id| self.lengths.get(id as usize).copied().unwrap_or(0) > 0)
    }

    /// Encode, or refuse. Silently skipping an id without a code is what made a stored
    /// chunk come back with its last word missing: the header promised more ids than
    /// the payload held, and decoding ran off the end.
    pub fn try_encode(&self, ids: &[u16]) -> Option<Vec<u8>> {
        if self.covers(ids) {
            Some(self.encode(ids))
        } else {
            None
        }
    }

    /// Ids with no code are skipped. Call `covers` first, or use `try_encode`.
    pub fn encode(&self, ids: &[u16]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut acc: u64 = 0;
        let mut bits = 0u32;
        for &id in ids {
            let len = self.lengths.get(id as usize).copied().unwrap_or(0) as u32;
            if len == 0 {
                continue; // no code: this id never appeared when the table was built
            }
            // Most significant bit first, so the decoder can walk lengths upward.
            acc = (acc << len) | self.codes[id as usize] as u64;
            bits += len;
            while bits >= 8 {
                out.push(((acc >> (bits - 8)) & 0xFF) as u8);
                bits -= 8;
            }
        }
        if bits > 0 {
            out.push(((acc << (8 - bits)) & 0xFF) as u8);
        }
        out
    }

    /// `count` is carried alongside the stream; a bitstream has no natural end.
    pub fn decode(&self, bytes: &[u8], count: usize) -> Vec<u16> {
        let mut out = Vec::with_capacity(count);
        if self.max_len == 0 {
            return out;
        }
        let mut code = 0u32;
        let mut len = 0usize;
        let mut bit_pos = 0usize;
        let total_bits = bytes.len() * 8;

        while out.len() < count && bit_pos < total_bits {
            let byte = bytes[bit_pos / 8];
            let bit = (byte >> (7 - (bit_pos % 8))) & 1;
            bit_pos += 1;
            code = (code << 1) | bit as u32;
            len += 1;
            if len > self.max_len {
                break; // corrupt stream rather than a silent wrong answer
            }
            let count_at_len = if len + 1 <= self.max_len {
                self.first_index[len + 1] - self.first_index[len]
            } else {
                self.order.len() - self.first_index[len]
            };
            if count_at_len > 0 {
                let offset = code.wrapping_sub(self.first_code[len]);
                if (offset as usize) < count_at_len {
                    out.push(self.order[self.first_index[len] + offset as usize]);
                    code = 0;
                    len = 0;
                }
            }
        }
        out
    }
}

/// Standard Huffman by repeated smallest-pair merge, returning code lengths only.
fn huffman_lengths(present: &[(u16, usize)]) -> Vec<(u16, u8)> {
    #[derive(Clone)]
    struct Node {
        weight: usize,
        symbol: Option<u16>,
        left: usize,
        right: usize,
    }
    let mut nodes: Vec<Node> = present
        .iter()
        .map(|&(s, w)| Node { weight: w, symbol: Some(s), left: usize::MAX, right: usize::MAX })
        .collect();
    // Deterministic: ties break on symbol, so the same corpus yields the same table.
    let mut live: Vec<usize> = (0..nodes.len()).collect();
    live.sort_by_key(|&i| (nodes[i].weight, nodes[i].symbol));

    while live.len() > 1 {
        let a = live.remove(0);
        let b = live.remove(0);
        let node = Node {
            weight: nodes[a].weight + nodes[b].weight,
            symbol: None,
            left: a,
            right: b,
        };
        nodes.push(node);
        let idx = nodes.len() - 1;
        let w = nodes[idx].weight;
        // Insert keeping the order the sort established.
        let pos = live
            .iter()
            .position(|&i| nodes[i].weight > w)
            .unwrap_or(live.len());
        live.insert(pos, idx);
    }

    let mut out = Vec::with_capacity(present.len());
    let mut stack = vec![(live[0], 0u8)];
    while let Some((idx, depth)) = stack.pop() {
        if let Some(sym) = nodes[idx].symbol {
            out.push((sym, depth.max(1)));
        } else {
            stack.push((nodes[idx].left, depth + 1));
            stack.push((nodes[idx].right, depth + 1));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn freq(pairs: &[(u16, usize)]) -> HashMap<u16, usize> {
        pairs.iter().copied().collect()
    }

    #[test]
    fn a_common_id_gets_a_shorter_code_than_a_rare_one() {
        // The whole idea: what is ordinary should be cheap.
        let f = freq(&[(10, 1000), (11, 100), (12, 10), (13, 1)]);
        let code = Code::from_frequencies(&f, 32);
        assert!(
            code.lengths()[10] < code.lengths()[13],
            "common {} vs rare {}",
            code.lengths()[10],
            code.lengths()[13]
        );
    }

    #[test]
    fn round_trips_a_stream() {
        let ids: Vec<u16> = "aaaaabbbcd".bytes().map(|b| b as u16).collect();
        let mut f = HashMap::new();
        for &id in &ids {
            *f.entry(id).or_default() += 1;
        }
        let code = Code::from_frequencies(&f, 256);
        let bytes = code.encode(&ids);
        assert_eq!(code.decode(&bytes, ids.len()), ids);
    }

    #[test]
    fn a_receiver_rebuilds_the_same_code_from_lengths_alone() {
        // This is why the table shipped with the book is one byte per id.
        let f = freq(&[(1, 50), (2, 20), (3, 20), (4, 5), (5, 5)]);
        let sender = Code::from_frequencies(&f, 16);
        let receiver = Code::from_lengths(sender.lengths().to_vec());
        let ids = vec![1u16, 2, 1, 3, 4, 5, 1, 1];
        let bytes = sender.encode(&ids);
        assert_eq!(receiver.decode(&bytes, ids.len()), ids);
    }

    #[test]
    fn the_same_frequencies_always_produce_the_same_table() {
        let f = freq(&[(1, 5), (2, 5), (3, 5), (4, 5), (5, 1)]);
        let a = Code::from_frequencies(&f, 16);
        let b = Code::from_frequencies(&f, 16);
        assert_eq!(a.lengths(), b.lengths(), "ties must not depend on hash order");
    }

    #[test]
    fn a_single_symbol_still_costs_a_bit() {
        let f = freq(&[(7, 100)]);
        let code = Code::from_frequencies(&f, 16);
        assert_eq!(code.lengths()[7], 1);
        let ids = vec![7u16; 5];
        assert_eq!(code.decode(&code.encode(&ids), 5), ids);
    }

    #[test]
    fn it_spends_fewer_bits_than_fixed_width_on_a_skewed_stream() {
        let mut ids = vec![1u16; 900];
        ids.extend(vec![2u16; 90]);
        ids.extend(vec![3u16; 10]);
        let mut f = HashMap::new();
        for &id in &ids {
            *f.entry(id).or_default() += 1;
        }
        let code = Code::from_frequencies(&f, 2048);
        // Fixed width for a 2048-id vocabulary is 11 bits each.
        assert!(
            code.bit_cost(&ids) < ids.len() as u64 * 11,
            "{} bits against {}",
            code.bit_cost(&ids),
            ids.len() * 11
        );
    }

    #[test]
    fn an_empty_stream_is_handled_rather_than_panicking() {
        let code = Code::from_frequencies(&HashMap::new(), 16);
        assert!(code.is_empty());
        assert!(code.encode(&[]).is_empty());
        assert!(code.decode(&[], 0).is_empty());
    }
}

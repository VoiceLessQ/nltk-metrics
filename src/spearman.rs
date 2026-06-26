//! Spearman rank correlation, ported from `nltk.metrics.spearman`.
//!
//! Rankings are sequences of `(key, rank)` pairs. The correlation is computed only
//! over keys present in both rankings.

use std::collections::HashMap;
use std::hash::Hash;

/// Spearman correlation coefficient for two rankings, in `[-1.0, 1.0]`.
///
/// Only keys present in both rankings contribute. Returns `0.0` when fewer than two
/// keys overlap (NLTK's division-by-zero fallback). Duplicate keys keep the last
/// rank, matching Python's `dict(seq)`.
pub fn spearman_correlation<K: Eq + Hash + Clone>(ranks1: &[(K, f64)], ranks2: &[(K, f64)]) -> f64 {
    let map1: HashMap<K, f64> = ranks1.iter().cloned().collect();
    let map2: HashMap<K, f64> = ranks2.iter().cloned().collect();

    let mut res = 0.0;
    let mut n: i64 = 0;
    for (k, &v1) in &map1 {
        if let Some(&v2) = map2.get(k) {
            let d = v1 - v2;
            res += d * d;
            n += 1;
        }
    }

    let denom = n * (n * n - 1);
    if denom == 0 {
        0.0
    } else {
        1.0 - 6.0 * res / denom as f64
    }
}

/// Rank a sequence positionally: each element gets an increasing rank.
pub fn ranks_from_sequence<K: Clone>(seq: &[K]) -> Vec<(K, f64)> {
    seq.iter()
        .enumerate()
        .map(|(i, k)| (k.clone(), i as f64))
        .collect()
}

/// Rank `(key, score)` pairs, tying a key with the previous one when their scores
/// differ by less than `rank_gap` (NLTK default `1e-15`).
pub fn ranks_from_scores<K: Clone>(scores: &[(K, f64)], rank_gap: f64) -> Vec<(K, f64)> {
    let mut out = Vec::with_capacity(scores.len());
    let mut prev: Option<f64> = None;
    let mut rank = 0usize;
    for (i, (key, score)) in scores.iter().enumerate() {
        if let Some(p) = prev {
            if (score - p).abs() > rank_gap {
                rank = i;
            }
        }
        out.push((key.clone(), rank as f64));
        prev = Some(*score);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    #[test]
    fn identical_and_opposite() {
        let a = ranks_from_sequence(&["a", "b", "c"]);
        approx(spearman_correlation(&a, &a), 1.0);

        let b = ranks_from_sequence(&["c", "b", "a"]);
        approx(spearman_correlation(&a, &b), -1.0);
    }

    #[test]
    fn single_overlap_is_zero() {
        let a = ranks_from_sequence(&["a"]);
        approx(spearman_correlation(&a, &a), 0.0);
    }

    #[test]
    fn ranks_from_scores_ties() {
        let scores = [("a", 1.0), ("b", 1.0), ("c", 2.0)];
        let ranks = ranks_from_scores(&scores, 1e-15);
        assert_eq!(ranks, vec![("a", 0.0), ("b", 0.0), ("c", 2.0)]);
    }
}

//! Scoring metrics, ported from `nltk.metrics.scores`.
//!
//! Ports the self-contained scoring functions. Not ported, by design:
//! `log_likelihood` (operates on probability-distribution objects, not a plain
//! numeric API) and `approxrand` (an RNG-driven approximate significance test).

use std::collections::HashSet;
use std::hash::Hash;

/// Fraction of positions where `reference` and `test` are equal.
///
/// Port of `accuracy`. Panics if the slices differ in length (NLTK raises
/// `ValueError`).
pub fn accuracy<T: PartialEq>(reference: &[T], test: &[T]) -> f64 {
    assert_eq!(
        reference.len(),
        test.len(),
        "lists must have the same length"
    );
    let equal = reference.iter().zip(test).filter(|(a, b)| a == b).count();
    equal as f64 / test.len() as f64
}

/// Fraction of `test` values that appear in `reference`: `|ref ∩ test| / |test|`.
/// `None` if `test` is empty. Port of `precision`.
pub fn precision<T: Eq + Hash>(reference: &HashSet<T>, test: &HashSet<T>) -> Option<f64> {
    if test.is_empty() {
        None
    } else {
        Some(reference.intersection(test).count() as f64 / test.len() as f64)
    }
}

/// Fraction of `reference` values that appear in `test`: `|ref ∩ test| / |ref|`.
/// `None` if `reference` is empty. Port of `recall`.
pub fn recall<T: Eq + Hash>(reference: &HashSet<T>, test: &HashSet<T>) -> Option<f64> {
    if reference.is_empty() {
        None
    } else {
        Some(reference.intersection(test).count() as f64 / reference.len() as f64)
    }
}

/// Harmonic mean of precision and recall, weighted by `alpha` (NLTK default 0.5):
/// `1 / (alpha/p + (1-alpha)/r)`.
///
/// `None` if either set is empty; `Some(0.0)` if precision or recall is 0.
/// Port of `f_measure`.
pub fn f_measure<T: Eq + Hash>(
    reference: &HashSet<T>,
    test: &HashSet<T>,
    alpha: f64,
) -> Option<f64> {
    let p = precision(reference, test)?;
    let r = recall(reference, test)?;
    if p == 0.0 || r == 0.0 {
        return Some(0.0);
    }
    Some(1.0 / (alpha / p + (1.0 - alpha) / r))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }
    fn list(words: &str) -> Vec<String> {
        words.split_whitespace().map(str::to_string).collect()
    }
    fn set(words: &str) -> HashSet<String> {
        words.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn demo_values() {
        // From the NLTK scores demo.
        let reference = list("DET NN VB DET JJ NN NN IN DET NN");
        let test = list("DET VB VB DET NN NN NN IN DET NN");
        approx(accuracy(&reference, &test), 0.8);

        let r = set("DET NN VB DET JJ NN NN IN DET NN");
        let t = set("DET VB VB DET NN NN NN IN DET NN");
        approx(precision(&r, &t).unwrap(), 1.0);
        approx(recall(&r, &t).unwrap(), 0.8);
        approx(f_measure(&r, &t, 0.5).unwrap(), 0.888_888_888_888_888_8);
    }

    #[test]
    fn empty_and_zero_cases() {
        let empty: HashSet<String> = HashSet::new();
        let nonempty = set("a b");
        assert_eq!(precision(&nonempty, &empty), None);
        assert_eq!(recall(&empty, &nonempty), None);
        // Disjoint sets: precision 0 -> f_measure 0.
        approx(f_measure(&set("a b"), &set("c d"), 0.5).unwrap(), 0.0);
    }
}

# nltk-metrics

A dependency-free Rust port of NLTK's `nltk.metrics`, verified differentially
against NLTK.

## distance

String and set distance metrics:

```rust
use nltk_metrics::distance::*;

assert_eq!(edit_distance("rain", "shine", 1, false), 3);
assert_eq!(edit_distance("abcdef", "acbdef", 1, true), 1); // Damerau (transpositions)
let s = jaro_winkler_similarity("MARTHA", "MARHTA", 0.1, 4);
```

Functions: `edit_distance` (with substitution cost and optional transpositions),
`binary_distance`, `jaccard_distance`, `masi_distance`, `interval_distance`,
`jaro_similarity`, `jaro_winkler_similarity`.

## agreement

Inter-annotator agreement coefficients from an `AnnotationTask` of `(coder, item,
label)` triples plus a label distance metric:

```rust
use nltk_metrics::agreement::AnnotationTask;

let data = vec![
    ("a".into(), "1".into(), "stat".into()),
    ("b".into(), "1".into(), "stat".into()),
    // ...
];
let task = AnnotationTask::with_binary(data);
let k = task.kappa();      // also: avg_ao, s, pi, multi_kappa, alpha, weighted_kappa
```

## scores

Scoring metrics over sequences and sets:

```rust
use nltk_metrics::scores::*;
use std::collections::HashSet;

assert_eq!(accuracy(&["a", "b", "c"], &["a", "x", "c"]), 2.0 / 3.0);
let r: HashSet<_> = ["a", "b"].into_iter().collect();
let t: HashSet<_> = ["a", "c"].into_iter().collect();
let f = f_measure(&r, &t, 0.5); // Option<f64>
```

Functions: `accuracy`, `precision`, `recall`, `f_measure`. (NLTK's `log_likelihood`
and `approxrand` are not ported, see the module docs.)

## association

Bigram association measures over contingency-table marginals (for collocation
finding):

```rust
use nltk_metrics::association::BigramMarginals;

let m = BigramMarginals { n_ii: 8.0, n_ix: 15.0, n_xi: 12.0, n_xx: 10000.0 };
let score = m.pmi(); // also: student_t, chi_sq, likelihood_ratio, dice, jaccard, ...
```

Measures: `raw_freq`, `student_t`, `pmi`, `mi_like`, `poisson_stirling`, `chi_sq`,
`phi_sq`, `likelihood_ratio`, `jaccard`, `dice`. (NLTK's scipy-based `fisher` and the
trigram/quadgram classes are not ported.)

## segmentation

Text segmentation evaluation metrics over `"0"`/`"1"` sequences:

```rust
use nltk_metrics::segmentation::*;

let wd = windowdiff("000100000010", "000010000100", 3, '1', false);
let p = pk("0100".repeat(100).as_str(), &"1".repeat(400), Some(2), '1');
let g = ghd("1100100000", "1100010000", 1.0, 1.0, 0.5, '1');
```

Functions: `windowdiff` (Pevzner & Hearst), `pk` (Beeferman), `ghd` (generalized
Hamming distance).

## spearman

Spearman rank correlation over `(key, rank)` rankings:

```rust
use nltk_metrics::spearman::*;

let a = ranks_from_sequence(&["a", "b", "c"]);
let b = ranks_from_sequence(&["c", "b", "a"]);
assert_eq!(spearman_correlation(&a, &b), -1.0);
```

Functions: `spearman_correlation`, `ranks_from_sequence`, `ranks_from_scores`.

## Verification

Differential-tested against Python `nltk` (the oracle), zero mismatches: distance
(20,000 string pairs), agreement (3,000 tasks × 7 coefficients), scores (16,000),
association (20,000 × 10 measures), segmentation (8,000), and spearman (8,000).

Strings are compared by Unicode scalar value (`char`), matching Python's `str`.
Ported from NLTK (Apache-2.0).

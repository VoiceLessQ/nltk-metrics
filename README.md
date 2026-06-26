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

## Verification

Differential-tested against Python `nltk` (the oracle), zero mismatches: 20,000 random
string pairs (distance), 3,000 random annotation tasks of 7 coefficients each
(agreement), and 16,000 random cases (scores).

Strings are compared by Unicode scalar value (`char`), matching Python's `str`.
Ported from NLTK (Apache-2.0).

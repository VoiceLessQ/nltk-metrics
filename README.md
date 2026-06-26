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

## Verification

Differential-tested against Python `nltk` (the oracle): 20,000 random string pairs
for the distance metrics and 3,000 random annotation tasks (7 coefficients each) for
agreement, zero mismatches.

Strings are compared by Unicode scalar value (`char`), matching Python's `str`.
Ported from NLTK (Apache-2.0).

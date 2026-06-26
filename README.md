# nltk-metrics

A dependency-free Rust port of NLTK's `nltk.metrics`, verified differentially
against NLTK.

## Implemented

`distance` — string and set distance metrics:

```rust
use nltk_metrics::distance::*;

assert_eq!(edit_distance("rain", "shine", 1, false), 3);
assert_eq!(edit_distance("abcdef", "acbdef", 1, true), 1); // Damerau (transpositions)
let s = jaro_winkler_similarity("MARTHA", "MARHTA", 0.1, 4);
```

Functions: `edit_distance` (with substitution cost and optional transpositions),
`binary_distance`, `jaccard_distance`, `masi_distance`, `interval_distance`,
`jaro_similarity`, `jaro_winkler_similarity`.

## In progress

`agreement` — inter-annotator agreement coefficients (Cohen/Fleiss kappa, Scott's
pi, Bennett's S, Krippendorff's alpha, weighted kappa). Not implemented yet.

## Notes

Strings are compared by Unicode scalar value (`char`), matching Python's `str`.
Ported from NLTK (Apache-2.0).

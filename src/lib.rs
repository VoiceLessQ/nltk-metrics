//! Rust port of NLTK's `nltk.metrics`.
//!
//! Faithful, dependency-free ports of the self-contained metrics, verified
//! differentially against NLTK (Python is the oracle).
//!
//! - [`distance`]: string/set distance metrics (edit distance, Jaro/Jaro-Winkler,
//!   Jaccard, MASI, ...).
//! - [`agreement`]: inter-annotator agreement coefficients (kappa, pi, alpha, ...).
//! - [`scores`]: scoring metrics (accuracy, precision, recall, f-measure).
//!
//! Ported from NLTK (Apache-2.0). Strings are compared by Unicode scalar value
//! (`char`), matching Python's `str` iteration.

pub mod agreement;
pub mod distance;
pub mod scores;

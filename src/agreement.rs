//! Inter-annotator agreement coefficients, ported from `nltk.metrics.agreement`.
//!
//! TODO (next): port `AnnotationTask`. Plan:
//! - Data is a list of `(coder, item, label)` triples; a distance metric (default
//!   `binary_distance`) compares labels.
//! - Reimplement the only `nltk.probability` pieces used: `FreqDist` (a counter)
//!   and `ConditionalFreqDist` (a map of counters). ~30 lines, no need to port the
//!   rest of `nltk.probability`.
//! - Coefficients to expose: `avg_Ao`, `S` (Bennett 1954), `pi` (Scott 1955),
//!   `kappa` (Cohen 1960), `multi_kappa` (Davies & Fleiss 1982), `alpha`
//!   (Krippendorff 1980), `weighted_kappa` (Cohen 1968). Skip the deprecated `N`.
//! - Distance metric: take a generic `Fn(&L, &L) -> f64` / trait so callers can pass
//!   `binary_distance`, `masi_distance`, etc., the way NLTK passes a function.
//! - Oracle anchors: the `artstein_poesio_example.txt` fixture in the NLTK clone
//!   (`avg_Ao = 0.88`, `pi = 0.79953`, `S = 0.82`).
//! - Watch float ordering: chained sums/products may differ from Python in the last
//!   ULP; the difftest compares within an epsilon.

//! Inter-annotator agreement coefficients, ported from `nltk.metrics.agreement`.
//!
//! An [`AnnotationTask`] holds `(coder, item, label)` triples and a distance metric
//! over labels (default: [`binary_distance`](crate::distance::binary_distance)), and
//! computes the chance-corrected agreement coefficients surveyed by Artstein and
//! Poesio (2007): observed agreement, Bennett's S, Scott's pi, Cohen's kappa,
//! Fleiss/Davies multi-kappa, Krippendorff's alpha, and weighted kappa.
//!
//! Coders and items are `String`; labels are generic (`L: Eq + Hash + Clone`), so
//! they can be plain categories, or sets for the MASI metric.

use std::collections::HashMap;
use std::hash::Hash;

use crate::distance::binary_distance;

type DistanceFn<L> = Box<dyn Fn(&L, &L) -> f64>;

/// An annotation task: coders assigning labels to items, plus a label distance.
pub struct AnnotationTask<L> {
    data: Vec<(String, String, L)>,
    coders: Vec<String>,
    items: Vec<String>,
    distance: DistanceFn<L>,
}

impl<L: Eq + Hash + Clone> AnnotationTask<L> {
    /// Build a task from `(coder, item, label)` triples and a label distance metric.
    pub fn new(data: Vec<(String, String, L)>, distance: impl Fn(&L, &L) -> f64 + 'static) -> Self {
        let mut coders = Vec::new();
        let mut items = Vec::new();
        for (c, i, _) in &data {
            if !coders.contains(c) {
                coders.push(c.clone());
            }
            if !items.contains(i) {
                items.push(i.clone());
            }
        }
        AnnotationTask {
            data,
            coders,
            items,
            distance: Box::new(distance),
        }
    }

    fn n_items(&self) -> f64 {
        self.items.len() as f64
    }

    /// Distinct labels (the set `K`).
    fn label_set(&self) -> Vec<L> {
        let mut out: Vec<L> = Vec::new();
        for (_, _, l) in &self.data {
            if !out.contains(l) {
                out.push(l.clone());
            }
        }
        out
    }

    /// The label coder `c` gave item `i` (first match, as in NLTK).
    fn label(&self, coder: &str, item: &str) -> &L {
        self.data
            .iter()
            .find(|(c, i, _)| c == coder && i == item)
            .map(|(_, _, l)| l)
            .expect("no label for (coder, item)")
    }

    /// Observed agreement between two coders on one item.
    fn agr(&self, c_a: &str, c_b: &str, item: &str) -> f64 {
        1.0 - (self.distance)(self.label(c_a, item), self.label(c_b, item))
    }

    /// Observed agreement between two coders over all items.
    fn ao(&self, c_a: &str, c_b: &str) -> f64 {
        let total: f64 = self.items.iter().map(|i| self.agr(c_a, c_b, i)).sum();
        total / self.n_items()
    }

    /// Average of `f(c_a, c_b)` over every unordered coder pair.
    fn pairwise_average(&self, f: impl Fn(&str, &str) -> f64) -> f64 {
        let mut total = 0.0;
        let mut n = 0;
        for a in 0..self.coders.len() {
            for b in (a + 1)..self.coders.len() {
                total += f(&self.coders[a], &self.coders[b]);
                n += 1;
            }
        }
        total / n as f64
    }

    /// Average observed agreement across all coder pairs.
    pub fn avg_ao(&self) -> f64 {
        self.pairwise_average(|a, b| self.ao(a, b))
    }

    fn chance_corrected(&self, observed: f64, expected: f64) -> f64 {
        // math.isclose(expected, 1.0) with the Python default rel_tol = 1e-9.
        if is_close(expected, 1.0) {
            if is_close(observed, 1.0) {
                return 1.0;
            }
            panic!(
                "expected agreement is 1.0 but observed is {observed:.4}; \
                 the distance function likely violates distance(l, l) = 0"
            );
        }
        (observed - expected) / (1.0 - expected)
    }

    /// Bennett, Albert and Goldstein (1954).
    pub fn s(&self) -> f64 {
        let k = self.label_set().len();
        assert!(k > 0, "cannot calculate S, no data present");
        let ae = 1.0 / k as f64;
        self.chance_corrected(self.avg_ao(), ae)
    }

    /// Scott (1955); here multi-pi.
    pub fn pi(&self) -> f64 {
        let mut freqs: HashMap<&L, usize> = HashMap::new();
        for (_, _, l) in &self.data {
            *freqs.entry(l).or_insert(0) += 1;
        }
        let total: f64 = freqs.values().map(|&f| (f * f) as f64).sum();
        let ae = total / ((self.items.len() * self.coders.len()) as f64).powi(2);
        self.chance_corrected(self.avg_ao(), ae)
    }

    /// Expected agreement for Cohen's kappa between two coders.
    fn ae_kappa(&self, c_a: &str, c_b: &str) -> f64 {
        let nitems = self.n_items();
        // ConditionalFreqDist((label, coder)): cfd[label][coder] = count.
        let mut cfd: HashMap<&L, HashMap<&str, usize>> = HashMap::new();
        for (c, _, l) in &self.data {
            *cfd.entry(l).or_default().entry(c.as_str()).or_insert(0) += 1;
        }
        let mut ae = 0.0;
        for coders in cfd.values() {
            let f_a = *coders.get(c_a).unwrap_or(&0) as f64;
            let f_b = *coders.get(c_b).unwrap_or(&0) as f64;
            ae += (f_a / nitems) * (f_b / nitems);
        }
        ae
    }

    fn kappa_pairwise(&self, c_a: &str, c_b: &str) -> f64 {
        self.chance_corrected(self.ao(c_a, c_b), self.ae_kappa(c_a, c_b))
    }

    /// Cohen (1960); averages kappa naively over coder pairs.
    pub fn kappa(&self) -> f64 {
        self.pairwise_average(|a, b| self.kappa_pairwise(a, b))
    }

    /// Davies and Fleiss (1982); averages observed and expected over coder pairs.
    pub fn multi_kappa(&self) -> f64 {
        let ae = self.pairwise_average(|a, b| self.ae_kappa(a, b));
        self.chance_corrected(self.avg_ao(), ae)
    }

    /// Disagreement within a single label-frequency distribution.
    fn disagreement(&self, freqs: &HashMap<L, usize>) -> f64 {
        let total: usize = freqs.values().sum();
        let mut pairs = 0.0;
        for (j, &nj) in freqs {
            for (l, &nl) in freqs {
                pairs += (nj * nl) as f64 * (self.distance)(l, j);
            }
        }
        pairs / (total as f64 * (total as f64 - 1.0))
    }

    fn labels_for_item(&self, item: &str) -> Vec<L> {
        self.data
            .iter()
            .filter(|(_, i, _)| i == item)
            .map(|(_, _, l)| l.clone())
            .collect()
    }

    /// Krippendorff (1980).
    pub fn alpha(&self) -> f64 {
        let k = self.label_set().len();
        assert!(k > 0, "cannot calculate alpha, no data present");
        if k == 1 {
            return 1.0;
        }
        assert!(
            !(self.coders.len() == 1 && self.items.len() == 1),
            "cannot calculate alpha, only one coder and item present"
        );

        let mut all_valid: HashMap<L, usize> = HashMap::new();
        let mut total_do = 0.0;
        for item in &self.items {
            let mut freqs: HashMap<L, usize> = HashMap::new();
            for l in self.labels_for_item(item) {
                *freqs.entry(l).or_insert(0) += 1;
            }
            let count: usize = freqs.values().sum();
            if count < 2 {
                continue;
            }
            for (l, n) in &freqs {
                *all_valid.entry(l.clone()).or_insert(0) += n;
            }
            total_do += self.disagreement(&freqs) * count as f64;
        }

        if all_valid.len() == 1 {
            return 1.0;
        }
        let observed = total_do / all_valid.values().sum::<usize>() as f64;
        let expected = self.disagreement(&all_valid);
        1.0 - observed / expected
    }

    fn do_kw_pairwise(&self, c_a: &str, c_b: &str, max_distance: f64) -> f64 {
        let total: f64 = self
            .items
            .iter()
            .map(|i| (self.distance)(self.label(c_a, i), self.label(c_b, i)))
            .sum();
        total / (self.n_items() * max_distance)
    }

    fn weighted_kappa_pairwise(&self, c_a: &str, c_b: &str, max_distance: f64) -> f64 {
        // ConditionalFreqDist((coder, label)) restricted to the two coders.
        let mut cfd: HashMap<&str, HashMap<&L, usize>> = HashMap::new();
        for (c, _, l) in &self.data {
            if c == c_a || c == c_b {
                *cfd.entry(c.as_str()).or_default().entry(l).or_insert(0) += 1;
            }
        }
        let labels = self.label_set();
        let mut total = 0.0;
        for j in &labels {
            for l in &labels {
                let f_a = cfd.get(c_a).and_then(|m| m.get(j)).copied().unwrap_or(0) as f64;
                let f_b = cfd.get(c_b).and_then(|m| m.get(l)).copied().unwrap_or(0) as f64;
                total += f_a * f_b * (self.distance)(j, l);
            }
        }
        let expected = total / (max_distance * (self.n_items()).powi(2));
        let observed = self.do_kw_pairwise(c_a, c_b, max_distance);
        1.0 - observed / expected
    }

    /// Cohen (1968), weighted kappa, averaged over coder pairs. `max_distance` is
    /// the maximum possible label distance (1.0 for `binary_distance`).
    pub fn weighted_kappa(&self, max_distance: f64) -> f64 {
        self.pairwise_average(|a, b| self.weighted_kappa_pairwise(a, b, max_distance))
    }
}

impl<L: Eq + Hash + Clone + PartialEq> AnnotationTask<L> {
    /// Build a task using [`binary_distance`] as the label metric.
    pub fn with_binary(data: Vec<(String, String, L)>) -> Self {
        AnnotationTask::new(data, |a, b| binary_distance(a, b))
    }
}

/// `math.isclose(a, b)` with Python's default relative tolerance (1e-9).
fn is_close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

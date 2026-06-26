//! Bigram association measures, ported from `nltk.metrics.association`
//! (`BigramAssocMeasures` and the generic `NgramAssocMeasures` it inherits).
//!
//! Scores a bigram from the marginals of its contingency table:
//! - `n_ii` = count of the bigram `(w1, w2)`
//! - `n_ix` = count of `(w1, *)`
//! - `n_xi` = count of `(*, w2)`
//! - `n_xx` = count of `(*, *)`, i.e. total bigrams
//!
//! Not ported: `fisher` (needs scipy's `fisher_exact`); trigram/quadgram measures.

/// NLTK's `_SMALL`, a floor added to avoid `log(0)` / division by zero.
const SMALL: f64 = 1e-20;

/// Bigram contingency-table marginals. Counts are `f64` (NLTK uses ints but every
/// measure divides, so the value is a float either way).
#[derive(Debug, Clone, Copy)]
pub struct BigramMarginals {
    pub n_ii: f64,
    pub n_ix: f64,
    pub n_xi: f64,
    pub n_xx: f64,
}

impl BigramMarginals {
    /// Product of the unigram counts, `marginals[UNIGRAMS]` in NLTK.
    fn unigram_product(&self) -> f64 {
        self.n_ix * self.n_xi
    }

    /// The contingency table `(n_ii, n_oi, n_io, n_oo)`.
    fn contingency(&self) -> [f64; 4] {
        let n_oi = self.n_xi - self.n_ii;
        let n_io = self.n_ix - self.n_ii;
        let n_oo = self.n_xx - self.n_ii - n_oi - n_io;
        [self.n_ii, n_oi, n_io, n_oo]
    }

    /// Expected values for each contingency cell (bigram override in NLTK).
    fn expected_values(&self) -> [f64; 4] {
        let cont = self.contingency();
        let n: f64 = cont.iter().sum();
        let mut e = [0.0; 4];
        for i in 0..4 {
            e[i] = (cont[i] + cont[i ^ 1]) * (cont[i] + cont[i ^ 2]) / n;
        }
        e
    }

    /// Raw frequency, `n_ii / n_xx`.
    pub fn raw_freq(&self) -> f64 {
        self.n_ii / self.n_xx
    }

    /// Student's t (Manning & Schutze 5.3.1).
    pub fn student_t(&self) -> f64 {
        (self.n_ii - self.unigram_product() / self.n_xx) / (self.n_ii + SMALL).sqrt()
    }

    /// Pointwise mutual information (Manning & Schutze 5.4).
    pub fn pmi(&self) -> f64 {
        (self.n_ii * self.n_xx).log2() - self.unigram_product().log2()
    }

    /// Mutual-information-like score, `n_ii^power / product(unigrams)`.
    /// NLTK's default `power` is 3.
    pub fn mi_like(&self, power: f64) -> f64 {
        self.n_ii.powf(power) / self.unigram_product()
    }

    /// Poisson-Stirling measure.
    pub fn poisson_stirling(&self) -> f64 {
        let exp = self.unigram_product() / self.n_xx;
        self.n_ii * ((self.n_ii / exp).log2() - 1.0)
    }

    /// Phi-square: the squared Pearson correlation coefficient of the table.
    pub fn phi_sq(&self) -> f64 {
        let [a, b, c, d] = self.contingency(); // n_ii, n_oi, n_io, n_oo
        // Numerator and denominator are symmetric in b<->c, so NLTK's swapped
        // (n_io, n_oi) unpacking yields the same value.
        (a * d - b * c).powi(2) / ((a + b) * (a + c) * (b + d) * (c + d))
    }

    /// Chi-square (Manning & Schutze 5.3.3): phi-square times the bigram count.
    pub fn chi_sq(&self) -> f64 {
        self.n_xx * self.phi_sq()
    }

    /// Likelihood ratio (Manning & Schutze 5.3.4).
    pub fn likelihood_ratio(&self) -> f64 {
        let cont = self.contingency();
        let exp = self.expected_values();
        2.0 * (0..4)
            .map(|i| cont[i] * ((cont[i] / (exp[i] + SMALL)) + SMALL).ln())
            .sum::<f64>()
    }

    /// Jaccard index, `n_ii / (n_ii + n_oi + n_io)`.
    pub fn jaccard(&self) -> f64 {
        let cont = self.contingency();
        cont[0] / (cont[0] + cont[1] + cont[2])
    }

    /// Dice's coefficient, `2 * n_ii / (n_ix + n_xi)`.
    pub fn dice(&self) -> f64 {
        2.0 * self.n_ii / (self.n_ix + self.n_xi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}");
    }

    #[test]
    fn closed_form_values() {
        let m = BigramMarginals {
            n_ii: 2.0,
            n_ix: 4.0,
            n_xi: 3.0,
            n_xx: 100.0,
        };
        approx(m.raw_freq(), 0.02);
        approx(m.dice(), 4.0 / 7.0);
        // pmi = log2(n_ii * n_xx) - log2(n_ix * n_xi) = log2(200) - log2(12).
        approx(m.pmi(), 200f64.log2() - 12f64.log2());
        // mi_like default power 3 = 2^3 / (4*3) = 8/12.
        approx(m.mi_like(3.0), 8.0 / 12.0);
    }
}

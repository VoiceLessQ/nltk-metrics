//! Text segmentation metrics, ported from `nltk.metrics.segmentation`.
//!
//! A segmentation is a sequence over a two-item vocabulary (e.g. `"0"`/`"1"`),
//! where the `boundary` value marks a segment edge.
//!
//! - [`windowdiff`] (Pevzner & Hearst 2002)
//! - [`ghd`] generalized Hamming distance (Bookstein, Kulyukin & Raita 2002)
//! - [`pk`] (Beeferman, Berger & Lafferty 1999)

fn count(window: &[u8], boundary: u8) -> usize {
    window.iter().filter(|&&b| b == boundary).count()
}

/// Python 3 `round`: round half to even.
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        // Exactly halfway: round to the even integer.
        let f = x.floor();
        if (f as i64) % 2 == 0 {
            f
        } else {
            f + 1.0
        }
    } else {
        r
    }
}

/// WindowDiff score for a pair of equal-length segmentations and window width `k`.
///
/// The `weighted` variant sums the absolute boundary-count differences instead of
/// thresholding each window at 1. Panics on unequal lengths or `k > len`.
pub fn windowdiff(seg1: &str, seg2: &str, k: usize, boundary: char, weighted: bool) -> f64 {
    let s1 = seg1.as_bytes();
    let s2 = seg2.as_bytes();
    assert_eq!(s1.len(), s2.len(), "segmentations have unequal length");
    assert!(k <= s1.len(), "window width k must be <= segmentation length");
    let b = boundary as u8;

    let mut wd = 0usize;
    for i in 0..(s1.len() - k + 1) {
        let ndiff = count(&s1[i..i + k], b).abs_diff(count(&s2[i..i + k], b));
        wd += if weighted { ndiff } else { ndiff.min(1) };
    }
    wd as f64 / (s1.len() - k + 1) as f64
}

/// Pk score for a reference and hypothesis segmentation.
///
/// If `k` is `None` it defaults to half the average reference segment length.
pub fn pk(reference: &str, hypothesis: &str, k: Option<usize>, boundary: char) -> f64 {
    let r = reference.as_bytes();
    let h = hypothesis.as_bytes();
    let b = boundary as u8;

    let k = k.unwrap_or_else(|| round_half_even(r.len() as f64 / (count(r, b) as f64 * 2.0)) as usize);

    let mut err = 0usize;
    for i in 0..(r.len() - k + 1) {
        let rb = count(&r[i..i + k], b) > 0;
        let hb = count(&h[i..i + k], b) > 0;
        if rb != hb {
            err += 1;
        }
    }
    err as f64 / (r.len() - k + 1) as f64
}

/// Generalized Hamming Distance: the cost of turning `hypothesis` into `reference`
/// via boundary insertion, deletion, and shift operations.
pub fn ghd(
    reference: &str,
    hypothesis: &str,
    ins_cost: f64,
    del_cost: f64,
    shift_cost_coeff: f64,
    boundary: char,
) -> f64 {
    let b = boundary as u8;
    let bounds = |s: &str| -> Vec<usize> {
        s.as_bytes()
            .iter()
            .enumerate()
            .filter(|(_, &c)| c == b)
            .map(|(i, _)| i)
            .collect()
    };
    let ref_idx = bounds(reference);
    let hyp_idx = bounds(hypothesis);
    let (nref, nhyp) = (ref_idx.len(), hyp_idx.len());

    if nref == 0 && nhyp == 0 {
        return 0.0;
    } else if nhyp == 0 {
        return nref as f64 * ins_cost;
    } else if nref == 0 {
        return nhyp as f64 * del_cost;
    }

    // mat has nhyp+1 rows (over the hypothesis boundaries) and nref+1 columns.
    let rows = nhyp + 1;
    let cols = nref + 1;
    let mut mat = vec![vec![0.0f64; cols]; rows];
    for (j, cell) in mat[0].iter_mut().enumerate() {
        *cell = ins_cost * j as f64;
    }
    for (i, row) in mat.iter_mut().enumerate() {
        row[0] = del_cost * i as f64;
    }

    for (i, &rowi) in hyp_idx.iter().enumerate() {
        for (j, &colj) in ref_idx.iter().enumerate() {
            let shift_cost = shift_cost_coeff * (rowi as f64 - colj as f64).abs() + mat[i][j];
            let tcost = if rowi == colj {
                mat[i][j]
            } else if rowi > colj {
                del_cost + mat[i][j + 1]
            } else {
                ins_cost + mat[i + 1][j]
            };
            mat[i + 1][j + 1] = tcost.min(shift_cost);
        }
    }
    mat[rows - 1][cols - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r2(x: f64) -> f64 {
        (x * 100.0).round() / 100.0
    }

    #[test]
    fn windowdiff_doctest() {
        let s1 = "000100000010";
        let s2 = "000010000100";
        let s3 = "100000010000";
        assert_eq!(r2(windowdiff(s1, s1, 3, '1', false)), 0.00);
        assert_eq!(r2(windowdiff(s1, s2, 3, '1', false)), 0.30);
        assert_eq!(r2(windowdiff(s2, s3, 3, '1', false)), 0.80);
    }

    #[test]
    fn ghd_doctest() {
        assert_eq!(ghd("1100100000", "1100010000", 1.0, 1.0, 0.5, '1'), 0.5);
        assert_eq!(ghd("1100100000", "1100000001", 1.0, 1.0, 0.5, '1'), 2.0);
        assert_eq!(ghd("011", "110", 1.0, 1.0, 0.5, '1'), 1.0);
        assert_eq!(ghd("1", "0", 1.0, 1.0, 0.5, '1'), 1.0);
        assert_eq!(ghd("111", "000", 1.0, 1.0, 0.5, '1'), 3.0);
        assert_eq!(ghd("000", "111", 1.0, 2.0, 0.5, '1'), 6.0);
    }

    #[test]
    fn pk_doctest() {
        let ref_seg = "0100".repeat(100);
        assert_eq!(r2(pk(&ref_seg, &"1".repeat(400), Some(2), '1')), 0.50);
        assert_eq!(r2(pk(&ref_seg, &"0".repeat(400), Some(2), '1')), 0.50);
        assert_eq!(r2(pk(&ref_seg, &ref_seg, Some(2), '1')), 0.00);
    }
}

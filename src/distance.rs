//! Distance metrics, ported from `nltk.metrics.distance`.
//!
//! Strings are iterated by `char` (Unicode scalar value) to match Python's `str`.

use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;

/// Levenshtein edit distance: the number of single-character substitutions,
/// insertions, and deletions to turn `s1` into `s2`.
///
/// `substitution_cost` weights substitutions (NLTK default 1). With
/// `transpositions = true` it becomes the Damerau-Levenshtein distance, also
/// counting adjacent-character transpositions.
///
/// Port of `nltk.metrics.distance.edit_distance`.
pub fn edit_distance(s1: &str, s2: &str, substitution_cost: usize, transpositions: bool) -> usize {
    let a: Vec<char> = s1.chars().collect();
    let b: Vec<char> = s2.chars().collect();
    let len1 = a.len();
    let len2 = b.len();

    // lev[i][j], dimensions (len1+1) x (len2+1), seeded with the edit prefixes.
    let mut lev = vec![vec![0usize; len2 + 1]; len1 + 1];
    for (i, row) in lev.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in lev[0].iter_mut().enumerate() {
        *cell = j;
    }

    // last position (1-based) each char was seen in s1, for transpositions.
    let mut last_left_t: HashMap<char, usize> = HashMap::new();

    for i in 1..=len1 {
        let mut last_right_buf = 0usize;
        for j in 1..=len2 {
            let last_left = *last_left_t.get(&b[j - 1]).unwrap_or(&0);
            let last_right = last_right_buf;
            if a[i - 1] == b[j - 1] {
                last_right_buf = j;
            }

            let delete = lev[i - 1][j] + 1;
            let insert = lev[i][j - 1] + 1;
            let substitute =
                lev[i - 1][j - 1] + if a[i - 1] != b[j - 1] { substitution_cost } else { 0 };
            let mut transpose = substitute + 1; // never picked unless enabled below
            if transpositions && last_left > 0 && last_right > 0 {
                transpose = lev[last_left - 1][last_right - 1] + (i - last_left) + (j - last_right) - 1;
            }

            lev[i][j] = delete.min(insert).min(substitute).min(transpose);
        }
        last_left_t.insert(a[i - 1], i);
    }

    lev[len1][len2]
}

/// Simple equality test: `0.0` if equal, `1.0` if different.
/// Port of `binary_distance`.
pub fn binary_distance<T: PartialEq>(label1: &T, label2: &T) -> f64 {
    if label1 == label2 {
        0.0
    } else {
        1.0
    }
}

/// Set-similarity distance: `1 - |A ∩ B| / |A ∪ B|`.
/// Port of `jaccard_distance`.
pub fn jaccard_distance<T: Eq + Hash>(label1: &HashSet<T>, label2: &HashSet<T>) -> f64 {
    let inter = label1.intersection(label2).count();
    let union = label1.union(label2).count();
    (union - inter) as f64 / union as f64
}

/// MASI distance: Jaccard weighted by a monotonicity factor that rewards partial
/// agreement on set-valued labels (Passonneau 2006).
/// Port of `masi_distance`.
pub fn masi_distance<T: Eq + Hash>(label1: &HashSet<T>, label2: &HashSet<T>) -> f64 {
    let len_intersection = label1.intersection(label2).count();
    let len_union = label1.union(label2).count();
    let len1 = label1.len();
    let len2 = label2.len();

    let m = if len1 == len2 && len1 == len_intersection {
        1.0
    } else if len_intersection == len1.min(len2) {
        0.67
    } else if len_intersection > 0 {
        0.33
    } else {
        0.0
    };

    1.0 - (len_intersection as f64 / len_union as f64) * m
}

/// Krippendorff's interval distance: `(a - b)^2`.
/// Port of `interval_distance`.
pub fn interval_distance(label1: f64, label2: f64) -> f64 {
    (label1 - label2).powi(2)
}

/// Jaro similarity in `[0, 1]` (1.0 = identical).
/// Port of `jaro_similarity`.
pub fn jaro_similarity(s1: &str, s2: &str) -> f64 {
    if s1 == s2 {
        return 1.0;
    }
    let a: Vec<char> = s1.chars().collect();
    let b: Vec<char> = s2.chars().collect();
    let len1 = a.len() as isize;
    let len2 = b.len() as isize;

    // Matching window. Can go negative for very short strings; that just yields
    // an empty search range below, matching NLTK.
    let match_bound = len1.max(len2) / 2 - 1;

    let mut flagged_1: Vec<usize> = Vec::new();
    let mut flagged_2: Vec<usize> = Vec::new();
    for i in 0..len1 {
        let upperbound = (i + match_bound).min(len2 - 1);
        let lowerbound = (i - match_bound).max(0);
        let mut j = lowerbound;
        while j <= upperbound {
            let ju = j as usize;
            if a[i as usize] == b[ju] && !flagged_2.contains(&ju) {
                flagged_1.push(i as usize);
                flagged_2.push(ju);
                break;
            }
            j += 1;
        }
    }

    let matches = flagged_1.len();
    if matches == 0 {
        return 0.0;
    }
    flagged_2.sort_unstable();

    let mut transpositions = 0usize;
    for (k, &i) in flagged_1.iter().enumerate() {
        if a[i] != b[flagged_2[k]] {
            transpositions += 1;
        }
    }

    let m = matches as f64;
    (1.0 / 3.0) * (m / len1 as f64 + m / len2 as f64 + (m - (transpositions / 2) as f64) / m)
}

/// Jaro-Winkler similarity: Jaro boosted by a shared prefix of up to `max_l`
/// characters, scaled by `p` (Winkler 1990; defaults `p = 0.1`, `max_l = 4`).
/// Port of `jaro_winkler_similarity`.
pub fn jaro_winkler_similarity(s1: &str, s2: &str, p: f64, max_l: usize) -> f64 {
    let jaro_sim = jaro_similarity(s1, s2);
    let mut l = 0usize;
    for (c1, c2) in s1.chars().zip(s2.chars()) {
        if c1 == c2 {
            l += 1;
        } else {
            break;
        }
        if l == max_l {
            break;
        }
    }
    jaro_sim + (l as f64 * p * (1.0 - jaro_sim))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set<const N: usize>(items: [i32; N]) -> HashSet<i32> {
        items.into_iter().collect()
    }

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}");
    }

    #[test]
    fn edit_distance_basic() {
        assert_eq!(edit_distance("rain", "shine", 1, false), 3);
        assert_eq!(edit_distance("", "abc", 1, false), 3);
        assert_eq!(edit_distance("abc", "abc", 1, false), 0);
    }

    #[test]
    fn edit_distance_transpositions() {
        // "abcdef" -> "acbdef": one transposition vs two substitutions.
        assert_eq!(edit_distance("abcdef", "acbdef", 1, false), 2);
        assert_eq!(edit_distance("abcdef", "acbdef", 1, true), 1);
    }

    #[test]
    fn simple_distances() {
        approx(binary_distance(&1, &1), 0.0);
        approx(binary_distance(&1, &3), 1.0);
        approx(jaccard_distance(&set([1, 2, 3, 4]), &set([3, 4, 5])), 0.6);
        // NLTK doctest: masi_distance({1,2}, {1,2,3,4}) == 0.665
        approx(masi_distance(&set([1, 2]), &set([1, 2, 3, 4])), 0.665);
        approx(interval_distance(1.0, 10.0), 81.0);
    }

    #[test]
    fn jaro_and_winkler() {
        approx(jaro_similarity("billy", "billy"), 1.0);
        approx(jaro_similarity("billy", "susan"), 0.0);
        // NLTK census table (rounded to 3 places).
        let r3 = |x: f64| (x * 1000.0).round() / 1000.0;
        assert_eq!(r3(jaro_similarity("MARHTA", "MARTHA")), 0.944);
        assert_eq!(r3(jaro_similarity("DWAYNE", "DUANE")), 0.822);
        // jaro_winkler('TANYA','TONYA', p=0.1, max_l=100) == 0.88
        assert_eq!(r3(jaro_winkler_similarity("TANYA", "TONYA", 0.1, 100)), 0.88);
    }
}

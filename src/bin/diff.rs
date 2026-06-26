//! Differential-test driver (dev only; excluded from the published crate).
//!
//! - `diff distance`: stdin lines `s1<TAB>s2` -> `ed edt jaro jw` per line
//!   (edit distance, Damerau edit distance, Jaro, Jaro-Winkler p=0.1 max_l=4).
//! - `diff agreement`: stdin = tasks separated by blank lines, each task a set of
//!   `coder item label` lines -> `avg_ao s pi kappa multi_kappa alpha weighted_kappa`.
//! - `diff scores`: per-line, `acc<TAB>a<TAB>b` -> `accuracy`, or
//!   `set<TAB>alpha<TAB>a<TAB>b` -> `precision recall f_measure` (NA for None).

use std::collections::HashSet;
use std::io::{self, Read, Write};

use nltk_metrics::agreement::AnnotationTask;
use nltk_metrics::distance::{edit_distance, jaro_similarity, jaro_winkler_similarity};
use nltk_metrics::scores::{accuracy, f_measure, precision, recall};

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    // Normalize CRLF so blank-line block splitting works regardless of how the
    // caller's stdin was encoded (Windows text-mode pipes inject \r\n).
    let input = input.replace("\r\n", "\n");
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    match mode.as_str() {
        "distance" => {
            for line in input.lines() {
                let mut parts = line.split('\t');
                let s1 = parts.next().unwrap_or("");
                let s2 = parts.next().unwrap_or("");
                let ed = edit_distance(s1, s2, 1, false);
                let edt = edit_distance(s1, s2, 1, true);
                let jaro = jaro_similarity(s1, s2);
                let jw = jaro_winkler_similarity(s1, s2, 0.1, 4);
                writeln!(out, "{ed} {edt} {jaro:.12} {jw:.12}").unwrap();
            }
        }
        "agreement" => {
            for block in input.split("\n\n") {
                let data: Vec<(String, String, String)> = block
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .map(|l| {
                        let mut it = l.split_whitespace();
                        (
                            it.next().unwrap().to_string(),
                            it.next().unwrap().to_string(),
                            it.next().unwrap().to_string(),
                        )
                    })
                    .collect();
                if data.is_empty() {
                    continue;
                }
                let t = AnnotationTask::with_binary(data);
                writeln!(
                    out,
                    "{:.12} {:.12} {:.12} {:.12} {:.12} {:.12} {:.12}",
                    t.avg_ao(),
                    t.s(),
                    t.pi(),
                    t.kappa(),
                    t.multi_kappa(),
                    t.alpha(),
                    t.weighted_kappa(1.0)
                )
                .unwrap();
            }
        }
        "scores" => {
            let fmt = |v: Option<f64>| v.map_or_else(|| "NA".to_string(), |x| format!("{x:.12}"));
            for line in input.lines() {
                let mut f = line.split('\t');
                match f.next().unwrap_or("") {
                    "acc" => {
                        let a: Vec<&str> = f.next().unwrap_or("").split_whitespace().collect();
                        let b: Vec<&str> = f.next().unwrap_or("").split_whitespace().collect();
                        writeln!(out, "{:.12}", accuracy(&a, &b)).unwrap();
                    }
                    "set" => {
                        let alpha: f64 = f.next().unwrap_or("0.5").parse().unwrap();
                        let a: HashSet<&str> = f.next().unwrap_or("").split_whitespace().collect();
                        let b: HashSet<&str> = f.next().unwrap_or("").split_whitespace().collect();
                        writeln!(
                            out,
                            "{} {} {}",
                            fmt(precision(&a, &b)),
                            fmt(recall(&a, &b)),
                            fmt(f_measure(&a, &b, alpha))
                        )
                        .unwrap();
                    }
                    _ => {}
                }
            }
        }
        _ => {
            eprintln!("usage: diff <distance|agreement|scores>");
            std::process::exit(2);
        }
    }
}

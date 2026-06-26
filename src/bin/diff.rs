//! Differential-test driver (dev only; excluded from the published crate).
//!
//! - `diff distance`: stdin lines `s1<TAB>s2` -> `ed edt jaro jw` per line
//!   (edit distance, Damerau edit distance, Jaro, Jaro-Winkler p=0.1 max_l=4).
//! - `diff agreement`: stdin = tasks separated by blank lines, each task a set of
//!   `coder item label` lines -> `avg_ao s pi kappa multi_kappa alpha weighted_kappa`.

use std::io::{self, Read, Write};

use nltk_metrics::agreement::AnnotationTask;
use nltk_metrics::distance::{edit_distance, jaro_similarity, jaro_winkler_similarity};

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
        _ => {
            eprintln!("usage: diff <distance|agreement>");
            std::process::exit(2);
        }
    }
}

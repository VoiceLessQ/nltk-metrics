//! Agreement coefficients against NLTK's Artstein & Poesio (2007) example data.
//! Oracle values produced by `nltk.metrics.agreement.AnnotationTask`.

use nltk_metrics::agreement::AnnotationTask;

fn fixture() -> AnnotationTask<String> {
    let text = include_str!("artstein_poesio_example.txt");
    let data = text
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
    AnnotationTask::with_binary(data)
}

fn approx(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9, "{a} != {b}");
}

#[test]
fn artstein_poesio_coefficients() {
    let t = fixture();
    approx(t.avg_ao(), 0.88);
    approx(t.s(), 0.8199999999999998);
    approx(t.pi(), 0.7995322418977615);
    approx(t.kappa(), 0.8013245033112583);
    approx(t.multi_kappa(), 0.8013245033112583);
    approx(t.alpha(), 0.8005345806882727);
    approx(t.weighted_kappa(1.0), 0.8013245033112583);
}

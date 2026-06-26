"""Differential test: the diff binary vs NLTK's nltk.metrics (the oracle).

Covers both modules with randomized inputs:
  distance  - random string pairs -> edit_distance (plain + Damerau), Jaro, Jaro-Winkler
  agreement - random annotation tasks -> avg_Ao, S, pi, kappa, multi_kappa, alpha, weighted_kappa

Floats compare within a small epsilon (chained float math, e.g. jaro_winkler and the
agreement coefficients, can differ from Python in the last ULP). Exits non-zero on any
real mismatch.

Usage: python difftest.py [path-to-diff-binary]
"""
import math
import os
import random
import subprocess
import sys

from nltk.metrics.agreement import AnnotationTask
from nltk.metrics.distance import (
    binary_distance,
    edit_distance,
    jaro_similarity,
    jaro_winkler_similarity,
)
from nltk.metrics.association import BigramAssocMeasures
from nltk.metrics.scores import accuracy, f_measure, precision, recall
from nltk.metrics.segmentation import ghd, pk, windowdiff
from nltk.metrics.spearman import ranks_from_sequence, spearman_correlation

random.seed(20260626)
EPS = 1e-9


def find_binary():
    if len(sys.argv) > 1:
        return sys.argv[1]
    exe = "diff.exe" if os.name == "nt" else "diff"
    for profile in ("release", "debug"):
        path = os.path.join("target", profile, exe)
        if os.path.exists(path):
            return path
    sys.exit("diff binary not found; build it first (cargo build --release --bin diff)")


def run(binary, mode, stdin):
    proc = subprocess.run(
        [binary, mode], input=stdin, capture_output=True, text=True, encoding="utf-8"
    )
    if proc.returncode != 0:
        sys.exit(f"diff {mode} failed (exit {proc.returncode}):\n{proc.stderr}")
    return proc.stdout.splitlines()


def close(a, b):
    return abs(a - b) <= EPS


def check_distance(binary):
    alphabet = "abcde"

    def rand_str():
        return "".join(random.choice(alphabet) for _ in range(random.randint(0, 8)))

    pairs = [(rand_str(), rand_str()) for _ in range(20000)]
    stdin = "".join(f"{s1}\t{s2}\n" for s1, s2 in pairs)
    out = run(binary, "distance", stdin)

    mismatches = 0
    for (s1, s2), line in zip(pairs, out):
        ed, edt, jaro, jw = line.split()
        exp = (
            edit_distance(s1, s2),
            edit_distance(s1, s2, transpositions=True),
            jaro_similarity(s1, s2),
            jaro_winkler_similarity(s1, s2),
        )
        got = (int(ed), int(edt), float(jaro), float(jw))
        if got[0] != exp[0] or got[1] != exp[1] or not close(got[2], exp[2]) or not close(got[3], exp[3]):
            mismatches += 1
            if mismatches <= 15:
                print(f"  distance {s1!r},{s2!r}: nltk={exp} diff={got}")
    print(f"distance: {len(pairs)} cases, {mismatches} mismatches")
    return mismatches


def gen_task():
    coders = [f"c{i}" for i in range(random.randint(2, 4))]
    items = [f"i{i}" for i in range(random.randint(3, 12))]
    pool = [f"l{i}" for i in range(random.randint(2, 4))]
    return [(c, it, random.choice(pool)) for c in coders for it in items]


def check_agreement(binary):
    coeffs = ["avg_Ao", "S", "pi", "kappa", "multi_kappa", "alpha", "weighted_kappa"]
    tasks, expected = [], []
    attempts = 0
    while len(tasks) < 3000 and attempts < 8000:
        attempts += 1
        data = gen_task()
        t = AnnotationTask(data=data, distance=binary_distance)
        try:
            vals = [float(getattr(t, c)()) for c in coeffs]
        except Exception:
            continue  # degenerate task (e.g. division by zero); NLTK would raise too
        if any(math.isnan(v) or math.isinf(v) for v in vals):
            continue
        tasks.append(data)
        expected.append(vals)

    blocks = ["\n".join(f"{c} {it} {l}" for c, it, l in data) for data in tasks]
    stdin = "\n\n".join(blocks) + "\n"
    out = run(binary, "agreement", stdin)
    if len(out) != len(tasks):
        sys.exit(f"agreement output line count {len(out)} != task count {len(tasks)}")

    mismatches = 0
    for idx, (line, exp) in enumerate(zip(out, expected)):
        got = [float(x) for x in line.split()]
        for name, g, e in zip(coeffs, got, exp):
            if not close(g, e):
                mismatches += 1
                if mismatches <= 15:
                    print(f"  agreement task#{idx} {name}: nltk={e!r} diff={g!r}")
    print(f"agreement: {len(tasks)} tasks, {mismatches} coefficient mismatches")
    return mismatches


def check_scores(binary):
    alphabet = "abcdef"
    kinds, lines, expected = [], [], []

    # accuracy: equal-length non-empty token lists.
    for _ in range(8000):
        n = random.randint(1, 8)
        ref = [random.choice(alphabet) for _ in range(n)]
        test = [random.choice(alphabet) for _ in range(n)]
        kinds.append("acc")
        lines.append("acc\t" + " ".join(ref) + "\t" + " ".join(test))
        expected.append(accuracy(ref, test))

    # precision/recall/f_measure: random sets (possibly empty), random alpha.
    for _ in range(8000):
        ref = set(random.sample(alphabet, random.randint(0, len(alphabet))))
        test = set(random.sample(alphabet, random.randint(0, len(alphabet))))
        alpha = random.choice([0.1, 0.3, 0.5, 0.7, 0.9])
        kinds.append("set")
        lines.append(f"set\t{alpha}\t" + " ".join(ref) + "\t" + " ".join(test))
        expected.append((precision(ref, test), recall(ref, test), f_measure(ref, test, alpha)))

    out = run(binary, "scores", "\n".join(lines) + "\n")
    if len(out) != len(lines):
        sys.exit(f"scores output line count {len(out)} != input {len(lines)}")

    def match(got_str, exp):
        if exp is None:
            return got_str == "NA"
        return got_str != "NA" and close(float(got_str), float(exp))

    mismatches = 0
    for i, kind in enumerate(kinds):
        if kind == "acc":
            ok = close(float(out[i]), expected[i])
        else:
            ok = all(match(g, e) for g, e in zip(out[i].split(), expected[i]))
        if not ok:
            mismatches += 1
            if mismatches <= 15:
                print(f"  scores {kind} #{i}: nltk={expected[i]} diff={out[i]!r}")
    print(f"scores: {len(lines)} cases, {mismatches} mismatches")
    return mismatches


def close_rel(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def check_association(binary):
    B = BigramAssocMeasures
    measures = [
        lambda m: B.raw_freq(*m),
        lambda m: B.student_t(*m),
        lambda m: B.pmi(*m),
        lambda m: B.mi_like(*m),
        lambda m: B.poisson_stirling(*m),
        lambda m: B.chi_sq(*m),
        lambda m: B.phi_sq(*m),
        lambda m: B.likelihood_ratio(*m),
        lambda m: B.jaccard(*m),
        lambda m: B.dice(*m),
    ]

    lines, expected = [], []
    for _ in range(20000):
        # Random valid contingency cells (>= 1 avoids any zero denominators / log(0)).
        a, b, c, d = (random.randint(1, 60) for _ in range(4))
        n_ii, n_ix, n_xi, n_xx = a, b + a, c + a, a + b + c + d
        marginals = (n_ii, (n_ix, n_xi), n_xx)
        lines.append(f"{n_ii} {n_ix} {n_xi} {n_xx}")
        expected.append([fn(marginals) for fn in measures])

    out = run(binary, "association", "\n".join(lines) + "\n")
    if len(out) != len(lines):
        sys.exit(f"association output line count {len(out)} != input {len(lines)}")

    mismatches = 0
    for i, line in enumerate(out):
        got = [float(x) for x in line.split()]
        for g, e in zip(got, expected[i]):
            if not close_rel(g, e):
                mismatches += 1
                if mismatches <= 15:
                    print(f"  association #{i} {lines[i]}: nltk={e!r} diff={g!r}")
    print(f"association: {len(lines)} cases, {mismatches} measure mismatches")
    return mismatches


def rand_seg(n):
    return "".join(random.choice("01") for _ in range(n))


def check_segmentation(binary):
    lines, expected = [], []
    for _ in range(8000):
        n = random.randint(2, 30)
        s1, s2 = rand_seg(n), rand_seg(n)
        op = random.choice(["wd", "pk", "ghd"])
        if op == "wd":
            k = random.randint(1, n)
            w = random.randint(0, 1)
            lines.append(f"wd {s1} {s2} {k} {w}")
            expected.append(windowdiff(s1, s2, k, weighted=bool(w)))
        elif op == "pk":
            k = random.randint(1, n)
            lines.append(f"pk {s1} {s2} {k}")
            expected.append(pk(s1, s2, k))
        else:
            ins = random.choice([1.0, 2.0, 0.5])
            dele = random.choice([1.0, 2.0, 0.5])
            shift = random.choice([0.5, 1.0, 2.0])
            lines.append(f"ghd {s1} {s2} {ins} {dele} {shift}")
            expected.append(ghd(s1, s2, ins, dele, shift))

    out = run(binary, "segmentation", "\n".join(lines) + "\n")
    if len(out) != len(lines):
        sys.exit(f"segmentation output {len(out)} != input {len(lines)}")
    mismatches = 0
    for i, line in enumerate(out):
        if not close_rel(float(line), expected[i]):
            mismatches += 1
            if mismatches <= 15:
                print(f"  segmentation #{i} {lines[i]}: nltk={expected[i]!r} diff={line!r}")
    print(f"segmentation: {len(lines)} cases, {mismatches} mismatches")
    return mismatches


def check_spearman(binary):
    keys = [f"k{i}" for i in range(12)]
    lines, expected = [], []
    for _ in range(8000):
        m = random.randint(1, len(keys))
        ks = random.sample(keys, m)
        seq1 = ks[:]
        seq2 = ks[:]
        random.shuffle(seq1)
        random.shuffle(seq2)
        r1 = list(ranks_from_sequence(seq1))
        r2 = list(ranks_from_sequence(seq2))
        enc = lambda r: ",".join(f"{k}:{v}" for k, v in r)
        lines.append(enc(r1) + "\t" + enc(r2))
        expected.append(spearman_correlation(r1, r2))

    out = run(binary, "spearman", "\n".join(lines) + "\n")
    if len(out) != len(lines):
        sys.exit(f"spearman output {len(out)} != input {len(lines)}")
    mismatches = 0
    for i, line in enumerate(out):
        if not close_rel(float(line), expected[i]):
            mismatches += 1
            if mismatches <= 15:
                print(f"  spearman #{i}: nltk={expected[i]!r} diff={line!r}")
    print(f"spearman: {len(lines)} cases, {mismatches} mismatches")
    return mismatches


def main():
    binary = find_binary()
    total = (
        check_distance(binary)
        + check_agreement(binary)
        + check_scores(binary)
        + check_association(binary)
        + check_segmentation(binary)
        + check_spearman(binary)
    )
    if total:
        sys.exit(1)
    print("ALL MATCH")


if __name__ == "__main__":
    main()

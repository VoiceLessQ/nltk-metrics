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


def main():
    binary = find_binary()
    total = check_distance(binary) + check_agreement(binary)
    if total:
        sys.exit(1)
    print("ALL MATCH")


if __name__ == "__main__":
    main()

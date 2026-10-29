"""Forgetting curves beyond the exponential: fit quality and retrieval at scale.

Part 1 (fit): how well R non-negative exponentials approximate three forgetting curves over a
year of ages: ACT-R's power law (d = 0.5, one-hour offset), a hyperbolic curve and FadeMem's
stretched exponential. Output: results/curve_fit.csv

Part 2 (static): retrieval of the top-10 memories under the ACT-R-style score
    a*D + b*importance + c*relevance, min-max normalised, a = b = c = 1,
    D_i(T) = sum over every use j of memory i of (1 + (T - t_ij) / 1 h) ** -0.5
on GloVe vectors with synthetic use histories: creation uniform over 90 days, a geometric
number of later uses (mean 1.2) at uniform times after creation, importance drawn from the
Willowbrook distribution. Methods: exact scan with the fitted curve (approximation only), the
folded store (recent part + one segment, as after a compaction), LangChain-style overfetch with
exact normalisation ranges, and relevance only. Recall is against an exact scan with the true
curve. Output: results/static_curves.csv

    python -m experiments.exp_curves --part fit
    python -m experiments.exp_curves --part static --sizes 100000 1000000
    python -m experiments.exp_curves --part fronts --sizes 100000 1000000
    python -m experiments.exp_curves --part ranges --sizes 100000
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from agentmem import ACT_R, FADEMEM, PROFILES, fit_expsum, hyperbolic
from agentmem.curve_stores import CurveExact, CurveExactFit, CurveFolded, CurveOverfetch
from agentmem.score import minmax

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
HORIZON_H = 90 * 24.0
CURVES = {"actr": ACT_R, "hyperbolic": hyperbolic(24.0), "fademem": FADEMEM}


def part_fit() -> None:
    rows = []
    for key, curve in CURVES.items():
        for R in (2, 4, 6, 8, 10, 12):
            f = fit_expsum(curve, R)
            rel, rng = f.errors()
            rows.append(dict(curve=key, label=curve.label, R=R, max_rel_err=rel, max_range_err=rng,
                             fastest_rate=float(f.lam.max()), slowest_rate=float(f.lam.min())))
            print(f"{curve.label:55s} R={R:2d}  relative {rel * 100:7.3f}%  of range {rng * 100:7.4f}%")
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "curve_fit.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def histories_dataset(n: int, nq: int, seed: int = 0):
    from agentmem.embed import load_glove
    from experiments.exp_static import town_importance_distribution

    _, V = load_glove("glove-twitter-100")
    V = np.asarray(V, dtype=np.float32)
    nrm = np.linalg.norm(V, axis=1)
    V = V[nrm > 0] / nrm[nrm > 0, None]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(V))
    Q = np.ascontiguousarray(V[perm[:nq]])
    E = np.ascontiguousarray(V[perm[nq:nq + n]])
    created = np.sort(rng.uniform(0.0, HORIZON_H, n))       # ids in creation order
    uses = rng.geometric(0.45, n) - 1
    hist = [np.concatenate([[c], np.sort(rng.uniform(c, HORIZON_H, u))]) for c, u in zip(created, uses)]
    vals, p = town_importance_distribution()
    imp = rng.choice(vals, size=n, p=p)
    return E, imp, hist, Q, HORIZON_H + 0.25


def recall(got, truth) -> float:
    return len(set(np.asarray(got).tolist()) & truth) / len(truth)


def part_static(sizes, nq: int, k: int, R: int, out: Path, threads: int) -> None:
    w = PROFILES["paper"]
    rows = []

    def emit(**r):
        rows.append(r)
        print("  " + "  ".join(f"{a}={b:.4g}" if isinstance(b, float) else f"{a}={b}" for a, b in r.items()), flush=True)
        with out.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)

    for n in sizes:
        E, imp, hist, Q, T = histories_dataset(n, nq)
        d = E.shape[1]
        print(f"n = {n:,}: {sum(len(h) for h in hist):,} uses", flush=True)
        # ground truth with the true curve, and the paper's score for comparison
        st = CurveExact(d, curve=ACT_R, R=R, capacity=n)
        st.load(E, imp, hist, T)
        truth, paper_overlap, age_truth, age_paper = [], [], [], []
        rec_exp = minmax(0.995 ** (T - st.t_last[:n]))
        for q in Q:
            s = st.exact(q, T, w)
            top = set(np.argpartition(-s, k)[:k].tolist())
            truth.append(top)
            sp = rec_exp + minmax(st.imp[:n]) + minmax((E @ q).astype(np.float64))
            tp = set(np.argpartition(-sp, k)[:k].tolist())
            paper_overlap.append(len(top & tp) / k)
            age_truth.append(np.median(T - st.t_last[list(top)]))
            age_paper.append(np.median(T - st.t_last[list(tp)]))
        emit(n=n, method="paper score vs ACT-R score", param="overlap of top-10", recall=float(np.mean(paper_overlap)),
             ms=float("nan"), build_s=0.0, extra=f"median age of last use {np.median(age_truth):.1f} h vs {np.median(age_paper):.1f} h")
        del st
        gc.collect()
        # approximation alone
        st = CurveExactFit(d, curve=ACT_R, R=R, capacity=n)
        st.load(E, imp, hist, T)
        res, tms = [], []
        for q, tr in zip(Q, truth):
            t0 = time.perf_counter()
            got = st.retrieve(q, T, k, w, refresh=False)
            tms.append(time.perf_counter() - t0)
            res.append(recall(got, tr))
        emit(n=n, method="exact scan, fitted curve", param=f"R={R}", recall=float(np.mean(res)),
             ms=1e3 * float(np.median(tms)), build_s=0.0, extra="")
        del st
        gc.collect()
        # the folded store
        for W in ((24.0, 6.0) if n >= 1_000_000 else (24.0,)):
            for RR in ((4, 6, 8, 10) if n < 1_000_000 else (R,)):
                t0 = time.perf_counter()
                st = CurveFolded(d, curve=ACT_R, R=RR, recent_hours=W, capacity=n, threads=threads)
                st.load(E, imp, hist, T)
                build = time.perf_counter() - t0
                for seg in st.segments:
                    seg["h"].index.set_num_threads(1)
                efs = (32, 64, 128, 256) if RR == R else (128,)
                for ef in efs:
                    for seg in st.segments:
                        seg["h"].ef = ef
                    res, tms = [], []
                    for q, tr in zip(Q, truth):
                        t1 = time.perf_counter()
                        got = st.retrieve(q, T, k, w, refresh=False)
                        tms.append(time.perf_counter() - t1)
                        res.append(recall(got, tr))
                    emit(n=n, method="folded (ours)", param=f"R={RR},ef={ef},W={W:g}h", recall=float(np.mean(res)),
                         ms=1e3 * float(np.median(tms)), build_s=build,
                         extra=f"recent part {len(st.recent):,} memories")
                del st
                gc.collect()
        # overfetch and relevance only, on one similarity index
        t0 = time.perf_counter()
        st = CurveOverfetch(d, curve=ACT_R, R=R, fetch=100, capacity=n, threads=threads)
        st.load(E, imp, hist, T)
        st.h.index.set_num_threads(1)
        build = time.perf_counter() - t0
        exact_ranges = [st.fit_ranges(q, T) for q in Q]        # given to it for free
        for K in (100, 1000):
            st.fetch = K
            st.h.ef = max(64, K)
            res, tms = [], []
            for q, tr, rg in zip(Q, truth, exact_ranges):
                t1 = time.perf_counter()
                got = st.search_with_ranges(q, T, k, w, rg)
                tms.append(time.perf_counter() - t1)
                res.append(recall(got, tr))
            emit(n=n, method="overfetch + re-rank", param=f"K={K}", recall=float(np.mean(res)),
                 ms=1e3 * float(np.median(tms)), build_s=build, extra="exact normalisation ranges given")
        st.h.ef = 64
        res = [recall(st.h.query(q[None, :], k), tr) for q, tr in zip(Q, truth)]
        emit(n=n, method="relevance only", param="", recall=float(np.mean(res)), ms=float("nan"), build_s=build, extra="")
        del st, E, hist
        gc.collect()


def part_fronts(sizes, R: int, W: float = 24.0) -> None:
    """Size of the Pareto fronts of the stored z vectors in the bulk-loaded segment (no index needed)."""
    import json
    from agentmem.curve_stores import pareto_max

    out = {}
    for n in sizes:
        E, imp, hist, Q, T = histories_dataset(n, 1)
        st = CurveExactFit(E.shape[1], curve=ACT_R, R=R, capacity=n)
        st.load(E, imp, hist, T)
        old = np.nonzero(st.t_last[:n] < T - W)[0]
        anchor = float(st.t_last[old].max())
        z = st.y[old] * np.exp(-np.outer(anchor - st.t_last[old], st.lam)) * st.wr
        t0 = time.perf_counter()
        hi, lo = pareto_max(z), pareto_max(-z)
        out[str(n)] = dict(segment=int(len(old)), front_max=int(len(hi)), front_min=int(len(lo)),
                           seconds=time.perf_counter() - t0)
        print(n, out[str(n)], flush=True)
        del st, E, hist
        gc.collect()
    (RESULTS / "curve_fronts.json").write_text(json.dumps(out, indent=1))


def part_ranges(n: int, nq: int = 200, ef: int = 64) -> None:
    """How much of the folded store's loss comes from the estimated relevance range: the same
    queries with the exact ranges (an O(n) scan, for the analysis only)."""
    import json

    E, imp, hist, Q, T = histories_dataset(n, nq)
    w = PROFILES["paper"]
    ex = CurveExact(E.shape[1], capacity=n)
    ex.load(E, imp, hist, T)
    truth = [set(np.argpartition(-ex.exact(q, T, w), 10)[:10].tolist()) for q in Q]
    del ex
    st = CurveFolded(E.shape[1], curve=ACT_R, R=8, recent_hours=24.0, capacity=n, threads=2)
    st.load(E, imp, hist, T)
    for seg in st.segments:
        seg["h"].ef = ef
        seg["h"].index.set_num_threads(1)
    est, exa = [], []
    estimated_ranges = st._ranges
    for q, tr in zip(Q, truth):
        est.append(recall(st.retrieve(q, T, 10, w, refresh=False), tr))
        rng = (float(np.ptp(st.D_fit(T))), st.imp_hi - st.imp_lo, float(np.ptp(E @ q)))
        st._ranges = lambda *_a, r=rng: r
        exa.append(recall(st.retrieve(q, T, 10, w, refresh=False), tr))
        st._ranges = estimated_ranges
    out = dict(n=n, ef=ef, estimated=float(np.mean(est)), exact_ranges=float(np.mean(exa)))
    print(out, flush=True)
    (RESULTS / "curve_ranges.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["fit", "static", "fronts", "ranges"], default="fit")
    ap.add_argument("--sizes", type=int, nargs="+", default=[100_000])
    ap.add_argument("--nq", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=str(RESULTS / "static_curves.csv"))
    a = ap.parse_args()
    if a.part == "fit":
        part_fit()
    elif a.part == "fronts":
        part_fronts(a.sizes, a.R)
    elif a.part == "ranges":
        part_ranges(a.sizes[0], a.nq)
    else:
        part_static(a.sizes, a.nq, a.k, a.R, Path(a.out), a.threads)


if __name__ == "__main__":
    main()

"""Reproduce the parameter study of the HNSW paper with a from-scratch implementation.

Experiments (names refer to figures of Malkov & Yashunin, arXiv:1603.09320):

  validate  our implementation vs hnswlib (the authors' C++ library) on GloVe-100
            (angular), same M and efConstruction: recall@10 for a sweep of ef
  fig3      uniform random d=4 (paper: 10M points), M=6, 1-NN: cost at recall 0.9
            as a function of mL, simple vs heuristic neighbour selection
  fig4      uniform random d=1024, 100k points, M=20, 1-NN: same as fig3
  fig7      d=10 (paper: 10M points), uniform vs 100 isolated clusters, M=16,
            10-NN: recall vs cost, simple vs heuristic selection
  glove     GloVe-100 angular (real data, in place of the paper's SIFT), M=16,
            10-NN: flat (mL=0) vs hierarchical, simple vs heuristic

Cost is the number of distance evaluations per query, which does not depend
on the machine or on other processes. Output: results/hnsw_<exp>.csv

    python -m experiments.exp_hnsw_ablation validate fig3 fig4 fig7 glove
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import faiss
import numpy as np

from agentmem.paper_hnsw import IP, L2, PaperHNSW

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EF_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
           1536, 2048, 3072, 4096]


# ------------------------------------------------------------------ data

def ground_truth(X: np.ndarray, Q: np.ndarray, k: int, metric: int) -> np.ndarray:
    index = faiss.IndexFlatL2(X.shape[1]) if metric == L2 else faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    _, I = index.search(Q, k)
    return I


def uniform(n: int, d: int, nq: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.random((n, d), dtype=np.float32), rng.random((nq, d), dtype=np.float32)


def isolated_clusters(n: int, d: int, nq: int, n_clusters: int = 100, seed: int = 0):
    """Gaussian blobs whose centres are far apart compared with their spread."""
    rng = np.random.default_rng(seed)
    centres = rng.random((n_clusters, d), dtype=np.float32) * 100.0
    lab = rng.integers(n_clusters, size=n + nq)
    P = centres[lab] + rng.normal(size=(n + nq, d)).astype(np.float32)
    return P[:n], P[n:]


def glove(n: int | None, nq: int, seed: int = 0):
    """GloVe Twitter 100-d (the vectors of ANN-Benchmarks' glove-100-angular), unit-normalized."""
    from agentmem.embed import load_glove
    _, V = load_glove("glove-twitter-100")
    V = np.asarray(V, dtype=np.float32)
    nrm = np.linalg.norm(V, axis=1)
    V = V[nrm > 0] / nrm[nrm > 0, None]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(V))
    Q = V[perm[:nq]]
    base = V[perm[nq:]] if n is None else V[perm[nq:nq + n]]
    return np.ascontiguousarray(base), np.ascontiguousarray(Q)


# ------------------------------------------------------------------ helpers

def recall(found: np.ndarray, gt: np.ndarray) -> float:
    k = gt.shape[1]
    return float(np.mean([len(set(f[:k]) & set(g)) / k for f, g in zip(found, gt)]))


def sweep(h: PaperHNSW, Q: np.ndarray, gt: np.ndarray, k: int, stop_at: float = 0.97):
    rows = []
    h.search(Q[:2], k, 1)   # compile / warm up
    for ef in EF_GRID:
        if ef < k and k > 1:
            continue
        out, dist, qps = h.search(Q, k, ef)
        r = recall(out, gt)
        rows.append(dict(ef=ef, recall=r, dist_evals=dist, qps=qps))
        if r >= stop_at:
            break
    return rows


def cost_at(rows: list[dict], target: float) -> float:
    """Distance evaluations needed for `target` recall (log-linear interpolation)."""
    prev = None
    for r in rows:
        if r["recall"] >= target:
            if prev is None or prev["recall"] >= target:
                return r["dist_evals"]
            f = (target - prev["recall"]) / (r["recall"] - prev["recall"])
            return math.exp(math.log(prev["dist_evals"]) + f * (math.log(r["dist_evals"]) - math.log(prev["dist_evals"])))
        prev = r
    return float("nan")


def write(name: str, rows: list[dict]) -> None:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"hnsw_{name}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path.relative_to(ROOT)} ({len(rows)} rows)", flush=True)


def mL_study(name, X, Q, k, metric, M, efc, mls, target):
    gt = ground_truth(X, Q, k, metric)
    rows = []
    for heuristic in (False, True):
        for mL in mls:
            h = PaperHNSW(M=M, ef_construction=efc, mL=mL, heuristic=heuristic, metric=metric).build(X)
            sw = sweep(h, Q, gt, k)
            c = cost_at(sw, target)
            st = h.degree_stats()
            print(f"  {name} {'heuristic' if heuristic else 'simple   '} mL={mL:.3f} levels={st['max_level']:2d} "
                  f"build={h.build_s:6.1f}s cost@{target}={c:8.1f}", flush=True)
            for r in sw:
                rows.append(dict(dataset=name, n=len(X), d=X.shape[1], M=M, efc=efc, mL=mL,
                                 heuristic=int(heuristic), cost_at_target=c, target=target,
                                 build_s=h.build_s, build_dist=h.build_dist, max_level=st["max_level"],
                                 mean_degree0=st["mean_degree0"], **r))
    write(name, rows)


# ------------------------------------------------------------------ experiments

def exp_validate(args):
    import hnswlib
    X, Q = glove(args.glove_n, args.nq)
    k, M, efc = 10, 16, 200
    gt = ground_truth(X, Q, k, IP)
    ours = PaperHNSW(M=M, ef_construction=efc, metric=IP).build(X)
    ref = hnswlib.Index(space="ip", dim=X.shape[1])
    ref.init_index(max_elements=len(X), ef_construction=efc, M=M, random_seed=0)
    ref.set_num_threads(1)
    t0 = time.perf_counter()
    ref.add_items(X)
    ref_build = time.perf_counter() - t0
    print(f"  build: ours {ours.build_s:.1f}s, hnswlib {ref_build:.1f}s (1 thread)", flush=True)
    rows = []
    ours.search(Q[:2], k, 10)
    for ef in [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]:
        out, dist, qps = ours.search(Q, k, ef)
        ref.set_ef(ef)
        t0 = time.perf_counter()
        lab, _ = ref.knn_query(Q, k=k)
        qps_ref = len(Q) / (time.perf_counter() - t0)
        rows.append(dict(n=len(X), ef=ef, recall_ours=recall(out, gt), recall_hnswlib=recall(lab, gt),
                         dist_evals_ours=dist, qps_ours=qps, qps_hnswlib=qps_ref,
                         build_s_ours=ours.build_s, build_s_hnswlib=ref_build))
        print(f"  ef={ef:4d} recall ours={rows[-1]['recall_ours']:.4f} hnswlib={rows[-1]['recall_hnswlib']:.4f} "
              f"qps ours={qps:7.0f} hnswlib={qps_ref:7.0f}", flush=True)
    write("validate", rows)


def exp_fig3(args):
    X, Q = uniform(args.n4, 4, args.nq)
    ml0 = 1 / math.log(6)
    mL_study("fig3_d4", X, Q, 1, L2, M=6, efc=100,
             mls=[0.0, 0.1, 0.2, 0.3, 0.4, ml0, 0.8, 1.2], target=0.9)


def exp_fig4(args):
    X, Q = uniform(args.n1024, 1024, args.nq1024)
    mL_study("fig4_d1024", X, Q, 1, L2, M=20, efc=100,
             mls=[0.0, 1 / math.log(20), 0.6, 1.0], target=0.9)


def exp_fig7(args):
    rows = []
    for name, (X, Q) in (("uniform", uniform(args.n10, 10, args.nq)),
                         ("100 clusters", isolated_clusters(args.n10, 10, args.nq))):
        gt = ground_truth(X, Q, 10, L2)
        for heuristic in (False, True):
            h = PaperHNSW(M=16, ef_construction=100, heuristic=heuristic, metric=L2).build(X)
            sw = sweep(h, Q, gt, 10, stop_at=0.9999)
            print(f"  fig7 {name:12s} {'heuristic' if heuristic else 'simple   '} build={h.build_s:6.1f}s "
                  f"best recall={max(r['recall'] for r in sw):.4f} cost@0.9={cost_at(sw, 0.9):8.1f}", flush=True)
            for r in sw:
                rows.append(dict(dataset=name, n=len(X), heuristic=int(heuristic), build_s=h.build_s, **r))
    write("fig7_d10", rows)


def exp_glove(args):
    X, Q = glove(args.glove_n, args.nq)
    mL_study("glove", X, Q, 10, IP, M=16, efc=200, mls=[0.0, 1 / math.log(16)], target=0.9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+", choices=["validate", "fig3", "fig4", "fig7", "glove"])
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--nq1024", type=int, default=300)
    ap.add_argument("--n4", type=int, default=1_000_000)
    ap.add_argument("--n10", type=int, default=1_000_000)
    ap.add_argument("--n1024", type=int, default=100_000)
    ap.add_argument("--glove-n", type=int, default=500_000)
    args = ap.parse_args()
    for e in args.experiments:
        print(f"== {e}", flush=True)
        globals()[f"exp_{e}"](args)


if __name__ == "__main__":
    main()

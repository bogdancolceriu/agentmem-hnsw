"""Static retrieval benchmark: how well and how fast does each store return the
top-k memories under the Generative-Agents score?

Datasets
  glove   GloVe Twitter 100-d unit vectors (the vectors of ANN-Benchmarks'
          glove-100-angular) as memory embeddings, with synthetic agent
          metadata: creation times uniform over a 90-day horizon, 5% of the
          memories re-accessed later, importance drawn from the distribution
          of the Willowbrook memory streams. Queries are held-out vectors.
  town    the pooled memory streams of the 25 Willowbrook agents after a
          90-day run (real sentences, SIF embeddings, simulated importance
          and access times); queries are the focal points the agents use.

Methods (all single-threaded at query time)
  exact            numpy scan over all memories (ground truth)
  relonly          hnswlib top-k by cosine similarity only
  rerank-K         hnswlib top-K by similarity + k newest, re-ranked by the full score
  folded-ip        ours: full score folded into an inner-product hnswlib index
  folded-l2        ours with the classic MIPS -> L2 reduction
  folded-ivf       ours on a FAISS IVF-Flat inner-product index (index-agnostic)
  folded-s=...     construction-scale ablation; folded-oldanchor: anchor at the
                   oldest access time instead of the newest

Output: results/static_<dataset>.csv
    python -m experiments.exp_static --dataset glove --sizes 10000 100000 1000000
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import time
from pathlib import Path

import faiss
import numpy as np

from agentmem import PROFILES, FoldedHNSW, OverfetchRerank, Weights, exact_scores, topk
from agentmem.score import fold_memories, fold_query, ranges, recency

faiss.omp_set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
HORIZON_H = 90 * 24.0


# ------------------------------------------------------------------ data

def town_importance_distribution() -> tuple[np.ndarray, np.ndarray]:
    """Empirical importance values of the Willowbrook memory streams (cached)."""
    path = ROOT / "data" / "cache" / "town_importance.npy"
    if path.exists():
        v = np.load(path)
    else:
        from town.sim import SimConfig, Town
        town = Town(SimConfig(days=14, party_day=12, seed=123, measure_recall=False))
        town.run()
        v = np.concatenate([a.store.imp[: a.store.n] for a in town.agents])
        np.save(path, v)
    vals, counts = np.unique(np.round(v, 1), return_counts=True)
    return vals, counts / counts.sum()


def glove_dataset(n: int, nq: int, seed: int = 0):
    from agentmem.embed import load_glove
    _, V = load_glove("glove-twitter-100")
    V = np.asarray(V, dtype=np.float32)
    nrm = np.linalg.norm(V, axis=1)
    V = V[nrm > 0] / nrm[nrm > 0, None]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(V))
    Qv = np.ascontiguousarray(V[perm[:nq]])
    E = np.ascontiguousarray(V[perm[nq:nq + n]])
    t_created = np.sort(rng.uniform(0.0, HORIZON_H, size=n))
    t_last = t_created.copy()
    re = rng.random(n) < 0.05
    t_last[re] = rng.uniform(t_created[re], HORIZON_H)
    vals, p = town_importance_distribution()
    imp = rng.choice(vals, size=n, p=p)
    T = np.full(nq, HORIZON_H + 0.25)
    return E, imp, t_last, Qv, T


def town_dataset(nq: int, seed: int = 0, days: int = 92):
    path = ROOT / "data" / "cache" / f"town_pool_{days}d_s{seed}.npz"
    if not path.exists():
        from town.sim import SimConfig, Town
        from town.world import date_str
        town = Town(SimConfig(days=days, party_day=days - 1, seed=seed, measure_recall=False))
        town.run()
        E = np.concatenate([a.store.E[: a.store.n] for a in town.agents])
        imp = np.concatenate([a.store.imp[: a.store.n] for a in town.agents])
        t_last = np.concatenate([a.store.t_last[: a.store.n] for a in town.agents])
        t_created = np.concatenate([a.store.t_created[: a.store.n] for a in town.agents])
        order = np.argsort(t_created, kind="stable")   # insertion order = creation order
        focal = []
        for a in town.agents:
            for d in range(days - 7, days):
                focal += [f"{a.name}'s plan for {date_str(d)}.", f"Important recent events for {a.name}'s life.",
                          a.name]
        Qv = town.enc.encode(focal)
        np.savez(path, E=E[order], imp=imp[order], t_last=t_last[order], Q=Qv,
                 T=np.full(len(focal), days * 24.0))
    z = np.load(path)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(z["Q"]), size=min(nq, len(z["Q"])), replace=False)
    return z["E"], z["imp"], z["t_last"], z["Q"][pick], z["T"][pick]


# ------------------------------------------------------------------ helpers

def _norm(x, lo, hi):
    return (x - lo) / (hi - lo) if hi > lo else np.full(np.shape(x), 0.5)


def _combine(rec, im, rel, w, st):
    if w.minmax:
        rec, im, rel = _norm(rec, st[0], st[1]), _norm(im, st[2], st[3]), _norm(rel, st[4], st[5])
    return w.recency * rec + w.importance * im + w.relevance * rel


class Truth:
    """Exact top-k threshold and normalization statistics per query, without storing n scores."""

    def __init__(self, E, imp, t_last, Q, T, W: dict, k: int, decay: float = 0.995):
        self.E, self.imp, self.t_last, self.Q, self.T, self.decay, self.k = E, imp, t_last, Q, T, decay, k
        self.kth = {p: np.empty(len(Q)) for p in W}
        self.stats = []
        for i in range(len(Q)):
            rec = recency(t_last, T[i], decay)
            rel = (E @ Q[i]).astype(np.float64)
            st = (rec.min(), rec.max(), imp.min(), imp.max(), rel.min(), rel.max())
            self.stats.append(st)
            for p, w in W.items():
                sc = _combine(rec, imp, rel, w, st)
                self.kth[p][i] = np.partition(sc, len(sc) - k)[len(sc) - k]

    def recall(self, p, w, i, ids) -> float:
        ids = np.asarray(ids[: self.k], dtype=np.int64)
        rec = recency(self.t_last[ids], self.T[i], self.decay)
        rel = (self.E[ids] @ self.Q[i]).astype(np.float64)
        sc = _combine(rec, self.imp[ids], rel, w, self.stats[i])
        return float(np.sum(sc >= self.kth[p][i] - 1e-6)) / self.k


def evaluate(name, param, search, Q, T, truth, p, w, k, n_time=200):
    """Recall over all queries, latency over the first n_time queries (after a warm-up)."""
    for i in range(min(5, len(Q))):
        search(Q[i], T[i], k, w)
    rec, times = [], []
    for i in range(len(Q)):
        t0 = time.perf_counter()
        ids = search(Q[i], T[i], k, w)
        dt = time.perf_counter() - t0
        if i < n_time:
            times.append(dt)
        rec.append(truth.recall(p, w, i, ids))
    return dict(method=name, param=param, profile=p, recall=float(np.mean(rec)),
                recall_p10=float(np.percentile(rec, 10)), ms=1000 * float(np.mean(times)) if times else float("nan"))


class FoldedIVF:
    """The folded score on a FAISS IVF-Flat inner-product index (static, single anchor)."""

    def __init__(self, E, imp, t_last, decay, nlist):
        self.E, self.imp, self.t_last, self.decay = E, imp, t_last, decay
        self.anchor = float(t_last.max())
        X = fold_memories(E, imp, t_last, self.anchor, decay)
        q = faiss.IndexFlatIP(X.shape[1])
        self.index = faiss.IndexIVFFlat(q, X.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
        rng = np.random.default_rng(0)
        self.index.train(X[rng.choice(len(X), size=min(len(X), 50 * nlist), replace=False)])
        self.index.add(X)
        self.t_lo, self.t_hi = float(t_last.min()), float(t_last.max())
        self.imp_range = float(imp.max() - imp.min())

    def search(self, q, T, k, w):
        if w.minmax:
            pad = np.zeros((2, 2), dtype=np.float32)
            Qs = np.hstack([np.stack([q, -q]), pad]).astype(np.float32)
            _, I = self.index.search(Qs, 1)
            r = self.E[I[:, 0]] @ q
            rec_range = self.decay ** (T - self.t_hi) - self.decay ** (T - self.t_lo)
            w = w.raw(rec_range, self.imp_range, float(r[0] - r[1]))
        Q = fold_query(q, T, self.anchor, w, self.decay)
        _, I = self.index.search(Q[None, :], k)
        cand = I[0][I[0] >= 0]
        s = exact_scores(self.E[cand], self.imp[cand], self.t_last[cand], q, T, w, self.decay)
        return topk(s, k, ids=cand)


def build_folded(E, imp, t_last, threads, **kw):
    n, d = E.shape
    st = FoldedHNSW(d, buffer_size=n + 1, threads=threads, capacity=n, **kw)
    t0 = time.perf_counter()
    if kw.get("single_anchor"):
        first = int(np.argmin(t_last))
        rest = np.setdiff1d(np.arange(n), [first])
        order = np.concatenate([[first], rest])
        st.buffer_size = 1
        st.add(E[order[:1]], imp[order[:1]], t_last[order[:1]])
        st.buffer_size = n + 1
        st.add(E[order[1:]], imp[order[1:]], t_last[order[1:]])
        st.flush(float(t_last.max()))
        st.id_map = order
    else:
        st.add(E, imp, t_last)
        st.flush(float(t_last.max()))
        st.id_map = None
    for seg in st.segments:
        seg["h"].index.set_num_threads(1)
    return st, time.perf_counter() - t0


def set_ef(st, ef):
    for seg in st.segments:
        seg["h"].ef = ef
        seg["h"].index.set_ef(ef)


def search_fn(st):
    if getattr(st, "id_map", None) is not None:
        return lambda q, T, k, w: st.id_map[st._search(q, float(T), k, w)]
    return lambda q, T, k, w: st._search(q, float(T), k, w)


# ------------------------------------------------------------------ main

def run(dataset: str, n: int, nq: int, k: int, profiles: list[str], threads: int, out_rows: list,
        full: bool) -> None:
    if dataset == "glove":
        E, imp, t_last, Q, T = glove_dataset(n, nq)
    else:
        E, imp, t_last, Q, T = town_dataset(nq)
        n = len(E)
    d = E.shape[1]
    print(f"\n== {dataset} n={n} d={d} queries={len(Q)} k={k}", flush=True)
    W = {p: PROFILES[p] for p in profiles}
    t0 = time.perf_counter()
    truth = Truth(E, imp, t_last, Q, T, W, k)
    print(f"  ground truth: {time.perf_counter() - t0:.0f}s", flush=True)

    def add(row, method_build_s=0.0, mb=0.0, **extra):
        row.update(dataset=dataset, n=n, k=k, build_s=method_build_s, index_mb=mb, **extra)
        out_rows.append(row)
        print(f"  {row['profile']:6s} {row['method']:16s} {str(row['param']):>6s} recall={row['recall']:.4f} "
              f"(p10 {row['recall_p10']:.3f}) {row['ms']:8.3f} ms", flush=True)

    # exact scan
    for p, w in W.items():
        r = evaluate("exact", "-", lambda q, TT, kk, ww: topk(exact_scores(E, imp, t_last, q, TT, ww), kk),
                     Q, T, truth, p, w, k, n_time=50)
        add(r, mb=E.nbytes / 1e6)

    # relevance index shared by relonly and rerank
    st = OverfetchRerank(d, M=16, ef_construction=200, threads=threads, capacity=n)
    t0 = time.perf_counter()
    st.add(E, imp, t_last)
    b_rel = time.perf_counter() - t0
    st.h.index.set_num_threads(1)
    mb_rel = st.memory_bytes() / 1e6
    for p, w in W.items():
        for ef in ([16, 64, 256] if full else [64]):
            st.h.ef = ef
            r = evaluate("relonly", ef, lambda q, TT, kk, ww: st.h.query(q[None, :], kk), Q, T, truth, p, w, k)
            add(r, b_rel, mb_rel)
        st.h.ef = 64
        for K in [10, 50, 100, 500, 1000, 5000]:
            if K > n:
                continue
            st.fetch = K
            r = evaluate("rerank", K, lambda q, TT, kk, ww: st._search(q, float(TT), kk, ww), Q, T, truth, p, w, k)
            add(r, b_rel, mb_rel)
        if not w.minmax:
            st.fetch, st.faithful = 100, True
            r = evaluate("rerank-faithful", 100, lambda q, TT, kk, ww: st._search(q, float(TT), kk, ww),
                         Q, T, truth, p, w, k)
            add(r, b_rel, mb_rel)
            st.faithful = False
    del st

    # ours: folded score in an inner-product HNSW index
    variants = [("folded-ip", dict(space="ip"))]
    if full:
        variants += [("folded-l2", dict(space="l2"))]
    for name, kw in variants:
        fst, b = build_folded(E, imp, t_last, threads, M=16, ef_construction=200, **kw)
        mb = fst.memory_bytes() / 1e6
        for p, w in W.items():
            for ef in [16, 32, 64, 128, 256, 512]:
                set_ef(fst, ef)
                r = evaluate(name, ef, search_fn(fst), Q, T, truth, p, w, k)
                add(r, b, mb)
            if name == "folded-ip" and w.minmax:
                # same index and depth, but with the exact normalization ranges (an O(n) scan, analysis only)
                set_ef(fst, 64)
                exact_norm = lambda q, TT, kk, ww: fst._search(q, float(TT), kk, ww.raw(*ranges(E, imp, t_last, q, TT)))
                r = evaluate("folded-ip-exactnorm", 64, exact_norm, Q, T, truth, p, w, k, n_time=0)
                add(r, b, mb)
        del fst

    if not full:
        return
    # construction-scale ablation and anchor ablation (paper profile only)
    p = "paper" if "paper" in W else profiles[0]
    w = W[p]
    for s in [0.25, 0.5, 2.0, 4.0]:
        fst, b = build_folded(E, imp, t_last, threads, M=16, ef_construction=200, s=s)
        for ef in [32, 64, 128]:
            set_ef(fst, ef)
            r = evaluate(f"folded-s={s:g}", ef, search_fn(fst), Q, T, truth, p, w, k)
            add(r, b, fst.memory_bytes() / 1e6)
        del fst
    fst, b = build_folded(E, imp, t_last, threads, M=16, ef_construction=200, single_anchor=True)
    for ef in [32, 64, 128, 512]:
        set_ef(fst, ef)
        r = evaluate("folded-oldanchor", ef, search_fn(fst), Q, T, truth, p, w, k)
        add(r, b, fst.memory_bytes() / 1e6)
    del fst
    # index-agnostic: FAISS IVF-Flat on the folded vectors
    nlist = int(4 * np.sqrt(n))
    t0 = time.perf_counter()
    ivf = FoldedIVF(E, imp, t_last, 0.995, nlist)
    b = time.perf_counter() - t0
    for pp, ww in W.items():
        for nprobe in [1, 4, 16, 64]:
            ivf.index.nprobe = nprobe
            r = evaluate("folded-ivf", nprobe, ivf.search, Q, T, truth, pp, ww, k)
            add(r, b, (n * (d + 2) * 4) / 1e6)


def rel_sweep(n: int, nq: int, k: int, threads: int, out: Path) -> None:
    """Recall as the weight of relevance grows (recency = importance = 1, min-max normalized)."""
    E, imp, t_last, Q, T = glove_dataset(n, nq)
    ws = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
    W = {f"rel={w:g}": Weights(1.0, 1.0, float(w), minmax=True) for w in ws}
    truth = Truth(E, imp, t_last, Q, T, W, k)
    st = OverfetchRerank(E.shape[1], M=16, ef_construction=200, threads=threads, capacity=n)
    st.add(E, imp, t_last)
    st.h.index.set_num_threads(1)
    folded = {sv: build_folded(E, imp, t_last, threads, M=16, ef_construction=200, s=sv)[0] for sv in (1.0, 0.5)}
    rows = []
    for p, w in W.items():
        start = len(rows)
        rows.append(evaluate("relonly", 64, lambda q, TT, kk, ww: st.h.query(q[None, :], kk), Q, T, truth, p, w, k))
        for K in (100, 1000):
            st.fetch = K
            rows.append(evaluate("rerank", K, lambda q, TT, kk, ww: st._search(q, float(TT), kk, ww),
                                 Q, T, truth, p, w, k))
        for sv, fst in folded.items():
            for ef in (64, 256):
                set_ef(fst, ef)
                rows.append(evaluate(f"folded-s={sv:g}", ef, search_fn(fst), Q, T, truth, p, w, k))
        for r in rows[start:]:
            r.update(dataset="glove", n=n, k=k, rel_weight=w.relevance)
            print(f"  {p:8s} {r['method']:12s} {str(r['param']):>5s} recall={r['recall']:.4f} {r['ms']:.3f} ms", flush=True)
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["glove", "town"], default="glove")
    ap.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--profiles", nargs="+", default=["paper", "code", "raw"])
    ap.add_argument("--threads", type=int, default=2, help="threads for index construction only")
    ap.add_argument("--full-at", type=int, nargs="*", default=[100_000],
                    help="sizes at which to also run the ablations (L2, s, anchor, IVF)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rel-sweep", type=int, default=0, help="run only the relevance-weight sweep at this size")
    args = ap.parse_args()
    if args.rel_sweep:
        rel_sweep(args.rel_sweep, args.nq, args.k, args.threads, Path(args.out or RESULTS / "static_relsweep.csv"))
        return
    out = Path(args.out or RESULTS / f"static_{args.dataset}_k{args.k}.csv")
    rows: list[dict] = []
    sizes = args.sizes if args.dataset == "glove" else [0]
    for n in sizes:
        run(args.dataset, n, args.nq, args.k, args.profiles, args.threads, rows,
            full=(n in args.full_at) or args.dataset == "town")
        out.parent.mkdir(exist_ok=True)
        with out.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["dataset", "n", "k", "profile", "method", "param", "recall",
                                               "recall_p10", "ms", "build_s", "index_mb"])
            wr.writeheader()
            wr.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

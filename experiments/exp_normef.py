"""How much of the folded store's recall loss comes from estimating the relevance range?

The min-max normalization needs the largest and smallest relevance of the query
over all memories. The store estimates them with two small HNSW searches (for q
and for -q) of depth norm_ef. This script builds the folded index once per data
set and compares norm_ef settings with the exact ranges (an O(n) scan).

    python -m experiments.exp_normef      -> results/static_normef.csv
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
from pathlib import Path

from agentmem import PROFILES
from agentmem.score import ranges
from experiments.exp_static import Truth, build_folded, evaluate, glove_dataset, search_fn, set_ef, town_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = []
    for name, data in (("glove", lambda: glove_dataset(1_000_000, 1000)), ("town", lambda: town_dataset(1000))):
        E, imp, t_last, Q, T = data()
        W = {p: PROFILES[p] for p in ("paper", "code")}
        truth = Truth(E, imp, t_last, Q, T, W, 10)
        fst, _ = build_folded(E, imp, t_last, 2, M=16, ef_construction=200)
        for ef in (64, 128):
            set_ef(fst, ef)
            for p, w in W.items():
                for ne in (32, 64, 128, 256):
                    fst.norm_ef = ne
                    r = evaluate("folded-ip", ef, search_fn(fst), Q, T, truth, p, w, 10)
                    r.update(dataset=name, n=len(E), norm_ef=ne)
                    rows.append(r)
                    print(f"  {name} ef={ef} {p:5s} norm_ef={ne:<4d} recall={r['recall']:.4f} {r['ms']:.3f} ms", flush=True)
                exact = lambda q, TT, kk, ww: fst._search(q, float(TT), kk, ww.raw(*ranges(E, imp, t_last, q, TT)))
                r = evaluate("folded-ip", ef, exact, Q, T, truth, p, w, 10, n_time=0)
                r.update(dataset=name, n=len(E), norm_ef="exact")
                rows.append(r)
                print(f"  {name} ef={ef} {p:5s} exact ranges   recall={r['recall']:.4f}", flush=True)
        del fst
    out = ROOT / "results" / "static_normef.csv"
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

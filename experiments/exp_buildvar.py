"""How much does recall vary from one build of the folded index to the next?

hnswlib builds with two threads, so the insertion order and therefore the graph
change from run to run. This script builds the folded index on the pooled town
memories five times and evaluates each build in the same way as exp_static.py.

    python -m experiments.exp_buildvar      -> results/static_town_builds.csv
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
from pathlib import Path

from agentmem import PROFILES
from experiments.exp_static import Truth, build_folded, evaluate, search_fn, set_ef, town_dataset

ROOT = Path(__file__).resolve().parents[1]


def main(n_builds: int = 5) -> None:
    E, imp, t_last, Q, T = town_dataset(1000)
    W = {p: PROFILES[p] for p in ("paper", "code", "raw")}
    truth = Truth(E, imp, t_last, Q, T, W, 10)
    rows = []
    for b in range(n_builds):
        fst, build_s = build_folded(E, imp, t_last, 2, M=16, ef_construction=200)
        for ef in (64, 256):
            set_ef(fst, ef)
            for p, w in W.items():
                r = evaluate("folded-ip", ef, search_fn(fst), Q, T, truth, p, w, 10)
                r.update(dataset="town", n=len(E), build=b, build_s=build_s)
                rows.append(r)
                print(f"  build {b} ef={ef:<3d} {p:5s} recall={r['recall']:.4f} {r['ms']:.3f} ms", flush=True)
        del fst
    out = ROOT / "results" / "static_town_builds.csv"
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

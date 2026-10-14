"""Time of an exact k-NN scan on GloVe, for comparison with the HNSW timings of the validation run.

Same vectors and queries as `exp_hnsw_ablation validate` (GloVe Twitter 100-d, unit length),
one thread, k = 10: a matrix-vector product and a partial sort per query.

    python -m experiments.time_exact_knn      -> results/exact_knn_timing.json
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import time
from pathlib import Path

import numpy as np

from experiments.exp_hnsw_ablation import glove

ROOT = Path(__file__).resolve().parents[1]


def main(nq: int = 1000, k: int = 10) -> None:
    out = {}
    for n in (500_000, 1_000_000):
        X, Q = glove(n, nq)
        X = np.ascontiguousarray(X, dtype=np.float32)
        ts = []
        for q in Q:
            t0 = time.perf_counter()
            s = X @ q
            idx = np.argpartition(-s, k)[:k]
            idx = idx[np.argsort(-s[idx])]
            ts.append(time.perf_counter() - t0)
        ts = 1000 * np.asarray(ts)
        out[str(n)] = {"median_ms": float(np.median(ts)), "mean_ms": float(ts.mean()), "queries": nq}
        print(f"n={n}: median {np.median(ts):.2f} ms, mean {ts.mean():.2f} ms", flush=True)
    path = ROOT / "results" / "exact_knn_timing.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

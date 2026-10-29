"""Willowbrook with forgetting curves beyond the exponential.

The same party scenario as exp_town.py, with the recency term of the paper's score replaced by
ACT-R's power-law curve (d = 0.5, one-hour offset), either of the last use only or summed over
every use of a memory. Retrieval keeps the paper's min-max normalisation and equal weights.
For each score we compare an exact scan with the true curve, the folded store (eight
exponentials, recent part of 24 h, daily segments) and LangChain-style overfetch (K = 100).
Recall is measured at every retrieval against the exact scan with the true curve.

Output: results/town_curve.csv and results/town_curve_recall.jsonl. Re-running skips finished runs.

    python -m experiments.exp_town_curve --burns 90 --seeds 30 --workers 2
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from experiments.exp_town import FIELDS as BASE_FIELDS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SCORES = {"last": dict(frequency=False), "uses": dict(frequency=True)}
BACKENDS = {
    "exact": ("curve_exact", {}),
    "folded": ("curve_folded", {}),
    "rerank100": ("curve_rerank", {"fetch": 100}),
}
FIELDS = ["score"] + BASE_FIELDS

_ENC = None


def _encoder():
    global _ENC
    if _ENC is None:
        from town.sim import load_encoder
        _ENC = load_encoder()
    return _ENC


def run_one(burn: int, score: str, backend_key: str, seed: int) -> tuple[dict, dict]:
    from agentmem import PROFILES
    from town.sim import SimConfig, Town

    kind, kw = BACKENDS[backend_key]
    cfg = SimConfig(days=burn + 2, party_day=burn + 1, seed=seed, backend=kind,
                    backend_kw={**kw, **SCORES[score]}, weights=PROFILES["paper"])
    town = Town(cfg, encoder=_encoder())
    row = town.run()
    row.update(burn=burn, profile="paper", score=score, backend_key=backend_key, seed=seed)
    T = np.asarray(town.metrics["recall_T"])
    r = np.asarray(town.metrics["recall"])
    days = (T // 24).astype(int) if len(T) else np.zeros(0, int)
    per_day = [float(r[days == d].mean()) if np.any(days == d) else None for d in range(cfg.days)]
    extra = dict(burn=burn, score=score, backend_key=backend_key, seed=seed, recall_per_day=per_day,
                 mem_final=row["mean_memories"])
    return row, extra


def done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {(int(r["burn"]), r["score"], r["backend_key"], int(r["seed"])) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burns", type=int, nargs="+", default=[90])
    ap.add_argument("--scores", nargs="+", default=list(SCORES))
    ap.add_argument("--backends", nargs="+", default=list(BACKENDS))
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    out, out_rec = RESULTS / "town_curve.csv", RESULTS / "town_curve_recall.jsonl"
    done = done_keys(out)
    todo = [(b, sc, k, s) for b in a.burns for sc in a.scores for s in range(a.seeds) for k in a.backends
            if (b, sc, k, s) not in done]
    print(f"{len(todo)} runs to do ({len(done)} already in {out.name})", flush=True)
    new_file = not out.exists()
    t0 = time.time()
    with out.open("a", newline="") as f, out_rec.open("a") as fr, ProcessPoolExecutor(a.workers) as ex:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        futs = [ex.submit(run_one, *job) for job in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            row, extra = fut.result()
            w.writerow(row)
            f.flush()
            fr.write(json.dumps(extra) + "\n")
            fr.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

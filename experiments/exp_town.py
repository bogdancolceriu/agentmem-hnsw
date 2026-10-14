"""Multi-agent experiment: does the memory backend change what the town does?

For every (burn-in length, retrieval profile, backend, seed) we run the
Willowbrook simulation: the agents live `burn` days, then Maria Lopez learns
35 hours in advance that she is hosting a party (as Isabella's Valentine's
Day party in Park et al., 2023) and the information spreads through
conversations. Seeds use common random numbers, so for a given seed the
backends see the same world and diverge only through retrieval.

Output: results/town.csv (one row per run) and results/town_recall.jsonl
(per-day mean retrieval recall of every run). Re-running skips finished runs.

    python -m experiments.exp_town --burns 0 7 30 90 --seeds 30 --workers 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

BACKENDS = {
    "exact": ("exact", {}),
    "relonly": ("relonly", {}),
    "rerank100": ("rerank", {"fetch": 100}),
    "rerank500": ("rerank", {"fetch": 500}),
    "folded": ("folded", {}),
    # ablation: overfetch without the "k most recently created" candidates that LangChain adds
    "rerank100_norecent": ("rerank", {"fetch": 100, "recent": 0}),
}

FIELDS = ["burn", "profile", "backend_key", "seed", "backend", "days", "party_day", "aware",
          "aware_at_party", "planned", "attended", "present", "recalled_at_plan", "mean_memories",
          "conversations", "shared", "retrievals", "recall", "recall_conv", "recall_plan",
          "search_ms", "insert_ms", "refresh_ms", "wall_s"]

_ENC = None


def _encoder():
    global _ENC
    if _ENC is None:
        from town.sim import load_encoder
        _ENC = load_encoder()
    return _ENC


def run_one(burn: int, profile: str, backend_key: str, seed: int) -> tuple[dict, dict]:
    from agentmem import PROFILES
    from town.sim import SimConfig, Town

    kind, kw = BACKENDS[backend_key]
    cfg = SimConfig(days=burn + 2, party_day=burn + 1, seed=seed, backend=kind, backend_kw=kw,
                    weights=PROFILES[profile])
    town = Town(cfg, encoder=_encoder())
    row = town.run()
    row.update(burn=burn, profile=profile, backend_key=backend_key, seed=seed)
    T = np.asarray(town.metrics["recall_T"])
    r = np.asarray(town.metrics["recall"])
    days = (T // 24).astype(int) if len(T) else np.zeros(0, int)
    per_day = [float(r[days == d].mean()) if np.any(days == d) else None for d in range(cfg.days)]
    extra = dict(burn=burn, profile=profile, backend_key=backend_key, seed=seed,
                 recall_per_day=per_day, mem_final=row["mean_memories"])
    return row, extra


def done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {(int(r["burn"]), r["profile"], r["backend_key"], int(r["seed"])) for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--burns", type=int, nargs="+", default=[0, 7, 30, 90])
    ap.add_argument("--profiles", nargs="+", default=["paper"])
    ap.add_argument("--backends", nargs="+", default=["exact", "relonly", "rerank100", "folded"])
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--out", default=str(RESULTS / "town.csv"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec_path = out.with_name(out.stem + "_recall.jsonl")
    done = done_keys(out)
    jobs = [(b, p, k, s) for b in args.burns for p in args.profiles for s in range(args.seeds)
            for k in args.backends if (b, p, k, s) not in done]
    # longest runs first so the pool stays busy
    jobs.sort(key=lambda j: -j[0])
    print(f"{len(jobs)} runs to do ({len(done)} already in {out.name})", flush=True)
    new_file = not out.exists()
    t0 = time.time()
    with out.open("a", newline="") as f, rec_path.open("a") as fr, \
            ProcessPoolExecutor(max_workers=args.workers) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        futs = {pool.submit(run_one, *j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            row, extra = fut.result()
            w.writerow(row)
            f.flush()
            fr.write(json.dumps(extra) + "\n")
            fr.flush()
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Export step-by-step traces of the party scenario for the interactive demo.

The same town (same seed) is run with three retrieval methods. The seed is the
one whose exact-vs-relevance-only difference in attendance is closest to the
average difference over all seeds in results/town.csv, so the demo shows a
typical run rather than the most dramatic one.

    python -m experiments.export_demo --burn 30   -> results/demo_trace.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agentmem import PROFILES
from town.sim import SimConfig, Town, load_encoder
from town.world import PLACES, RESIDENTS, date_str

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"exact": ("exact", {}, "Exact score (paper)"),
           "rerank100": ("rerank", {"fetch": 100}, "Overfetch + re-rank (LangChain, K=100)"),
           "relonly": ("relonly", {}, "Relevance only (plain RAG memory)")}


def typical_seed(burn: int) -> int:
    d = pd.read_csv(ROOT / "results" / "town.csv")
    d = d[(d.burn == burn) & (d.profile == "paper")]
    p = d.pivot_table(index="seed", columns="backend_key", values="attended").dropna()
    diff = p["exact"] - p["relonly"]
    return int((diff - diff.mean()).abs().idxmin())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--burn", type=int, default=30)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "demo_trace.json"))
    args = ap.parse_args()
    seed = args.seed if args.seed is not None else typical_seed(args.burn)
    enc = load_encoder()
    party_day = args.burn + 1
    start = party_day * 24.0 + 17.0 - 35.0 - 0.25
    homes = sorted({r.home for r in RESIDENTS})
    places = list(PLACES) + homes
    out = dict(seed=seed, burn=args.burn, party_day=date_str(party_day), places=list(PLACES), homes=homes,
               agents=[dict(name=r.name, home=r.home, role=(r.role or "retired").replace("_", " ")) for r in RESIDENTS],
               start_hour=start, step_hours=0.25, runs={})
    for key, (kind, kw, label) in METHODS.items():
        cfg = SimConfig(days=party_day + 1, party_day=party_day, seed=seed, backend=kind, backend_kw=kw,
                        weights=PROFILES["paper"], measure_recall=False, trace_from=start)
        town = Town(cfg, encoder=enc)
        res = town.run()
        frames = [[[places.index(pl), k, att] for pl, k, att, _ in fr] for fr in town.frames]
        acts = [[act for _, _, _, act in fr] for fr in town.frames]
        out["runs"][key] = dict(label=label, frames=frames, activities=acts,
                                summary={x: res[x] for x in ("aware_at_party", "recalled_at_plan", "attended",
                                                             "mean_memories")})
        print(f"  {key:10s} seed {seed}: knew {res['aware_at_party']}, attended {res['attended']}, "
              f"{len(frames)} frames", flush=True)
    out["party_frame"] = int(round((party_day * 24.0 + 17.0 - start) / 0.25))
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e3:.0f} kB)")
    tpl = (ROOT / "demo" / "template.html").read_text(encoding="utf-8")
    html = tpl.replace("__TRACE__", json.dumps(out, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/"))
    page = ROOT / "demo" / "willowbrook.html"
    page.write_text(html, encoding="utf-8")
    print(f"wrote {page} ({page.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()

"""Why counting every use can lock an agent's memory in.

For a few seeds of the 92-day party scenario, the town is run with the exact scan and
ACT-R's curve in three forms: of the last use only, summed over every use (the form that
folds into an inner product) and summed with ACT-R's logarithm. At the end of each run we
record, per agent, how often its most retrieved memory was retrieved, the share of all
retrievals taken by its 1% most retrieved memories and the normalised recency term that a
memory heard one hour earlier, used once, would get against the agent's memories.

    python -m experiments.town_curve_diag --seeds 5      -> results/town_curve_diag.json
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"last": dict(frequency=False), "uses": dict(frequency=True),
            "uses_log": dict(frequency=True, log=True)}


def run(variant: str, seed: int, burn: int) -> dict:
    from agentmem import ACT_R, PROFILES
    from town.sim import SimConfig, Town, load_encoder

    cfg = SimConfig(days=burn + 2, party_day=burn + 1, seed=seed, backend="curve_exact",
                    backend_kw=VARIANTS[variant], weights=PROFILES["paper"], measure_recall=False)
    town = Town(cfg, encoder=load_encoder())
    row = town.run()
    per_agent = []
    for a in town.agents:
        st = a.store
        k = st.n_acc
        T = float(st.acc_t[:k].max())
        retrievals = np.bincount(st.acc_mem[:k], minlength=st.n) - 1      # every memory starts with its creation
        top = np.sort(retrievals)[::-1][: max(1, st.n // 100)]
        D = st.D_true(T)
        new = float(ACT_R(1.0))
        new = np.log(new) if st.log else new
        span = float(D.max() - D.min())
        per_agent.append(dict(
            memories=int(st.n), retrievals=int(retrievals.sum()), max_retrievals=int(retrievals.max()),
            top1_share=float(top.sum() / max(1, retrievals.sum())),
            new_hour=float(np.clip((new - D.min()) / span, 0.0, 1.0)) if span > 0 else 0.5,
            most_retrieved=a.text[int(np.argmax(retrievals))] if hasattr(a, "text") else ""))
    return dict(variant=variant, seed=seed, burn=burn, aware=int(row["aware_at_party"]),
                recalled=int(row["recalled_at_plan"]), attended=int(row["attended"]), agents=per_agent)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--burn", type=int, default=90)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    jobs = [(v, s, a.burn) for v in VARIANTS for s in range(a.seeds)]
    with ProcessPoolExecutor(a.workers) as ex:
        runs = list(ex.map(run, *zip(*jobs)))
    summary = {}
    for v in VARIANTS:
        rs = [r for r in runs if r["variant"] == v]
        ags = [g for r in rs for g in r["agents"]]
        summary[v] = dict(
            seeds=len(rs),
            max_retrievals=float(np.mean([g["max_retrievals"] for g in ags])),
            retrievals_per_memory=float(np.mean([g["retrievals"] / g["memories"] for g in ags])),
            top1_share=float(np.mean([g["top1_share"] for g in ags])),
            new_hour=float(np.mean([g["new_hour"] for g in ags])),
            aware=float(np.mean([r["aware"] for r in rs])), recalled=float(np.mean([r["recalled"] for r in rs])),
            attended=float(np.mean([r["attended"] for r in rs])),
            examples=[g["most_retrieved"] for g in rs[0]["agents"][:5]])
        s = summary[v]
        print(f"{v:9s} most retrieved memory: {s['max_retrievals']:.0f} retrievals, top 1% of memories: "
              f"{100 * s['top1_share']:.0f}% of retrievals, a memory one hour old gets {s['new_hour']:.3f} "
              f"of the range | knew {s['aware']:.1f}, recalled {s['recalled']:.1f}, attended {s['attended']:.1f}",
              flush=True)
    out = ROOT / "results" / "town_curve_diag.json"
    out.write_text(json.dumps(dict(burn=a.burn, summary=summary, runs=runs), indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()

"""Calibrate the two free parameters of Willowbrook against the paper.

Park et al. (2023) report that after two game days 13 of 25 agents knew about
Isabella's party and 5 of the 12 invited agents attended. With exact retrieval
(the paper's score) and a fresh town, we sweep the conversation rate and the
probability of passing on news, with the party appeal fixed at 0.45.

    python -m experiments.calibrate_town   -> results/calibration.csv
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np

from town.sim import SimConfig, Town, load_encoder

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    enc = load_encoder()
    rows = []
    for talk, share in itertools.product([0.04, 0.06, 0.08, 0.1, 0.15], [0.3, 0.5, 0.7]):
        runs = [Town(SimConfig(days=2, party_day=1, seed=s, talk_rate=talk, share_prob=share,
                               measure_recall=False), encoder=enc).run() for s in range(12)]
        aware = np.array([r["aware"] for r in runs])
        att = np.array([r["attended"] for r in runs])
        rows.append(dict(talk_rate=talk, share_prob=share, seeds=len(runs), aware_mean=aware.mean(),
                         aware_sd=aware.std(ddof=1), attended_mean=att.mean(), attended_sd=att.std(ddof=1),
                         invited_mean=(aware - 1).mean(),
                         loss=abs(aware.mean() - 13) / 13 + abs(att.mean() - 5) / 5))
        r = rows[-1]
        print(f"talk={talk:.2f} share={share:.1f} knew={r['aware_mean']:5.1f}±{r['aware_sd']:.1f} "
              f"attended={r['attended_mean']:4.1f}±{r['attended_sd']:.1f} loss={r['loss']:.3f}", flush=True)
    out = ROOT / "results" / "calibration.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    best = min(rows, key=lambda r: r["loss"])
    print(f"best: talk={best['talk_rate']} share={best['share_prob']} "
          f"(knew {best['aware_mean']:.1f}, attended {best['attended_mean']:.1f}; paper: 13 and 5)")


if __name__ == "__main__":
    main()

"""Figures for the report and the slides (PDF for LaTeX, PNG for slides).

    python -m experiments.plots            # every figure whose input CSV exists
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"

# categorical slots (validated palette, fixed order); color follows the method in every figure
C = {"ours": "#2a78d6", "rerank": "#eb6834", "relonly": "#1baf7a", "l2": "#eda100", "ivf": "#e87ba4",
     "exact": "#52514e"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE_RAMP = ["#9ec5f4", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]   # sequential steps 200..700
LABEL = {"ours": "folded HNSW (ours)", "rerank": "overfetch + rerank", "relonly": "relevance-only HNSW",
         "exact": "exact scan", "l2": "folded, MIPS→L2 reduction", "ivf": "folded, FAISS IVF"}

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": "#b9b8b3", "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.6, "lines.markersize": 4.5, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})


from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402


def _num(v, _pos=None) -> str:
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


def logaxis(ax, axis: str = "x", base: int = 10) -> None:
    """Log scale with plain tick labels (1, 2, 5 per decade) and no minor labels."""
    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log", base=base)
    if base == 10:
        a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    a.set_major_formatter(FuncFormatter(_num))
    a.set_minor_formatter(NullFormatter())


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("  figure", name)


def ci95(x) -> float:
    x = np.asarray(x, dtype=float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


# ------------------------------------------------------------------ static benchmark

def _method_key(m: str) -> str | None:
    return {"folded-ip": "ours", "rerank": "rerank", "relonly": "relonly", "exact": "exact",
            "folded-l2": "l2", "folded-ivf": "ivf"}.get(m)


def fig_static_tradeoff(csv: Path, profile: str = "paper", name: str = "static_tradeoff") -> None:
    d = pd.read_csv(csv)
    d = d[d.profile == profile]
    sizes = sorted(d.n.unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(6.6, 2.35), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, sizes):
        dn = d[d.n == n]
        for m in ["relonly", "rerank", "folded-ip", "exact"]:
            dm = dn[dn.method == m].copy()
            if dm.empty:
                continue
            key = _method_key(m)
            if m == "exact":
                ax.plot(dm.ms, dm.recall, "s", color=C[key], label=LABEL[key], markersize=5)
                continue
            dm["p"] = pd.to_numeric(dm.param)
            dm = dm.sort_values("p")
            ax.plot(dm.ms, dm.recall, "-o", color=C[key], label=LABEL[key])
            if m == "rerank":
                for _, r in dm.iterrows():
                    if int(r.p) in (100, 1000, 5000):
                        # above and to the left of the point: the curve rises to the right, so this stays clear
                        ax.annotate(f"K={int(r.p)}", (r.ms, r.recall), xytext=(-3, 4), textcoords="offset points",
                                    ha="right", va="bottom", fontsize=6.5, color=INK2)
        logaxis(ax, "x")
        ax.set_title(f"{n:,}".replace(",", "\u2009") + " memories")
        ax.set_xlabel("ms per query")
        ax.set_ylim(-0.02, 1.03)
    axes[0].set_ylabel("recall@10 of the full score")
    axes[-1].legend(loc="center right", handlelength=1.6)
    save(fig, name)


def fig_static_scaling(csv: Path, profile: str = "paper") -> None:
    d = pd.read_csv(csv)
    d = d[d.profile == profile]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.4))
    sel = [("folded-ip", "64", "ours", "folded HNSW (ours), ef=64"),
           ("rerank", "100", "rerank", "overfetch + rerank, K=100"),
           ("rerank", "1000", "rerank", "overfetch + rerank, K=1000"),
           ("relonly", "64", "relonly", "relevance-only HNSW")]
    for m, p, key, lab in sel:
        dm = d[(d.method == m) & (d.param.astype(str) == p)].sort_values("n")
        if dm.empty:
            continue
        ls = "--" if p == "1000" else "-"
        a1.plot(dm.n, dm.recall, ls, marker="o", color=C[key], label=lab)
    logaxis(a1, "x")
    a1.set_ylim(-0.02, 1.03)
    a1.set_xlabel("memories")
    a1.set_ylabel("recall@10 of the full score")
    a1.legend(loc="center left", fontsize=6.8)
    for m, p, key, lab in [("exact", "-", "exact", "exact scan"), ("folded-ip", "64", "ours", "folded HNSW (ours), ef=64"),
                           ("rerank", "100", "rerank", "overfetch + rerank, K=100")]:
        dm = d[(d.method == m) & (d.param.astype(str) == p)].sort_values("n")
        if not dm.empty:
            a2.plot(dm.n, dm.ms, "-o", color=C[key], label=lab)
    logaxis(a2, "x")
    logaxis(a2, "y")
    a2.set_xlabel("memories")
    a2.set_ylabel("ms per query (1 thread)")
    a2.legend(loc="upper left", fontsize=6.8)
    save(fig, "static_scaling")


def fig_static_ablation(csv: Path, n: int = 100_000, profile: str = "paper") -> None:
    d = pd.read_csv(csv)
    d = d[(d.profile == profile) & (d.n == n)]
    if d.empty:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
    svals = [("folded-s=0.25", "s = 0.25", BLUE_RAMP[0]), ("folded-s=0.5", "s = 0.5", BLUE_RAMP[1]),
             ("folded-ip", "s = 1 (default)", BLUE_RAMP[2]), ("folded-s=2", "s = 2", BLUE_RAMP[3]),
             ("folded-s=4", "s = 4", BLUE_RAMP[4])]
    for m, lab, col in svals:
        dm = d[d.method == m].copy()
        if dm.empty:
            continue
        dm["p"] = pd.to_numeric(dm.param)
        dm = dm[dm.p <= 128].sort_values("p")
        a1.plot(dm.p, dm.recall, "-o", color=col, label=lab)
    dm = d[d.method == "folded-oldanchor"].copy()
    if not dm.empty:
        dm["p"] = pd.to_numeric(dm.param)
        dm = dm[dm.p <= 128].sort_values("p")
        a1.plot(dm.p, dm.recall, "-s", color=C["exact"], label="anchor at oldest access")
    logaxis(a1, "x", 2)
    a1.set_xlabel("ef (search depth)")
    a1.set_ylabel("recall@10")
    a1.set_title("construction scale s and anchor")
    # two columns in the empty band between s = 0.25 and s = 0.5
    a1.legend(fontsize=6.3, loc="center", bbox_to_anchor=(0.5, 0.47), ncol=2, columnspacing=1.0,
              handlelength=1.6)
    for m, key in [("folded-ip", "ours"), ("folded-l2", "l2"), ("folded-ivf", "ivf")]:
        dm = d[d.method == m].copy()
        if dm.empty:
            continue
        dm["p"] = pd.to_numeric(dm.param)
        dm = dm.sort_values("ms")
        a2.plot(dm.ms, dm.recall, "-o", color=C[key], label=LABEL[key])
    logaxis(a2, "x")
    a2.set_xlabel("ms per query")
    a2.set_title("index type")
    a2.legend(fontsize=6.5, loc="lower right")
    save(fig, "static_ablation")


# ------------------------------------------------------------------ town

TOWN = [("exact", "exact", "exact scan (paper's score)"), ("folded", "ours", "folded HNSW (ours)"),
        ("rerank100", "rerank", "overfetch + rerank, K=100"), ("relonly", "relonly", "relevance-only HNSW")]


def fig_town(csv: Path, profile: str = "paper") -> None:
    d = pd.read_csv(csv)
    d = d[d.profile == profile]
    if d.empty:
        return
    burns = sorted(d.burn.unique())
    mem = d[d.backend_key == "exact"].groupby("burn").mean_memories.mean()
    ticks = [f"{b + 2} d\n{mem.get(b, np.nan):,.0f}".replace(",", "\u2009") for b in burns]
    metrics = [("aware_at_party", "knew about the party"), ("recalled_at_plan", "recalled it when planning"),
               ("attended", "attended")]
    keys = [k for k, _, _ in TOWN]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.45), sharex=True)
    for ax, (col, title) in zip(axes, metrics):
        for key, ckey, lab in TOWN:
            g = d[d.backend_key == key].groupby("burn")[col]
            if g.ngroups == 0:
                continue
            m, e = g.mean(), g.apply(ci95)
            x = np.array([burns.index(b) for b in m.index]) + 0.12 * (keys.index(key) - 1.5)
            style = dict(linestyle="--", marker="s") if key == "exact" else dict(linestyle="-", marker="o")
            ax.errorbar(x, m.values, yerr=e.values, color=C[ckey], label=lab, capsize=2, elinewidth=0.8, **style)
        ax.set_xticks(range(len(burns)), ticks)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("agents (of 25), mean ± 95% CI")
    fig.supxlabel("age of the town at the party in days (memories per agent)", fontsize=8, y=-0.07)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=6.8, bbox_to_anchor=(0.5, 1.07))
    save(fig, "town_outcomes")


def fig_town_recall(jsonl: Path, burn: int = 90, profile: str = "paper") -> None:
    rows = [json.loads(line) for line in jsonl.open()]
    rows = [r for r in rows if r["burn"] == burn and r["profile"] == profile]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for key, ckey, lab in TOWN[1:]:
        R = [r["recall_per_day"] for r in rows if r["backend_key"] == key]
        if not R:
            continue
        A = np.array([[np.nan if v is None else v for v in r] for r in R], dtype=float)
        ax.plot(np.arange(A.shape[1]), np.nanmean(A, axis=0), color=C[ckey], label=lab)
    ax.set_xlabel("simulated day")
    ax.set_ylabel("recall of the paper's top-k")
    ax.set_ylim(-0.02, 1.03)
    ax.legend(fontsize=6.5, loc="lower left")
    save(fig, "town_recall_over_time")


# ------------------------------------------------------------------ HNSW reimplementation

def fig_hnsw_validate(csv: Path) -> None:
    d = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    ax.plot(d.ef, d.recall_hnswlib, "-s", color=C["rerank"], label="hnswlib (authors' C++)")
    ax.plot(d.ef, d.recall_ours, "-o", color=C["ours"], label="our implementation", markersize=3.5)
    logaxis(ax, "x", 2)
    ax.set_xlabel("ef")
    ax.set_ylabel("recall@10")
    ax.set_title("GloVe-100, " + f"{int(d.n.iloc[0]):,}".replace(",", "\u2009") + " vectors")
    ax.legend(loc="lower right")
    save(fig, "hnsw_validate")


def fig_hnsw_ml(csvs: list[tuple[Path, str]]) -> None:
    csvs = [(p, t) for p, t in csvs if p.exists()]
    if not csvs:
        return
    fig, axes = plt.subplots(1, len(csvs), figsize=(3.3 * len(csvs), 2.3))
    axes = np.atleast_1d(axes)
    for ax, (p, title) in zip(axes, csvs):
        d = pd.read_csv(p).groupby(["heuristic", "mL"]).cost_at_target.first().reset_index()
        for h, key, lab in [(0, "rerank", "simple selection (Alg. 3)"), (1, "ours", "heuristic (Alg. 4)")]:
            dh = d[d.heuristic == h]
            ax.plot(dh.mL, dh.cost_at_target, "-o", color=C[key], label=lab)
        M = int(pd.read_csv(p).M.iloc[0])
        ax.axvline(1 / np.log(M), color=INK2, linewidth=0.8)
        ax.annotate("1/ln M", (1 / np.log(M), 0), xytext=(3, 4), textcoords="offset points",
                    fontsize=6.5, color=INK2)
        ax.set_xlabel("mL (0 = flat graph)")
        ax.set_ylim(0, d.cost_at_target.max() * 1.12)   # from zero, so noise is not mistaken for an effect
        ax.set_title(title)
    axes[0].set_ylabel("distance evaluations\nper query at recall 0.9")
    axes[-1].legend(loc="center right", fontsize=6.5)
    save(fig, "hnsw_ml")


def fig_hnsw_fig7(csv: Path) -> None:
    d = pd.read_csv(csv)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.3), sharey=True)
    for ax, ds in zip(axes, ["uniform", "100 clusters"]):
        for h, key, lab in [(0, "rerank", "simple selection (Alg. 3)"), (1, "ours", "heuristic (Alg. 4)")]:
            dh = d[(d.dataset == ds) & (d.heuristic == h)].sort_values("dist_evals")
            ax.plot(dh.dist_evals, dh.recall, "-o", color=C[key], label=lab)
        logaxis(ax, "x")
        ax.set_title(f"d = 10, {ds}")
        ax.set_xlabel("distance evaluations per query")
    axes[0].set_ylabel("recall@10")
    axes[1].legend(loc="lower right")
    save(fig, "hnsw_clusters")


def fig_hnsw_glove(csv: Path) -> None:
    d = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    combos = [(1, False, "ours", "-", "hierarchical, heuristic"), (1, True, "ours", "--", "flat, heuristic"),
              (0, False, "rerank", "-", "hierarchical, simple"), (0, True, "rerank", "--", "flat, simple")]
    for h, flat, key, ls, lab in combos:
        dh = d[(d.heuristic == h) & ((d.mL == 0) == flat)].sort_values("dist_evals")
        ax.plot(dh.dist_evals, dh.recall, ls, marker="o" if not flat else "s", color=C[key], label=lab, markersize=3.5)
    logaxis(ax, "x")
    ax.set_xlabel("distance evaluations per query")
    ax.set_ylabel("recall@10")
    ax.set_title("GloVe-100, " + f"{int(d.n.iloc[0]):,}".replace(",", "\u2009") + " vectors")
    ax.legend(loc="lower right", fontsize=6.5)
    save(fig, "hnsw_glove_flat")


# ------------------------------------------------------------------ reference code

def fig_ga_recency(js: Path) -> None:
    toy = json.loads(js.read_text())["toy"]
    hrs = toy["last_access_hours"]
    ref = [toy["reference_recency_by_last_access_hour"][str(h)] for h in hrs]
    pap = [toy["paper_recency_by_last_access_hour"][str(h)] for h in hrs]
    ago = [toy["now"] - h for h in hrs]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    x = np.arange(len(hrs))
    ax.bar(x - 0.19, pap, 0.36, color=C["ours"], label="paper (Sec. 4.1)")
    ax.bar(x + 0.19, ref, 0.36, color=C["rerank"], label="public code (retrieve.py)")
    ax.set_xticks(x, [f"{a:g} h" for a in ago])
    ax.set_xlabel("time since last access")
    ax.set_ylabel("normalized recency")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=2, fontsize=6.5, bbox_to_anchor=(0.5, 1.18))
    save(fig, "ga_recency")


def main() -> None:
    jobs = [
        (RES / "static_glove_k10.csv", lambda p: (fig_static_tradeoff(p), fig_static_scaling(p), fig_static_ablation(p))),
        (RES / "static_town_k10.csv", lambda p: fig_static_tradeoff(p, name="static_tradeoff_town")),
        (RES / "town.csv", fig_town),
        (RES / "town_recall.jsonl", fig_town_recall),
        (RES / "hnsw_validate.csv", fig_hnsw_validate),
        (RES / "hnsw_fig7_d10.csv", fig_hnsw_fig7),
        (RES / "hnsw_glove.csv", fig_hnsw_glove),
        (RES / "ga_reference_check.json", fig_ga_recency),
    ]
    for path, fn in jobs:
        if path.exists():
            fn(path)
    fig_hnsw_ml([(RES / "hnsw_fig3_d4.csv", "uniform d = 4, M = 6, 1-NN"),
                 (RES / "hnsw_fig4_d1024.csv", "uniform d = 1024, M = 20, 1-NN")])


if __name__ == "__main__":
    main()

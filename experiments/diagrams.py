"""Explanatory diagrams for the slides, drawn from real data structures.

  hnsw_layers.png   an HNSW graph built by our implementation on 2-D points,
                    shown layer by layer with the greedy search path of a query
  overfetch.png     the memories of one Willowbrook agent for a planning query:
                    relevance against the rest of the score, with the paper's
                    top-k and the candidates an overfetch of K=100 would see

    python -m experiments.diagrams
"""

from __future__ import annotations

import numpy as np

from experiments.plots import C, FIG, INK, INK2, plt, save


def _links(G, i, lc):
    L0, c0, LU, cU, up_off = G
    if lc == 0:
        return L0[i, : c0[i]]
    r = up_off[i] + lc - 1
    return LU[r, : cU[r]]


def hnsw_layers() -> None:
    from agentmem.paper_hnsw import L2, PaperHNSW

    rng = np.random.default_rng(3)
    X = rng.random((90, 2), dtype=np.float32)
    h = PaperHNSW(M=3, ef_construction=20, mL=0.9, metric=L2, seed=5).build(X)
    G, top = h.G, h.max_level
    q = np.array([0.82, 0.18], dtype=np.float32)
    # greedy descent, recorded
    cur = int(h.state[0])
    path = {top: [cur]}
    for lc in range(top, -1, -1):
        path.setdefault(lc, [cur])
        improved = True
        while improved:
            improved = False
            for e in _links(G, cur, lc):
                if np.sum((X[e] - q) ** 2) < np.sum((X[cur] - q) ** 2):
                    cur = int(e)
                    improved = True
            if improved:
                path[lc].append(cur)
        if lc > 0:
            path[lc - 1] = [cur]
    layers = list(range(min(top, 2), -1, -1))
    fig, ax = plt.subplots(figsize=(4.6, 5.2))
    skew = np.array([[1.0, 0.0], [0.45, 0.32]])

    def P(pts, lc):
        Y = np.atleast_2d(pts) @ skew.T
        return Y + np.array([0.0, 0.62 * lc])

    for lc in layers:
        corners = P(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]), lc)
        ax.fill(corners[:, 0], corners[:, 1], color="#f3f2ee", zorder=0)
        ax.plot(corners[:, 0], corners[:, 1], color="#c9c8c2", linewidth=0.6, zorder=0)
        members = np.nonzero(h.levels >= lc)[0]
        for i in members:
            for e in _links(G, i, lc):
                a, b = P(X[i], lc)[0], P(X[e], lc)[0]
                ax.plot([a[0], b[0]], [a[1], b[1]], color="#b9b8b3", linewidth=0.5, zorder=1)
        Pm = P(X[members], lc)
        ax.scatter(Pm[:, 0], Pm[:, 1], s=7 if lc == 0 else 14, color=INK2, zorder=2)
        pp = P(X[path[lc]], lc)
        ax.plot(pp[:, 0], pp[:, 1], color=C["rerank"], linewidth=1.8, zorder=3)
        ax.scatter(pp[:, 0], pp[:, 1], s=22, color=C["rerank"], zorder=4)
        qq = P(q, lc)[0]
        ax.scatter([qq[0]], [qq[1]], marker="*", s=110, color=C["ours"], zorder=5)
        ax.text(-0.02, P(np.array([0, 0.5]), lc)[0][1], f"layer {lc}", ha="right", va="center", fontsize=8,
                color=INK)
        if lc > 0:   # drop to the next layer
            a = P(X[path[lc][-1]], lc)[0]
            b = P(X[path[lc][-1]], lc - 1)[0]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=C["rerank"], linewidth=1.0, linestyle=(0, (2, 2)), zorder=3)
    ax.scatter([], [], marker="*", s=80, color=C["ours"], label="query")
    ax.plot([], [], color=C["rerank"], linewidth=1.8, label="greedy search path")
    ax.legend(loc="upper center", bbox_to_anchor=(0.55, 1.04), ncol=2, fontsize=7.5)
    ax.set_axis_off()
    ax.set_aspect("equal")
    save(fig, "hnsw_layers")


def overfetch() -> None:
    from agentmem import PROFILES, exact_scores, topk
    from agentmem.score import minmax, recency
    from town.sim import SimConfig, Town
    from town.world import date_str

    town = Town(SimConfig(days=32, party_day=31, seed=0, measure_recall=False))
    town.run()
    a = town.by_name["Max Phillips"]
    n = a.store.n
    T = 31 * 24 + 6.0
    q = town.enc.encode([f"{a.name}'s plan for {date_str(31)}."])[0]
    E, imp, t = a.store.E[:n], a.store.imp[:n], a.store.t_last[:n]
    rel = E @ q
    score = exact_scores(E, imp, t, q, T, PROFILES["paper"])
    other = minmax(recency(t, T)) + minmax(imp)
    true = topk(score, 30)
    cand = np.union1d(topk(rel.astype(np.float64), 100), np.arange(n - 30, n))
    missed = np.setdiff1d(true, cand)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.scatter(minmax(rel), other, s=5, color="#c9c8c2", label="all memories", zorder=1)
    ax.scatter(minmax(rel)[cand], other[cand], s=9, color=C["rerank"], label="overfetch candidates (K=100)",
               zorder=2)
    ax.scatter(minmax(rel)[true], other[true], s=26, facecolors="none", edgecolors=C["ours"], linewidths=1.2,
               label="paper's top-30", zorder=3)
    ax.set_xlabel("relevance to the focal point (min-max)")
    ax.set_ylabel("recency + importance (min-max)")
    ax.set_title(f"{a.name}, {n} memories: {len(missed)} of the top-30 are never scored", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper left")
    save(fig, "overfetch")
    print(f"  overfetch diagram: {len(missed)} of 30 missed")


if __name__ == "__main__":
    hnsw_layers()
    overfetch()

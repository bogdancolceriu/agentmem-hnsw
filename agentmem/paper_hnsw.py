"""HNSW written from scratch, following Algorithms 1-5 of

    Y. A. Malkov and D. A. Yashunin, "Efficient and Robust Approximate Nearest
    Neighbor Search Using Hierarchical Navigable Small World Graphs",
    IEEE TPAMI 42(4), 2020 (arXiv:1603.09320).

The code mirrors the pseudocode on purpose (names, loop structure) so that it
can be read side by side with the paper; it is compiled with numba. It exists
to study the algorithm (ablations of the hierarchy and of the neighbour
selection heuristic, distance evaluations per query), not to replace hnswlib,
which is the authors' optimized C++ implementation and is used everywhere else.

Every distance evaluation is counted, which gives a hardware-independent cost
measure for the ablations.

Graph layout: level-0 links in L0 (n x Mmax0), upper-level links of a node i
with top level l_i >= 1 in rows up_off[i] .. up_off[i] + l_i - 1 of LU (x Mmax).
"""

from __future__ import annotations

import heapq
import math
import time

import numba as nb
import numpy as np

L2, IP = 0, 1   # squared Euclidean distance; 1 - inner product (cosine on unit vectors)


@nb.njit(inline="always", fastmath=True)
def _dist(X, i, q, metric):
    s = 0.0
    if metric == 0:
        for j in range(X.shape[1]):
            t = X[i, j] - q[j]
            s += t * t
        return s
    for j in range(X.shape[1]):
        s += X[i, j] * q[j]
    return 1.0 - s


@nb.njit(inline="always")
def _links(G, i, lc):
    L0, c0, LU, cU, up_off = G
    if lc == 0:
        return L0[i], c0[i]
    r = up_off[i] + lc - 1
    return LU[r], cU[r]


@nb.njit(inline="always")
def _set_count(G, i, lc, cnt):
    L0, c0, LU, cU, up_off = G
    if lc == 0:
        c0[i] = cnt
    else:
        cU[up_off[i] + lc - 1] = cnt


@nb.njit
def search_layer(X, q, eps, ef, lc, G, metric, vis, vtag, counter):
    """Algorithm 2. Returns the (at most ef) nearest elements found, sorted by distance."""
    vtag[0] += 1
    if vtag[0] == 2147483647:
        vis[:] = 0
        vtag[0] = 1
    tag = vtag[0]
    C = [(0.0, np.int64(0))]      # candidates: min-heap on distance
    C.pop()
    W = [(0.0, np.int64(0))]      # dynamic list of found neighbours: max-heap on distance
    W.pop()
    for e in eps:
        if vis[e] == tag:
            continue
        vis[e] = tag
        d = _dist(X, e, q, metric)
        counter[0] += 1
        heapq.heappush(C, (d, np.int64(e)))
        heapq.heappush(W, (-d, np.int64(e)))
    while len(W) > ef:
        heapq.heappop(W)
    while len(C) > 0:
        dc, c = heapq.heappop(C)          # nearest element from C to q
        if dc > -W[0][0]:                  # all elements in W are evaluated
            break
        nbrs, cnt = _links(G, c, lc)
        for j in range(cnt):
            e = np.int64(nbrs[j])
            if vis[e] != tag:
                vis[e] = tag
                de = _dist(X, e, q, metric)
                counter[0] += 1
                if len(W) < ef or de < -W[0][0]:
                    heapq.heappush(C, (de, e))
                    heapq.heappush(W, (-de, e))
                    if len(W) > ef:
                        heapq.heappop(W)
    n = len(W)
    ds = np.empty(n)
    ids = np.empty(n, dtype=np.int64)
    for j in range(n):
        ds[j] = -W[j][0]
        ids[j] = W[j][1]
    order = np.argsort(ds)
    return ds[order], ids[order]


@nb.njit
def select_heuristic(X, cand_d, cand_i, M, metric, counter, keep_pruned):
    """Algorithm 4 (extendCandidates = false). Candidates must be sorted by distance to the base."""
    R = np.empty(M, dtype=np.int64)
    nr = 0
    Wd = np.empty(len(cand_i), dtype=np.int64)
    nd = 0
    for j in range(len(cand_i)):
        if nr >= M:
            break
        e = cand_i[j]
        good = True
        for t in range(nr):
            counter[0] += 1
            if _dist(X, e, X[R[t]], metric) < cand_d[j]:   # e is closer to an accepted r than to q
                good = False
                break
        if good:
            R[nr] = e
            nr += 1
        else:
            Wd[nd] = e
            nd += 1
    if keep_pruned:
        for j in range(nd):
            if nr >= M:
                break
            R[nr] = Wd[j]
            nr += 1
    return R[:nr]


@nb.njit
def _select(X, cand_d, cand_i, M, metric, counter, heuristic, keep_pruned):
    if heuristic:
        return select_heuristic(X, cand_d, cand_i, M, metric, counter, keep_pruned)
    return cand_i[:M].copy()                                # Algorithm 3


@nb.njit
def _connect(G, X, e, lc, new, mmax, metric, counter, heuristic, keep_pruned):
    """Add link e -> new on layer lc, shrinking e's neighbourhood if it exceeds mmax."""
    links, cnt = _links(G, e, lc)
    if cnt < mmax:
        links[cnt] = new
        _set_count(G, e, lc, cnt + 1)
        return
    cand = np.empty(cnt + 1, dtype=np.int64)
    cd = np.empty(cnt + 1)
    for j in range(cnt):
        cand[j] = links[j]
        cd[j] = _dist(X, cand[j], X[e], metric)
    cand[cnt] = new
    cd[cnt] = _dist(X, new, X[e], metric)
    counter[0] += cnt + 1
    order = np.argsort(cd)
    keep = _select(X, cd[order], cand[order], mmax, metric, counter, heuristic, keep_pruned)
    for j in range(len(keep)):
        links[j] = keep[j]
    _set_count(G, e, lc, len(keep))


@nb.njit
def insert(i, X, G, levels, state, M, efc, metric, heuristic, keep_pruned, vis, vtag, counter):
    """Algorithm 1 for element i (its level was drawn in advance as floor(-ln(U) * mL))."""
    ep, L = state[0], state[1]
    l = levels[i]
    q = X[i]
    if ep < 0:
        state[0], state[1] = i, l
        return
    eps = np.array([ep], dtype=np.int64)
    for lc in range(L, l, -1):
        ds, ids = search_layer(X, q, eps, 1, lc, G, metric, vis, vtag, counter)
        eps = ids[:1].copy()
    for lc in range(min(L, l), -1, -1):
        ds, ids = search_layer(X, q, eps, efc, lc, G, metric, vis, vtag, counter)
        nbrs = _select(X, ds, ids, M, metric, counter, heuristic, keep_pruned)
        links, _ = _links(G, i, lc)
        for j in range(len(nbrs)):
            links[j] = nbrs[j]
        _set_count(G, i, lc, len(nbrs))
        mmax = 2 * M if lc == 0 else M          # Mmax0 = 2M, Mmax = M
        for j in range(len(nbrs)):
            _connect(G, X, nbrs[j], lc, i, mmax, metric, counter, heuristic, keep_pruned)
        eps = ids                               # ep <- W
    if l > L:
        state[0], state[1] = i, l


@nb.njit
def build(X, G, levels, state, M, efc, metric, heuristic, keep_pruned, counter):
    vis = np.zeros(X.shape[0], dtype=np.int32)
    vtag = np.zeros(1, dtype=np.int32)
    for i in range(X.shape[0]):
        insert(i, X, G, levels, state, M, efc, metric, heuristic, keep_pruned, vis, vtag, counter)


@nb.njit
def knn_search_batch(X, Q, K, ef, G, state, metric, counter):
    """Algorithm 5 for every row of Q."""
    out = np.empty((Q.shape[0], K), dtype=np.int64)
    vis = np.zeros(X.shape[0], dtype=np.int32)
    vtag = np.zeros(1, dtype=np.int32)
    for qi in range(Q.shape[0]):
        q = Q[qi]
        eps = np.array([state[0]], dtype=np.int64)
        for lc in range(state[1], 0, -1):
            ds, ids = search_layer(X, q, eps, 1, lc, G, metric, vis, vtag, counter)
            eps = ids[:1].copy()
        ds, ids = search_layer(X, q, eps, max(ef, K), 0, G, metric, vis, vtag, counter)
        for j in range(K):
            out[qi, j] = ids[j] if j < len(ids) else -1
    return out


class PaperHNSW:
    """Thin Python wrapper. mL=None uses the paper's default 1/ln(M); mL=0 gives a flat graph."""

    def __init__(self, M: int = 16, ef_construction: int = 200, mL: float | None = None,
                 heuristic: bool = True, keep_pruned: bool = False, metric: int = L2, seed: int = 0):
        self.M, self.efc = M, ef_construction
        self.mL = 1.0 / math.log(M) if mL is None else mL
        self.heuristic, self.keep_pruned, self.metric, self.seed = heuristic, keep_pruned, metric, seed

    def build(self, X: np.ndarray) -> "PaperHNSW":
        X = np.ascontiguousarray(X, dtype=np.float32)
        n = X.shape[0]
        rng = np.random.default_rng(self.seed)
        u = 1.0 - rng.random(n)                     # in (0, 1]
        levels = np.floor(-np.log(u) * self.mL).astype(np.int64) if self.mL > 0 else np.zeros(n, np.int64)
        up_off = np.zeros(n, dtype=np.int64)
        up_off[1:] = np.cumsum(levels)[:-1]
        rows = int(levels.sum())
        self.G = (np.zeros((n, 2 * self.M), dtype=np.int32), np.zeros(n, dtype=np.int32),
                  np.zeros((max(rows, 1), self.M), dtype=np.int32), np.zeros(max(rows, 1), dtype=np.int32),
                  up_off)
        self.X, self.levels = X, levels
        self.state = np.array([-1, 0], dtype=np.int64)
        counter = np.zeros(1, dtype=np.int64)
        t0 = time.perf_counter()
        build(X, self.G, levels, self.state, self.M, self.efc, self.metric, self.heuristic,
              self.keep_pruned, counter)
        self.build_s = time.perf_counter() - t0
        self.build_dist = int(counter[0])
        return self

    def search(self, Q: np.ndarray, k: int, ef: int) -> tuple[np.ndarray, float, float]:
        """Return (labels, mean distance evaluations per query, queries per second)."""
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        counter = np.zeros(1, dtype=np.int64)
        t0 = time.perf_counter()
        out = knn_search_batch(self.X, Q, k, ef, self.G, self.state, self.metric, counter)
        dt = time.perf_counter() - t0
        return out, counter[0] / len(Q), len(Q) / dt

    @property
    def max_level(self) -> int:
        return int(self.state[1])

    def degree_stats(self) -> dict:
        c0 = self.G[1]
        return dict(mean_degree0=float(c0.mean()), max_degree0=int(c0.max()),
                    nodes_above0=int((self.levels > 0).sum()), max_level=self.max_level)

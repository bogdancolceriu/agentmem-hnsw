"""Memory stores for scores with a general forgetting curve (see curves.py).

    score_i(T, q) = a * D_i(T) + b * importance_i + c * cos(q, e_i)

with each component min-max normalised over all memories when w.minmax (as in Park et al.
2023), and

    D_i(T) = sum_j phi(T - t_ij)      frequency=True: every use of the memory counts (ACT-R), or
    D_i(T) = phi(T - t_last,i)        frequency=False: only the last use (as in the paper).

log=True ranks by ln D_i(T) instead, ACT-R's base-level activation. The logarithm of an
inner product is not an inner product, so only the exact scan (CurveExact) supports it.

    CurveExact       scan with the true curve (ground truth)
    CurveExactFit    scan with the fitted sum of exponentials (isolates the approximation)
    CurveRelevance   plain similarity search with HNSW
    CurveOverfetch   similarity top-K plus the newest memories, re-ranked (LangChain's pattern)
    CurveFolded      ours: the score as one inner product over [e, s*z, s*imp] (d + R + 1
                     numbers), with a recent part scanned exactly and time-anchored segments

Bounded coordinates (CurveFolded). A memory stays in the recent part while its last use is
younger than `recent_hours` (W). After that it moves to the hot segment, which is open for
`seg_hours` (L <= W) and is anchored at tau = (its opening time) + L - W. A memory therefore
enters a segment only when t_last < T - W <= tau, so every stored coordinate
z_ir = w_r y_ir exp(-lam_r (tau - t_last,i)) lies in [0, w_r y_ir], and every query factor
exp(-lam_r (T - tau)) lies in (0, 1], for every rate however fast. Compaction merges closed
segments and re-anchors them at their latest use, which keeps the same bounds.
"""

from __future__ import annotations

import heapq
import time
from collections import defaultdict

import numpy as np

from .curves import ACT_R, YEAR_H, Curve, fit_expsum
from .score import Weights, minmax, topk
from .stores import _Hnsw


class CurveMemory:
    name = "curve base"
    supports_log = False

    def __init__(self, dim: int, curve: Curve = ACT_R, R: int = 8, frequency: bool = True,
                 age_max: float = YEAR_H, capacity: int = 1024, log: bool = False, **_ignored):
        if log and not self.supports_log:
            raise ValueError("the logarithm of D is not an inner product: use CurveExact for log=True")
        self.dim = dim
        self.curve = curve
        self.frequency = frequency
        self.log = log
        self.fit = fit_expsum(curve, R, age_max)
        self.lam, self.wr, self.R = self.fit.lam, self.fit.w, self.fit.R
        self.E = np.zeros((capacity, dim), dtype=np.float32)
        self.imp = np.zeros(capacity)
        self.t_last = np.zeros(capacity)
        self.t_created = np.zeros(capacity)
        self.y = np.zeros((capacity, self.R))
        self.n = 0
        self.imp_lo, self.imp_hi = np.inf, -np.inf
        self.t_hi = -np.inf
        self.acc_mem = np.zeros(4096, dtype=np.int64)   # every use: (memory, time), for the true curve
        self.acc_t = np.zeros(4096)
        self.n_acc = 0
        self.stats: dict[str, float] = defaultdict(float)
        self._cache: tuple | None = None

    # ------------------------------------------------------------ storage
    def _grow(self, need: int) -> None:
        cap = self.E.shape[0]
        if need <= cap:
            return
        new = max(need, 2 * cap)
        for attr in ("E", "imp", "t_last", "t_created", "y"):
            old = getattr(self, attr)
            arr = np.zeros((new,) + old.shape[1:], dtype=old.dtype)
            arr[: self.n] = old[: self.n]
            setattr(self, attr, arr)
        self._on_grow(new)

    def _on_grow(self, new_capacity: int) -> None:
        pass

    def _log(self, ids: np.ndarray, t: np.ndarray) -> None:
        m = len(ids)
        if self.n_acc + m > len(self.acc_t):
            new = max(self.n_acc + m, 2 * len(self.acc_t))
            for attr in ("acc_mem", "acc_t"):
                old = getattr(self, attr)
                arr = np.zeros(new, dtype=old.dtype)
                arr[: self.n_acc] = old[: self.n_acc]
                setattr(self, attr, arr)
        self.acc_mem[self.n_acc:self.n_acc + m] = ids
        self.acc_t[self.n_acc:self.n_acc + m] = t
        self.n_acc += m
        self._cache = None

    def add(self, E, imp, t) -> np.ndarray:
        E = np.atleast_2d(np.asarray(E, dtype=np.float32))
        m = E.shape[0]
        imp = np.broadcast_to(np.asarray(imp, dtype=np.float64), (m,))
        t = np.broadcast_to(np.asarray(t, dtype=np.float64), (m,))
        self._grow(self.n + m)
        ids = np.arange(self.n, self.n + m)
        self.E[ids], self.imp[ids] = E, imp
        self.t_last[ids] = self.t_created[ids] = t
        self.y[ids] = 1.0
        self.n += m
        self._log(ids, t)
        self.imp_lo, self.imp_hi = min(self.imp_lo, float(imp.min())), max(self.imp_hi, float(imp.max()))
        self.t_hi = max(self.t_hi, float(t.max()))
        t0 = time.perf_counter()
        self._on_add(ids, self.t_hi)
        self.stats["insert_s"] += time.perf_counter() - t0
        self.stats["inserted"] += m
        return ids

    def load(self, E, imp, histories: list[np.ndarray], T_now: float) -> np.ndarray:
        """Bulk load memories with their whole use history (creation first), as of time T_now."""
        E = np.atleast_2d(np.asarray(E, dtype=np.float32))
        m = E.shape[0]
        self._grow(self.n + m)
        ids = np.arange(self.n, self.n + m)
        mem = np.concatenate([np.full(len(h), i) for i, h in zip(ids, histories)])
        t = np.concatenate([np.asarray(h, dtype=np.float64) for h in histories])
        self.E[ids], self.imp[ids] = E, np.asarray(imp, dtype=np.float64)
        self.t_created[ids] = [float(np.min(h)) for h in histories]
        self.t_last[ids] = [float(np.max(h)) for h in histories]
        if self.frequency:
            age = self.t_last[mem] - t
            for r in range(self.R):
                self.y[ids, r] = np.bincount(mem - self.n, weights=np.exp(-self.lam[r] * age), minlength=m)
        else:
            self.y[ids] = 1.0
        self.n += m
        self._log(mem, t)
        self.imp_lo = min(self.imp_lo, float(np.min(imp)))
        self.imp_hi = max(self.imp_hi, float(np.max(imp)))
        self.t_hi = max(self.t_hi, float(self.t_last[ids].max()))
        self._on_load(ids, float(T_now))
        return ids

    def _on_add(self, ids: np.ndarray, T: float) -> None:
        pass

    def _on_load(self, ids: np.ndarray, T: float) -> None:
        self._on_add(ids, T)

    def touch(self, ids: np.ndarray, T: float) -> None:
        """Record a use of `ids` at time T (refresh on retrieval)."""
        if self.frequency:
            self.y[ids] = self.y[ids] * np.exp(-np.outer(T - self.t_last[ids], self.lam)) + 1.0
        else:
            self.y[ids] = 1.0
        self.t_last[ids] = T
        self.t_hi = max(self.t_hi, float(T))
        self._log(ids, np.full(len(ids), float(T)))

    # ------------------------------------------------------------ scoring
    def D_true(self, T: float) -> np.ndarray:
        """Recency-frequency term of every memory with the true curve (its logarithm if self.log)."""
        if not self.frequency:
            D = self.curve(T - self.t_last[: self.n])
        else:
            k = self.n_acc
            D = np.bincount(self.acc_mem[:k], weights=self.curve(T - self.acc_t[:k]), minlength=self.n)
        return np.log(D) if self.log else D

    def D_fit(self, T: float, ids: np.ndarray | None = None) -> np.ndarray:
        """The same term with the fitted sum of exponentials (what an inner product can carry)."""
        ids = np.arange(self.n) if ids is None else ids
        return (self.y[ids] * np.exp(-np.outer(T - self.t_last[ids], self.lam))) @ self.wr

    @staticmethod
    def combine(D, imp, rel, w: Weights) -> np.ndarray:
        if w.minmax and len(D):
            D, imp, rel = minmax(D), minmax(imp), minmax(rel)
        return w.recency * D + w.importance * imp + w.relevance * rel

    def exact(self, q: np.ndarray, T: float, w: Weights) -> np.ndarray:
        """Ground-truth score of every memory (true curve)."""
        key = (float(T), q.tobytes(), w, self.n_acc, self.n)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        rel = (self.E[: self.n] @ q).astype(np.float64)
        s = self.combine(self.D_true(T), self.imp[: self.n], rel, w)
        self._cache = (key, s)
        return s

    def exact_fit(self, q: np.ndarray, T: float, w: Weights) -> np.ndarray:
        rel = (self.E[: self.n] @ q).astype(np.float64)
        return self.combine(self.D_fit(T), self.imp[: self.n], rel, w)

    def fit_ranges(self, q: np.ndarray, T: float) -> tuple[float, float, float]:
        """Exact (recency-frequency, importance, relevance) ranges with the fitted curve."""
        D = self.D_fit(T)
        rel = self.E[: self.n] @ q
        return float(np.ptp(D)), float(self.imp_hi - self.imp_lo), float(np.ptp(rel))

    # ------------------------------------------------------------ retrieval
    def retrieve(self, q: np.ndarray, T: float, k: int, w: Weights, refresh: bool = True) -> np.ndarray:
        q = np.asarray(q, dtype=np.float32)
        t0 = time.perf_counter()
        ids = self._search(q, float(T), int(k), w)
        t1 = time.perf_counter()
        self.stats["search_s"] += t1 - t0
        self.stats["queries"] += 1
        if refresh and len(ids):
            self.touch(ids, float(T))
            self._on_refresh(ids, float(T))
            self.stats["refresh_s"] += time.perf_counter() - t1
        return ids

    def _search(self, q, T, k, w) -> np.ndarray:
        raise NotImplementedError

    def _on_refresh(self, ids: np.ndarray, T: float) -> None:
        pass

    def memory_bytes(self) -> int:
        return self.n * self.dim * 4


def _raw(w: Weights, ranges: tuple[float, float, float]) -> tuple[float, float, float]:
    """(recency, importance, relevance) weights that rank like the min-max normalised score."""
    if not w.minmax:
        return w.recency, w.importance, w.relevance
    f = lambda a, r: a / r if r > 1e-12 else 0.0   # noqa: E731 - a constant component cannot reorder
    return f(w.recency, ranges[0]), f(w.importance, ranges[1]), f(w.relevance, ranges[2])


def pareto_max(Z: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Indices of the Pareto-maximal rows of Z: no other row is >= in every column and > in one.

    For a query vector g with positive entries, max_i <g, z_i> is reached on this set (and
    min_i on the Pareto-maximal rows of -Z), so the range of the recency-frequency term over a
    segment needs no search. Rows are visited by decreasing sum, since a row can only be
    dominated by one with a larger or equal sum.
    """
    order = np.argsort(-Z.sum(1), kind="stable")
    front, keep = np.empty((0, Z.shape[1])), []
    for s0 in range(0, len(order), chunk):
        idx = order[s0:s0 + chunk]
        C = Z[idx]
        if len(front):
            dom = np.zeros(len(idx), dtype=bool)
            for f0 in range(0, len(front), 2048):
                F = front[f0:f0 + 2048]
                dom |= (np.all(F[None] >= C[:, None], axis=2) & np.any(F[None] > C[:, None], axis=2)).any(1)
            idx, C = idx[~dom], C[~dom]
        if len(idx) > 1:
            dom = (np.all(C[None] >= C[:, None], axis=2) & np.any(C[None] > C[:, None], axis=2)).any(1)
            idx, C = idx[~dom], C[~dom]
        keep.extend(idx.tolist())
        front = np.vstack([front, C])
    return np.asarray(keep, dtype=np.int64)


class CurveExact(CurveMemory):
    name = "exact scan (true curve)"
    supports_log = True

    def _search(self, q, T, k, w):
        return topk(self.exact(q, T, w), k)


class CurveExactFit(CurveMemory):
    name = "exact scan (fitted curve)"

    def _search(self, q, T, k, w):
        return topk(self.exact_fit(q, T, w), k)


class CurveRelevance(CurveMemory):
    name = "relevance-only HNSW"

    def __init__(self, dim, M=16, ef_construction=200, ef=64, seed=0, threads=1, **kw):
        super().__init__(dim, **kw)
        self.h = _Hnsw("ip", dim, M, ef_construction, ef, seed=seed, threads=threads)

    def _on_add(self, ids, T):
        self.h.add(self.E[ids], ids)

    def _search(self, q, T, k, w):
        return self.h.query(q[None, :], k)


class CurveOverfetch(CurveRelevance):
    """Similarity top-`fetch` plus the `recent` newest memories, re-ranked by the fitted score.

    It is given the exact normalisation ranges, which can only help it: its losses come from
    the candidate set alone.
    """

    def __init__(self, dim, fetch=100, recent=None, **kw):
        super().__init__(dim, **kw)
        self.fetch, self.recent = fetch, recent
        self.name = f"overfetch+rerank (K={fetch})"

    def _search(self, q, T, k, w):
        return self.search_with_ranges(q, T, k, w, self.fit_ranges(q, T) if w.minmax else (1.0, 1.0, 1.0))

    def search_with_ranges(self, q, T, k, w, ranges) -> np.ndarray:
        """The retrieval itself, with the normalisation ranges supplied (timed separately)."""
        fetched = self.h.query(q[None, :], self.fetch)
        n_recent = self.recent if self.recent is not None else k
        cand = np.union1d(fetched, np.arange(max(0, self.n - n_recent), self.n))
        a, b, c = _raw(w, ranges)
        s = a * self.D_fit(T, cand) + b * self.imp[cand] + c * (self.E[cand] @ q)
        return topk(s, k, ids=cand)


class CurveFolded(CurveMemory):
    """Ours: the whole score as one inner product, for any forgetting curve fitted by R exponentials.

    memory side  x_i = [ e_i,  s * z_i(tau),  s * importance_i ]        z_ir = w_r y_ir e^{-lam_r (tau - t_last,i)}
    query side   Q   = [ c q,  (a / s) * g(T),  b / s ]                 g_r = e^{-lam_r (T - tau)}
    <Q, x_i>     = a * D_i(T) + b * importance_i + c * cos(q, e_i)      (D with the fitted curve)
    """

    def __init__(self, dim, recent_hours: float = 24.0, seg_hours: float | None = None,
                 max_closed: int = 4, s: float = 1.0, M: int = 16, ef_construction: int = 200,
                 ef: int = 64, norm_ef: int = 32, seed: int = 0, threads: int = 1,
                 fetch_factor: int = 1, exact_ranges_below: int = 20000, **kw):
        super().__init__(dim, **kw)
        self.exact_ranges_below = exact_ranges_below   # small segments: ranges by a scan, not a search
        self.W = float(recent_hours)
        self.L = float(seg_hours) if seg_hours is not None else self.W
        if self.L > self.W:
            raise ValueError("seg_hours must not exceed recent_hours (bounded coordinates)")
        self.max_closed, self.s, self.norm_ef, self.fetch_factor = max_closed, s, norm_ef, fetch_factor
        self.hnsw_kw = dict(M=M, ef_construction=ef_construction, ef=ef, seed=seed, threads=threads)
        self.recent: set[int] = set()
        self._rec_arr: np.ndarray | None = None      # cached sorted array of the recent part
        self._rec_cache: tuple | None = None         # (ids, E, z at T0, importance, T0) of the recent part
        self._rheap: list[tuple[float, int]] = []
        self.segments: list[dict] = []
        self.seg_of: dict[int, dict] = {}
        self._next_sid = 0
        self.loc = np.full(self.E.shape[0], -2, dtype=np.int64)   # -1 recent part, >= 0 segment id
        self.name = f"folded HNSW, R={self.R} (ours)"

    def _on_grow(self, new_capacity):
        loc = np.full(new_capacity, -2, dtype=np.int64)
        loc[: len(self.loc)] = self.loc
        self.loc = loc

    # -------------------------------------------------------------- segments
    def _make_segment(self, anchor: float, opened: float, capacity: int = 1024) -> dict:
        seg = dict(sid=self._next_sid, anchor=float(anchor), opened=float(opened), arr=None, fronts=None,
                   h=_Hnsw("ip", self.dim + self.R + 1, capacity=max(1024, capacity), **self.hnsw_kw))
        self.seg_of[self._next_sid] = seg
        self._next_sid += 1
        return seg

    def _vectors(self, ids: np.ndarray, anchor: float) -> np.ndarray:
        dt = anchor - self.t_last[ids]
        assert np.all(dt >= -1e-9), "a memory used after the segment anchor would have unbounded coordinates"
        z = self.y[ids] * np.exp(-np.outer(dt, self.lam)) * self.wr
        X = np.empty((len(ids), self.dim + self.R + 1), dtype=np.float32)
        X[:, : self.dim] = self.E[ids]
        X[:, self.dim:-1] = self.s * z
        X[:, -1] = self.s * self.imp[ids]
        return X

    def _hot(self, T: float) -> dict:
        if not self.segments or T - self.segments[-1]["opened"] >= self.L:
            self.segments.append(self._make_segment(T + self.L - self.W, T))
            if len(self.segments) - 1 > self.max_closed:
                self._compact()
        return self.segments[-1]

    def _compact(self) -> None:
        t0 = time.perf_counter()
        closed, hot = self.segments[:-1], self.segments[-1]
        ids = np.fromiter(sorted(set().union(*(seg["h"].live for seg in closed))), dtype=np.int64)
        anchor = float(self.t_last[ids].max()) if len(ids) else hot["anchor"]
        merged = self._make_segment(anchor, -np.inf, capacity=len(ids))
        if len(ids):
            merged["h"].add(self._vectors(ids, anchor), ids)
            self.loc[ids] = merged["sid"]
            merged["arr"] = merged["fronts"] = None
        for seg in closed:
            del self.seg_of[seg["sid"]]
        self.segments = [merged, hot]
        self.stats["compactions"] += 1
        self.stats["compact_s"] += time.perf_counter() - t0

    def _age_out(self, T: float) -> None:
        """Move every memory whose last use is older than W from the recent part to the hot segment."""
        h, out = self._rheap, []
        while h and h[0][0] < T - self.W:
            t, i = heapq.heappop(h)
            if self.loc[i] == -1 and self.t_last[i] == t:
                out.append(i)
        if out:
            ids = np.asarray(sorted(out), dtype=np.int64)
            seg = self._hot(T)
            seg["h"].add(self._vectors(ids, seg["anchor"]), ids)
            self.loc[ids] = seg["sid"]
            seg["arr"] = seg["fronts"] = None
            self.recent.difference_update(out)
            self._rec_arr = self._rec_cache = None

    def _to_recent(self, ids: np.ndarray) -> None:
        self.loc[ids] = -1
        self.recent.update(ids.tolist())
        self._rec_arr = self._rec_cache = None
        entries = list(zip(self.t_last[ids].tolist(), ids.tolist()))
        if len(entries) > len(self._rheap) // 4:
            self._rheap.extend(entries)
            heapq.heapify(self._rheap)
        else:
            for e in entries:
                heapq.heappush(self._rheap, e)

    def _on_add(self, ids, T):
        self._to_recent(ids)
        self._age_out(T)

    def _on_load(self, ids, T):
        """Bulk load: recent memories go to the recent part, the rest to one compacted segment."""
        old = ids[self.t_last[ids] < T - self.W]
        self._to_recent(ids[self.t_last[ids] >= T - self.W])
        if len(old):
            anchor = float(self.t_last[old].max())
            seg = self._make_segment(anchor, -np.inf, capacity=len(old))
            seg["h"].add(self._vectors(old, anchor), old)
            self.loc[old] = seg["sid"]
            self.segments.append(seg)

    def _on_refresh(self, ids, T):
        for i in ids.tolist():
            sid = self.loc[i]
            if sid >= 0:
                seg = self.seg_of[sid]
                seg["h"].delete(i)
                seg["arr"] = None
                if seg["fronts"] is not None and (i in seg["fronts"][2]):
                    seg["fronts"] = None   # deleting a dominated memory leaves the fronts unchanged
        self._to_recent(ids)
        self._age_out(T)

    # -------------------------------------------------------------- search
    def _g(self, T: float, seg: dict) -> np.ndarray:
        return np.exp(-self.lam * (T - seg["anchor"]))

    def _recent_ids(self) -> np.ndarray:
        if self._rec_arr is None:
            self._rec_arr = np.fromiter(sorted(self.recent), dtype=np.int64, count=len(self.recent))
        return self._rec_arr

    def _recent(self, T: float):
        """Recent part as contiguous arrays, its z anchored at T0 = its latest use (<= T)."""
        if self._rec_cache is None:
            ids = self._recent_ids()
            T0 = float(self.t_last[ids].max()) if len(ids) else T
            z = self.y[ids] * np.exp(-np.outer(T0 - self.t_last[ids], self.lam)) * self.wr
            self._rec_cache = (ids, np.ascontiguousarray(self.E[ids]), z, self.imp[ids].copy(), T0)
        return self._rec_cache

    def _fronts(self, seg: dict):
        if seg["fronts"] is None:
            ids = np.fromiter(sorted(seg["h"].live), dtype=np.int64, count=len(seg["h"].live))
            z = self.y[ids] * np.exp(-np.outer(seg["anchor"] - self.t_last[ids], self.lam)) * self.wr
            hi, lo = ids[pareto_max(z)], ids[pareto_max(-z)]
            seg["fronts"] = (hi, lo, set(hi.tolist()) | set(lo.tolist()))
        return seg["fronts"]

    def _ranges(self, q: np.ndarray, T: float, r_rel: np.ndarray, r_D: np.ndarray) -> tuple[float, float, float]:
        """(D, importance, relevance) ranges. D: exact, from the recent part and the Pareto fronts of
        each segment's z vectors (or a scan of a small segment). Relevance: exact over the recent
        part and small segments, and the best match of +-q in a large segment (one small search each)."""
        rel_lo, rel_hi, d_lo, d_hi = np.inf, -np.inf, np.inf, -np.inf
        if len(r_rel):
            rel_lo, rel_hi = float(r_rel.min()), float(r_rel.max())
            d_lo, d_hi = float(r_D.min()), float(r_D.max())
        vq = np.concatenate([q, np.zeros(self.R + 1, dtype=np.float32)])[None, :]
        for seg in self.segments:
            if not seg["h"].live:
                continue
            if len(seg["h"].live) < self.exact_ranges_below:     # cheaper to scan than to search
                if seg["arr"] is None:
                    seg["arr"] = np.fromiter(sorted(seg["h"].live), dtype=np.int64, count=len(seg["h"].live))
                ids = seg["arr"]
                r, d = self.E[ids] @ q, self.D_fit(T, ids)
                rel_lo, rel_hi = min(rel_lo, float(r.min())), max(rel_hi, float(r.max()))
                d_lo, d_hi = min(d_lo, float(d.min())), max(d_hi, float(d.max()))
                continue
            hi, lo, _ = self._fronts(seg)
            d_hi = max(d_hi, float(self.D_fit(T, hi).max()))
            d_lo = min(d_lo, float(self.D_fit(T, lo).min()))
            found = []
            for v in (vq, -vq):
                found.extend(seg["h"].query(v, 1, ef=self.norm_ef)[:1].tolist())
            if found:
                r = self.E[found] @ q
                rel_lo, rel_hi = min(rel_lo, float(r.min())), max(rel_hi, float(r.max()))
        return d_hi - d_lo, self.imp_hi - self.imp_lo, rel_hi - rel_lo

    def _search(self, q, T, k, w):
        self._age_out(T)
        rec, E_r, z_r, imp_r, T0 = self._recent(T)
        r_rel = E_r @ q if len(rec) else np.empty(0)
        r_D = z_r @ np.exp(-self.lam * (T - T0)) if len(rec) else np.empty(0)
        a, b, c = _raw(w, self._ranges(q, T, r_rel, r_D)) if w.minmax else _raw(w, (1, 1, 1))
        parts = []
        if len(rec):   # the recent part is scored exactly, once
            parts.append(topk(a * r_D + b * imp_r + c * r_rel, k, ids=rec))
        for seg in self.segments:
            if not seg["h"].live:
                continue
            Q = np.concatenate([c * q, (a / self.s) * self._g(T, seg), [b / self.s]]).astype(np.float32)
            parts.append(seg["h"].query(Q[None, :], k * self.fetch_factor))
        if not parts:
            return np.empty(0, dtype=np.int64)
        cand = np.unique(np.concatenate(parts))
        s = a * self.D_fit(T, cand) + b * self.imp[cand] + c * (self.E[cand] @ q)
        return topk(s, k, ids=cand)

    def coordinates_bounded(self) -> bool:
        """Check the bound of the module docstring on every live memory of every segment."""
        for seg in self.segments:
            ids = np.fromiter(seg["h"].live, dtype=np.int64)
            if len(ids) and np.any(self.t_last[ids] > seg["anchor"] + 1e-9):
                return False
        return True

    def memory_bytes(self) -> int:
        return sum(seg["h"].bytes() for seg in self.segments) + len(self.recent) * (self.dim + self.R + 1) * 4


CURVE_KINDS = {
    "curve_exact": CurveExact,
    "curve_exact_fit": CurveExactFit,
    "curve_relonly": CurveRelevance,
    "curve_rerank": CurveOverfetch,
    "curve_folded": CurveFolded,
}

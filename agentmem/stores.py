"""Memory stores for generative agents.

All stores keep the same metadata (embedding, importance, creation time, last
access time) and implement `retrieve(q, T, k, w, refresh=True)`, which returns
the ids of the k memories the store believes have the highest Generative-Agents
score and, as in Park et al. (2023), marks them as accessed at time T.

    ExactStore          linear scan over every memory (ground truth; what the
                        reference implementation does, vectorized)
    RelevanceOnlyHNSW   plain semantic search with HNSW (typical RAG memory)
    OverfetchRerank     HNSW top-`fetch` by similarity plus the `recent` newest
                        memories, re-ranked by the full score (the approach of
                        LangChain's TimeWeightedVectorStoreRetriever)
    FoldedHNSW          ours: the full score folded into an inner-product HNSW
                        index over augmented vectors (see score.py), with
                        time-anchored segments, a write buffer for
                        refresh-on-access and compaction of old segments
"""

from __future__ import annotations

import heapq
import time
from collections import defaultdict

import numpy as np

from .score import DECAY_PAPER, Weights, exact_scores, fold_memories, fold_query, recency, topk


class MemoryStore:
    name = "base"

    def __init__(self, dim: int, decay: float = DECAY_PAPER, capacity: int = 1024, **_ignored):
        self.dim = dim
        self.decay = decay
        self.E = np.zeros((capacity, dim), dtype=np.float32)
        self.imp = np.zeros(capacity, dtype=np.float64)
        self.t_last = np.zeros(capacity, dtype=np.float64)
        self.t_created = np.zeros(capacity, dtype=np.float64)
        self.n = 0
        # running statistics for min-max normalization
        self.imp_lo, self.imp_hi = np.inf, -np.inf
        self.t_hi = -np.inf
        self._theap: list[tuple[float, int]] = []   # lazy min-heap of (t_last, id)
        self.stats: dict[str, float] = defaultdict(float)

    # ------------------------------------------------------------ storage
    def _grow(self, need: int) -> None:
        cap = self.E.shape[0]
        if need <= cap:
            return
        new = max(need, 2 * cap)
        for attr in ("E", "imp", "t_last", "t_created"):
            old = getattr(self, attr)
            arr = np.zeros((new,) + old.shape[1:], dtype=old.dtype)
            arr[: self.n] = old[: self.n]
            setattr(self, attr, arr)
        self._on_grow(new)

    def _on_grow(self, new_capacity: int) -> None:
        pass

    def _push_times(self, ids: np.ndarray, t: np.ndarray) -> None:
        entries = list(zip(t.tolist(), ids.tolist()))
        if len(entries) > len(self._theap) // 4:
            self._theap.extend(entries)
            heapq.heapify(self._theap)
        else:
            for e in entries:
                heapq.heappush(self._theap, e)

    def add(self, E: np.ndarray, imp, t) -> np.ndarray:
        E = np.atleast_2d(np.asarray(E, dtype=np.float32))
        m = E.shape[0]
        imp = np.broadcast_to(np.asarray(imp, dtype=np.float64), (m,))
        t = np.broadcast_to(np.asarray(t, dtype=np.float64), (m,))
        self._grow(self.n + m)
        ids = np.arange(self.n, self.n + m)
        self.E[ids] = E
        self.imp[ids] = imp
        self.t_last[ids] = t
        self.t_created[ids] = t
        self.n += m
        self.imp_lo = min(self.imp_lo, float(imp.min()))
        self.imp_hi = max(self.imp_hi, float(imp.max()))
        self.t_hi = max(self.t_hi, float(t.max()))
        self._push_times(ids, t)
        t0 = time.perf_counter()
        # flush at the latest time the store has seen, not at the batch's own times: a backdated
        # batch must not send memories refreshed later into an old segment (bounded coordinates)
        self._on_add(ids, self.t_hi)
        self.stats["insert_s"] += time.perf_counter() - t0
        self.stats["inserted"] += m
        return ids

    def _on_add(self, ids: np.ndarray, T: float) -> None:
        pass

    def t_lo(self) -> float:
        h = self._theap
        while h and self.t_last[h[0][1]] != h[0][0]:
            heapq.heappop(h)
        return h[0][0] if h else self.t_hi

    def raw_weights(self, w: Weights, T: float, rel_lo: float, rel_hi: float) -> Weights:
        """Raw weights equivalent to min-max normalization, from maintained statistics."""
        if not w.minmax:
            return w
        rec_range = self.decay ** (T - self.t_hi) - self.decay ** (T - self.t_lo())
        return w.raw(rec_range, self.imp_hi - self.imp_lo, rel_hi - rel_lo)

    # ------------------------------------------------------------ retrieval
    def exact(self, q: np.ndarray, T: float, w: Weights) -> np.ndarray:
        n = self.n
        return exact_scores(self.E[:n], self.imp[:n], self.t_last[:n], q, T, w, self.decay)

    def retrieve(self, q: np.ndarray, T: float, k: int, w: Weights, refresh: bool = True) -> np.ndarray:
        t0 = time.perf_counter()
        ids = self._search(np.asarray(q, dtype=np.float32), float(T), int(k), w)
        t1 = time.perf_counter()
        self.stats["search_s"] += t1 - t0
        self.stats["queries"] += 1
        if refresh and len(ids):
            self.t_last[ids] = T
            self.t_hi = max(self.t_hi, float(T))
            self._push_times(ids, np.full(len(ids), float(T)))
            self._on_refresh(ids, float(T))
            self.stats["refresh_s"] += time.perf_counter() - t1
        return ids

    def _search(self, q: np.ndarray, T: float, k: int, w: Weights) -> np.ndarray:
        raise NotImplementedError

    def _on_refresh(self, ids: np.ndarray, T: float) -> None:
        pass

    def memory_bytes(self) -> int:
        return 0


class ExactStore(MemoryStore):
    name = "exact scan"

    def _search(self, q, T, k, w):
        return topk(self.exact(q, T, w), k)

    def memory_bytes(self):
        return self.n * self.dim * 4


# ---------------------------------------------------------------- HNSW helper

class _Hnsw:
    """hnswlib index with automatic resizing, live-element bookkeeping and safe queries."""

    def __init__(self, space: str, dim: int, M: int = 16, ef_construction: int = 200,
                 ef: int = 64, capacity: int = 1024, seed: int = 0, threads: int = 1):
        import hnswlib  # imported lazily: the exact stores work without it

        self.index = hnswlib.Index(space=space, dim=dim)
        self.index.init_index(max_elements=capacity, ef_construction=ef_construction, M=M,
                              random_seed=seed)
        self.index.set_num_threads(threads)
        self.ef = ef
        self.index.set_ef(ef)
        self.cap = capacity
        self.labels: set[int] = set()   # every label ever inserted (live or deleted)
        self.live: set[int] = set()

    def add(self, X: np.ndarray, labels: np.ndarray) -> None:
        labels = np.asarray(labels, dtype=np.int64)
        lab_list = labels.tolist()
        new = sum(1 for lab in lab_list if lab not in self.labels)
        if len(self.labels) + new > self.cap:
            self.cap = max(2 * self.cap, len(self.labels) + new)
            self.index.resize_index(self.cap)
        # an existing (deleted) label is unmarked and updated in place by hnswlib
        self.index.add_items(X, labels)
        self.labels.update(lab_list)
        self.live.update(lab_list)

    def delete(self, label: int) -> None:
        if label in self.live:
            self.index.mark_deleted(int(label))
            self.live.discard(label)

    def query(self, Q: np.ndarray, k: int, ef: int | None = None) -> np.ndarray:
        k = min(k, len(self.live))
        if k <= 0:
            return np.empty(0, dtype=np.int64)
        base = self.ef if ef is None else ef
        e = max(base, k)
        for _ in range(5):
            self.index.set_ef(e)
            try:
                labels, _ = self.index.knn_query(Q, k=k)
                self.index.set_ef(self.ef)
                return labels[0].astype(np.int64)
            except RuntimeError:  # too few reachable live elements (many deletions)
                e *= 4
        self.index.set_ef(self.ef)
        return np.fromiter(self.live, dtype=np.int64)  # degenerate tiny segment: return all

    def bytes(self) -> int:
        # vector + level-0 links (2M ints) + label per element; upper levels add ~1/M of that
        M = self.index.M
        return len(self.labels) * (self.index.dim * 4 + 2 * M * 4 + 8)


class RelevanceOnlyHNSW(MemoryStore):
    """Top-k by cosine similarity only (no recency, no importance)."""

    name = "relevance-only HNSW"

    def __init__(self, dim, decay=DECAY_PAPER, M=16, ef_construction=200, ef=64, seed=0,
                 threads=1, **kw):
        super().__init__(dim, decay, **kw)
        self.h = _Hnsw("ip", dim, M, ef_construction, ef, seed=seed, threads=threads)

    def _on_add(self, ids, T):
        self.h.add(self.E[ids], ids)

    def _search(self, q, T, k, w):
        return self.h.query(q[None, :], k)

    def memory_bytes(self):
        return self.h.bytes()


class OverfetchRerank(RelevanceOnlyHNSW):
    """LangChain-style: `fetch` nearest by similarity + `recent` newest, re-ranked by the full score.

    The candidates are re-scored with the full score (with min-max statistics
    estimated like the folded store does), which is the best any re-ranker can
    do with this candidate set. faithful=True instead reproduces
    TimeWeightedVectorStoreRetriever literally for raw weights: the `recent`
    memories that the vector search did not return get no relevance term
    (default_salience=None).
    """

    name = "overfetch+rerank"

    def __init__(self, dim, decay=DECAY_PAPER, fetch=100, recent=None, faithful=False,
                 norm_ef=32, **kw):
        super().__init__(dim, decay, **kw)
        self.fetch = fetch
        self.recent = recent
        self.faithful = faithful
        self.norm_ef = norm_ef
        self.name = f"overfetch+rerank (K={fetch}{', faithful' if faithful else ''})"

    def _search(self, q, T, k, w):
        fetched = self.h.query(q[None, :], self.fetch)
        n_recent = self.recent if self.recent is not None else k
        recent_ids = np.arange(max(0, self.n - n_recent), self.n)
        cand = np.union1d(fetched, recent_ids)
        if w.minmax:
            rel = self.E[cand] @ q
            lo = self.h.query(-q[None, :], 1, ef=self.norm_ef)
            rel_lo = min(float(rel.min()), float(self.E[lo[0]] @ q)) if len(lo) else float(rel.min())
            w = self.raw_weights(w, T, rel_lo, float(rel.max()))
        elif self.faithful:
            rel = (self.E[cand] @ q).astype(np.float64)
            rel[~np.isin(cand, fetched)] = 0.0
            s = (w.recency * recency(self.t_last[cand], T, self.decay)
                 + w.importance * self.imp[cand] + w.relevance * rel)
            return topk(s, k, ids=cand)
        s = exact_scores(self.E[cand], self.imp[cand], self.t_last[cand], q, T, w, self.decay)
        return topk(s, k, ids=cand)


class FoldedHNSW(MemoryStore):
    """Ours: exact full-score retrieval as MIPS over augmented vectors.

    Every memory lives in exactly one place: the write buffer (scored exactly at
    query time) or one HNSW segment. Each segment has its own time anchor, so the
    stored recency coordinate stays bounded (it lies in [1, decay**-seg_hours) in
    the hot segment and in (0, 1] in compacted ones). A refreshed memory is
    marked deleted in its segment and moves to the buffer; the buffer is flushed
    into the hot segment when full. When more than `max_closed` closed segments
    exist they are merged into one, re-anchored at its latest access time
    (forward-decay re-landmarking), which also drops deleted entries.
    Candidates from all segments and the buffer are merged and re-scored exactly.

    space="ip"  : hnswlib inner-product index on [e, s*rho, s*imp]
    space="l2"  : classic MIPS -> NN reduction (append sqrt(Phi^2 - |x|^2),
                  Bachrach et al. 2014)
    single_anchor=True disables segmentation (one anchor for all time), used
    only to show why anchoring matters.
    """

    def __init__(self, dim, decay=DECAY_PAPER, s=1.0, seg_hours=168.0, buffer_size=256,
                 max_closed=4, M=16, ef_construction=200, ef=64, norm_ef=32, space="ip",
                 single_anchor=False, seed=0, threads=1, fetch_factor=1, **kw):
        super().__init__(dim, decay, **kw)
        self.fetch_factor = fetch_factor   # candidates per segment = fetch_factor * k
        self.s = s
        self.seg_hours = seg_hours
        self.buffer_size = buffer_size
        self.max_closed = max_closed
        self.norm_ef = norm_ef
        self.hnsw_kw = dict(M=M, ef_construction=ef_construction, ef=ef, seed=seed, threads=threads)
        self.space = space
        self.single_anchor = single_anchor
        self.segments: list[dict] = []
        self._next_sid = 0
        self.seg_of: dict[int, dict] = {}
        self.loc = np.full(self.E.shape[0], -2, dtype=np.int64)   # -1 buffer, >= 0 segment id
        self.buffer: set[int] = set()
        tag = "ours" if space == "ip" else "ours, L2 reduction"
        if single_anchor:
            tag += ", single anchor"
        self.name = f"folded HNSW ({tag})"
        # norm bound for the L2 reduction: |x|^2 <= 1 + s^2 (rho_max^2 + 1)
        rho_max = decay ** (-(seg_hours + 24.0))
        self.phi2 = 1.0 + s * s * (rho_max ** 2 + 1.0) + 1e-3

    def _on_grow(self, new_capacity):
        loc = np.full(new_capacity, -2, dtype=np.int64)
        loc[: len(self.loc)] = self.loc
        self.loc = loc

    # -------------------------------------------------------------- segments
    def _make_segment(self, anchor: float, capacity: int = 1024) -> dict:
        dim = self.dim + 2 + (1 if self.space == "l2" else 0)
        seg = dict(sid=self._next_sid, anchor=anchor,
                   h=_Hnsw(self.space, dim, capacity=max(1024, capacity), **self.hnsw_kw))
        self.seg_of[self._next_sid] = seg
        self._next_sid += 1
        return seg

    def _hot(self, T: float) -> dict:
        if not self.segments:
            self.segments.append(self._make_segment(T))
        elif not self.single_anchor and T - self.segments[-1]["anchor"] >= self.seg_hours:
            self.segments.append(self._make_segment(T))
            if len(self.segments) - 1 > self.max_closed:
                self._compact()
        return self.segments[-1]

    def _compact(self) -> None:
        t0 = time.perf_counter()
        closed, hot = self.segments[:-1], self.segments[-1]
        ids = np.fromiter(sorted(set().union(*(seg["h"].live for seg in closed))), dtype=np.int64)
        anchor = float(self.t_last[ids].max()) if len(ids) else hot["anchor"]
        merged = self._make_segment(anchor, capacity=len(ids))
        if len(ids):
            merged["h"].add(self._vectors(ids, anchor), ids)
            self.loc[ids] = merged["sid"]
        for seg in closed:
            del self.seg_of[seg["sid"]]
        self.segments = [merged, hot]
        self.stats["compactions"] += 1
        self.stats["compact_s"] += time.perf_counter() - t0

    def _vectors(self, ids: np.ndarray, anchor: float) -> np.ndarray:
        X = fold_memories(self.E[ids], self.imp[ids], self.t_last[ids], anchor, self.decay, self.s)
        if self.space == "l2":
            extra = np.sqrt(np.maximum(self.phi2 - np.sum(X.astype(np.float64) ** 2, axis=1), 0.0))
            X = np.hstack([X, extra[:, None].astype(np.float32)])
        return X

    def flush(self, T: float) -> None:
        if not self.buffer:
            return
        seg = self._hot(T)
        ids = np.fromiter(sorted(self.buffer), dtype=np.int64)
        seg["h"].add(self._vectors(ids, seg["anchor"]), ids)
        self.loc[ids] = seg["sid"]
        self.buffer.clear()

    def _on_add(self, ids, T):
        self.loc[ids] = -1
        self.buffer.update(ids.tolist())
        if len(self.buffer) >= self.buffer_size:
            self.flush(T)

    def _on_refresh(self, ids, T):
        for i in ids.tolist():
            sid = self.loc[i]
            if sid >= 0:
                self.seg_of[sid]["h"].delete(i)
                self.loc[i] = -1
                self.buffer.add(i)
        if len(self.buffer) >= self.buffer_size:
            self.flush(T)

    # -------------------------------------------------------------- search
    def _query_vec(self, Q: np.ndarray) -> np.ndarray:
        if self.space == "l2":
            nrm = float(np.linalg.norm(Q))
            Q = np.append(Q / (nrm if nrm > 0 else 1.0), np.float32(0.0))
        return Q.astype(np.float32)[None, :]

    def _rel_range(self, q: np.ndarray) -> tuple[float, float]:
        lo, hi = np.inf, -np.inf
        if self.buffer:
            r = self.E[np.fromiter(self.buffer, dtype=np.int64)] @ q
            lo, hi = float(r.min()), float(r.max())
        pad = np.zeros(2, dtype=np.float32)
        for seg in self.segments:
            if not seg["h"].live:
                continue
            for sign in (1.0, -1.0):
                top = seg["h"].query(self._query_vec(np.concatenate([sign * q, pad])), 1, ef=self.norm_ef)
                if len(top):
                    r = float(self.E[top[0]] @ q)
                    lo, hi = min(lo, r), max(hi, r)
        return lo, hi

    def _search(self, q, T, k, w):
        if w.minmax:
            lo, hi = self._rel_range(q)
            w = self.raw_weights(w, T, lo, hi)
        parts = []
        if self.buffer:
            parts.append(np.fromiter(self.buffer, dtype=np.int64))
        for seg in self.segments:
            if not seg["h"].live:
                continue
            Q = fold_query(q, T, seg["anchor"], w, self.decay, self.s)
            parts.append(seg["h"].query(self._query_vec(Q), k * self.fetch_factor))
        if not parts:
            return np.empty(0, dtype=np.int64)
        cand = np.unique(np.concatenate(parts))
        s = exact_scores(self.E[cand], self.imp[cand], self.t_last[cand], q, T, w, self.decay)
        return topk(s, k, ids=cand)

    def memory_bytes(self):
        return sum(seg["h"].bytes() for seg in self.segments) + len(self.buffer) * self.dim * 4


class ExactRelevanceOnly(MemoryStore):
    """Relevance-only policy with an exact scan (isolates the policy from ANN error)."""

    name = "relevance-only (exact)"

    def _search(self, q, T, k, w):
        return topk((self.E[: self.n] @ q).astype(np.float64), k)


class ExactOverfetchRerank(MemoryStore):
    """Overfetch+rerank policy with an exact similarity search (isolates the policy from ANN error)."""

    def __init__(self, dim, decay=DECAY_PAPER, fetch=100, recent=None, **kw):
        super().__init__(dim, decay, **kw)
        self.fetch, self.recent = fetch, recent
        self.name = f"overfetch+rerank (K={fetch}, exact)"

    def _search(self, q, T, k, w):
        rel = (self.E[: self.n] @ q).astype(np.float64)
        fetched = topk(rel, self.fetch)
        n_recent = self.recent if self.recent is not None else k
        cand = np.union1d(fetched, np.arange(max(0, self.n - n_recent), self.n))
        if w.minmax:
            w = self.raw_weights(w, T, float(rel.min()), float(rel.max()))
        s = exact_scores(self.E[cand], self.imp[cand], self.t_last[cand], q, T, w, self.decay)
        return topk(s, k, ids=cand)


def make_store(kind: str, dim: int, **kw) -> MemoryStore:
    if kind.startswith("curve_"):   # general forgetting curves (curves.py, curve_stores.py)
        from .curve_stores import CURVE_KINDS
        return CURVE_KINDS[kind](dim, **kw)
    kinds = {
        "exact": ExactStore,
        "relonly": RelevanceOnlyHNSW,
        "rerank": OverfetchRerank,
        "folded": FoldedHNSW,
        "relonly_exact": ExactRelevanceOnly,
        "rerank_exact": ExactOverfetchRerank,
    }
    return kinds[kind](dim, **kw)

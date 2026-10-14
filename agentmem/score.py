"""The Generative Agents retrieval score and its exact inner-product form.

Park et al. (2023), Sec. 4.1, score a memory i for a query q at time T as

    score_i = a * recency_i + b * importance_i + c * relevance_i
    recency_i    = decay ** (T - t_i)        (t_i = last access time, in game hours)
    importance_i = poignancy_i / 10          (LLM rating 1..10)
    relevance_i  = cos(q, e_i)

after min-max normalizing each component over all memories (all weights are 1
in the paper; the public code uses recency 0.5, relevance 3, importance 2).

Exact folding. The decay is exponential, so (forward decay, Cormode et al. 2009)
decay**(T - t_i) = decay**(T - tau) * decay**(tau - t_i) for any anchor time tau:
the first factor depends only on the query, the second only on the memory.
For any construction scale s > 0,

    x_i = [ e_i,  s * decay**(tau - t_i),  s * importance_i ]          (memory side)
    Q   = [ c * q,  (a / s) * decay**(T - tau),  b / s ]               (query side)
    <Q, x_i> = a * recency_i + b * importance_i + c * relevance_i      (exactly)

Min-max normalization subtracts a per-query constant from each component and
divides it by a per-query range. The constants do not change the ranking and
the ranges only rescale the weights, so the normalized ranking equals the raw
ranking with weights (a / range_rec, b / range_imp, c / range_rel). Top-k
retrieval under either score is a maximum inner product search (MIPS) over
(d + 2)-dimensional vectors, with a, b, c and T chosen at query time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DECAY_PAPER = 0.995  # per game hour, Park et al. (2023), Sec. 4.1


@dataclass(frozen=True)
class Weights:
    recency: float = 1.0
    importance: float = 1.0
    relevance: float = 1.0
    minmax: bool = False   # min-max normalize each component first (as in the paper)

    def __str__(self) -> str:  # compact label for tables
        s = f"rec={self.recency:g},imp={self.importance:g},rel={self.relevance:g}"
        return s + (",minmax" if self.minmax else "")

    def raw(self, rec_range: float, imp_range: float, rel_range: float) -> "Weights":
        """Unnormalized weights that rank exactly like the min-max normalized score."""
        if not self.minmax:
            return self

        def f(a: float, r: float) -> float:
            return a / r if r > 1e-12 else 0.0  # a constant component does not affect ranking

        return Weights(f(self.recency, rec_range), f(self.importance, imp_range),
                       f(self.relevance, rel_range), minmax=False)


PROFILES: dict[str, Weights] = {
    # Park et al. (2023), Sec. 4.1: min-max normalized, all weights 1
    "paper": Weights(1.0, 1.0, 1.0, minmax=True),
    # public retrieve.py: gw = [0.5, 3, 2] for (recency, relevance, importance), normalized
    "code": Weights(0.5, 2.0, 3.0, minmax=True),
    # plain sum without normalization, as in LangChain's TimeWeightedVectorStoreRetriever
    "raw": Weights(1.0, 1.0, 1.0, minmax=False),
}


def recency(t_last: np.ndarray, T: float, decay: float = DECAY_PAPER) -> np.ndarray:
    return np.power(decay, T - np.asarray(t_last, dtype=np.float64))


def minmax(x: np.ndarray) -> np.ndarray:
    """Min-max scaling to [0, 1]; a constant vector maps to 0.5 (as normalize_dict_floats)."""
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo <= 0.0:
        return np.full(x.shape, 0.5)
    return (x - lo) / (hi - lo)


def exact_scores(E: np.ndarray, imp: np.ndarray, t_last: np.ndarray, q: np.ndarray,
                 T: float, w: Weights, decay: float = DECAY_PAPER) -> np.ndarray:
    """Generative-Agents score of every memory (float64)."""
    rec = recency(t_last, T, decay)
    im = np.asarray(imp, dtype=np.float64)
    rel = (E.astype(np.float32, copy=False) @ q.astype(np.float32, copy=False)).astype(np.float64)
    if w.minmax and len(rec):
        rec, im, rel = minmax(rec), minmax(im), minmax(rel)
    return w.recency * rec + w.importance * im + w.relevance * rel


def ranges(E: np.ndarray, imp: np.ndarray, t_last: np.ndarray, q: np.ndarray, T: float,
           decay: float = DECAY_PAPER) -> tuple[float, float, float]:
    """Exact (recency, importance, relevance) ranges over all memories."""
    rec = recency(t_last, T, decay)
    rel = E @ q
    return (float(rec.max() - rec.min()), float(np.max(imp) - np.min(imp)),
            float(rel.max() - rel.min()))


def fold_memories(E: np.ndarray, imp: np.ndarray, t_last: np.ndarray, anchor: float,
                  decay: float = DECAY_PAPER, s: float = 1.0) -> np.ndarray:
    rho = np.power(decay, anchor - np.asarray(t_last, dtype=np.float64))
    X = np.empty((E.shape[0], E.shape[1] + 2), dtype=np.float32)
    X[:, :-2] = E
    X[:, -2] = s * rho
    X[:, -1] = s * np.asarray(imp, dtype=np.float64)
    return X


def fold_query(q: np.ndarray, T: float, anchor: float, w: Weights,
               decay: float = DECAY_PAPER, s: float = 1.0) -> np.ndarray:
    """Query vector for raw (unnormalized) weights; use Weights.raw() first for min-max."""
    Q = np.empty(q.shape[0] + 2, dtype=np.float32)
    Q[:-2] = w.relevance * q
    Q[-2] = (w.recency / s) * decay ** (T - anchor)
    Q[-1] = w.importance / s
    return Q


def topk(scores: np.ndarray, k: int, ids: np.ndarray | None = None) -> np.ndarray:
    """Indices (or ids) of the k largest scores, highest first; ties broken by smaller id."""
    n = scores.shape[0]
    if ids is None:
        ids = np.arange(n)
    if n == 0:
        return ids[:0]
    k = min(k, n)
    if k < n:
        part = np.argpartition(-scores, k - 1)[:k]
        # include every element tied with the k-th score so tie-breaking is deterministic
        kth = scores[part].min()
        part = np.nonzero(scores >= kth)[0]
    else:
        part = np.arange(n)
    order = np.lexsort((ids[part], -scores[part]))
    return ids[part[order[:k]]]


def tie_aware_recall(retrieved: np.ndarray, true_scores: np.ndarray, k: int,
                     eps: float = 1e-6) -> float:
    """Fraction of the k retrieved items whose true score reaches the true k-th best score.

    Equal to the usual recall@k when there are no ties at the k-th position, and
    does not penalize a method for returning a different member of a tie.
    """
    if len(true_scores) == 0:
        return 1.0
    k = min(k, len(true_scores))
    kth = np.partition(true_scores, len(true_scores) - k)[len(true_scores) - k]
    got = np.asarray(retrieved[:k], dtype=np.int64)
    return float(np.sum(true_scores[got] >= kth - eps)) / k

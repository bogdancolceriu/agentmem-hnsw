"""Correctness tests: the folded inner product equals the Generative-Agents score,
and every approximate store matches the exact store when it is configured to be exact."""

import numpy as np
import pytest

from agentmem import (ExactStore, FoldedHNSW, OverfetchRerank, PROFILES, RelevanceOnlyHNSW,
                      Weights, exact_scores, fold_memories, fold_query, minmax, ranges,
                      tie_aware_recall, topk)
from agentmem.score import recency


def random_memories(n, d=32, seed=0, horizon=500.0):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(n, d)).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    imp = rng.integers(1, 11, size=n) / 10.0
    t = np.sort(rng.uniform(0, horizon, size=n))
    return E, imp, t


def unit(rng, d):
    q = rng.normal(size=d).astype(np.float32)
    return q / np.linalg.norm(q)


@pytest.mark.parametrize("s", [0.25, 1.0, 4.0])
@pytest.mark.parametrize("anchor", [0.0, 250.0, 500.0])
def test_fold_is_exact_raw(s, anchor):
    E, imp, t = random_memories(500)
    rng = np.random.default_rng(1)
    for w in (PROFILES["raw"], Weights(0.5, 2.0, 3.0), Weights(3.0, 1.0, 0.2)):
        q, T = unit(rng, E.shape[1]), 520.0
        ref = exact_scores(E, imp, t, q, T, w)
        X = fold_memories(E, imp, t, anchor, s=s)
        Q = fold_query(q, T, anchor, w, s=s)
        np.testing.assert_allclose(X.astype(np.float64) @ Q, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("name", ["paper", "code"])
def test_fold_is_exact_minmax(name):
    """Min-max normalized score = folded inner product with rescaled weights, minus a constant."""
    E, imp, t = random_memories(800, seed=5)
    rng = np.random.default_rng(2)
    w = PROFILES[name]
    for _ in range(20):
        q, T = unit(rng, E.shape[1]), 510.0
        ref = exact_scores(E, imp, t, q, T, w)
        wr = w.raw(*ranges(E, imp, t, q, T))
        ip = fold_memories(E, imp, t, 300.0).astype(np.float64) @ fold_query(q, T, 300.0, wr)
        diff = ip - ref
        assert np.ptp(diff) < 1e-4            # constant offset
        assert set(topk(ip, 30)) == set(topk(ref, 30))


def test_minmax_matches_reference_normalization():
    """Our normalization reproduces normalize_dict_floats from the reference retrieve.py."""
    def normalize_dict_floats(d, target_min, target_max):  # verbatim logic from retrieve.py
        min_val, max_val = min(d.values()), max(d.values())
        range_val = max_val - min_val
        if range_val == 0:
            for key in d:
                d[key] = (target_max - target_min) / 2
        else:
            for key, val in d.items():
                d[key] = ((val - min_val) * (target_max - target_min) / range_val + target_min)
        return d

    x = np.random.default_rng(0).normal(size=50)
    ref = normalize_dict_floats(dict(enumerate(x.tolist())), 0, 1)
    np.testing.assert_allclose(minmax(x), [ref[i] for i in range(50)])
    np.testing.assert_allclose(minmax(np.ones(5)), 0.5)


def test_topk_ties_are_deterministic():
    s = np.array([1.0, 3.0, 3.0, 2.0, 3.0])
    assert topk(s, 2).tolist() == [1, 2]
    assert topk(s, 4).tolist() == [1, 2, 4, 3]
    assert tie_aware_recall(np.array([4, 1]), s, 2) == 1.0


def fill(store, E, imp, t, batch=100):
    for i in range(0, len(E), batch):
        store.add(E[i:i + batch], imp[i:i + batch], t[i:i + batch])


@pytest.mark.parametrize("space", ["ip", "l2"])
@pytest.mark.parametrize("profile", ["raw", "paper", "code"])
def test_folded_matches_exact_with_refresh(space, profile):
    """A long query sequence with refresh-on-access gives the same trajectory as the exact store."""
    E, imp, t = random_memories(3000, horizon=400.0)
    d = E.shape[1]
    exact = ExactStore(d)
    folded = FoldedHNSW(d, s=1.0, seg_hours=48.0, buffer_size=64, max_closed=3, ef=400, M=32,
                        norm_ef=200, space=space)
    fill(exact, E, imp, t)
    fill(folded, E, imp, t)
    rng = np.random.default_rng(7)
    w = PROFILES[profile]
    agree = []
    for step in range(300):
        T = 400.0 + step * 0.5
        q = unit(rng, d)
        a = exact.retrieve(q, T, 10, w)
        b = folded.retrieve(q, T, 10, w)
        agree.append(len(set(a) & set(b)) / 10)
        if step % 50 == 0:  # memories keep arriving during the run
            new_E, new_imp, _ = random_memories(40, seed=step + 100)
            exact.add(new_E, new_imp, T)
            folded.add(new_E, new_imp, T)
    assert np.mean(agree) > 0.99, np.mean(agree)
    np.testing.assert_allclose(exact.t_last[: exact.n], folded.t_last[: folded.n])


def test_compaction_keeps_results_exact_and_bounds_segments():
    E, imp, t = random_memories(4000, horizon=2000.0, seed=11)
    d = E.shape[1]
    exact = ExactStore(d)
    folded = FoldedHNSW(d, seg_hours=48.0, buffer_size=100, max_closed=3, ef=400, M=32, norm_ef=200)
    rng = np.random.default_rng(12)
    agree = []
    for i in range(0, 4000, 100):        # memories arrive over 2000 hours, queries in between
        T = float(t[i + 99])
        exact.add(E[i:i + 100], imp[i:i + 100], t[i:i + 100])
        folded.add(E[i:i + 100], imp[i:i + 100], t[i:i + 100])
        for _ in range(3):
            q = unit(rng, d)
            a = exact.retrieve(q, T, 10, PROFILES["paper"])
            b = folded.retrieve(q, T, 10, PROFILES["paper"])
            agree.append(len(set(a) & set(b)) / 10)
        assert len(folded.segments) <= folded.max_closed + 2
    assert folded.stats["compactions"] > 0
    assert np.mean(agree) > 0.99, np.mean(agree)


def test_running_time_statistics():
    E, imp, t = random_memories(500, seed=3)
    st = ExactStore(E.shape[1])
    fill(st, E, imp, t)
    rng = np.random.default_rng(0)
    for step in range(50):
        st.retrieve(unit(rng, E.shape[1]), 600.0 + step, 10, PROFILES["paper"])
        assert st.t_lo() == st.t_last[: st.n].min()
        assert st.t_hi == st.t_last[: st.n].max()


def test_overfetch_with_everything_is_exact():
    E, imp, t = random_memories(800)
    d = E.shape[1]
    exact, ofr = ExactStore(d), OverfetchRerank(d, fetch=800, ef=800)
    fill(exact, E, imp, t)
    fill(ofr, E, imp, t)
    rng = np.random.default_rng(3)
    for name in ("raw", "paper"):
        for _ in range(30):
            q, w = unit(rng, d), PROFILES[name]
            assert (set(exact.retrieve(q, 600.0, 10, w, refresh=False))
                    == set(ofr.retrieve(q, 600.0, 10, w, refresh=False)))


def test_relevance_only_is_exact_for_pure_relevance():
    E, imp, t = random_memories(1500)
    d = E.shape[1]
    exact, rel = ExactStore(d), RelevanceOnlyHNSW(d, ef=400, M=32)
    fill(exact, E, imp, t)
    fill(rel, E, imp, t)
    rng = np.random.default_rng(4)
    w = Weights(0.0, 0.0, 1.0)
    hits = [len(set(exact.retrieve(q, 600.0, 10, w, refresh=False))
                & set(rel.retrieve(q, 600.0, 10, w, refresh=False))) / 10
            for q in (unit(rng, d) for _ in range(100))]
    assert np.mean(hits) > 0.99


def test_recency_is_hours_based():
    assert np.isclose(recency(np.array([0.0]), 100.0, 0.995)[0], 0.995 ** 100)


def test_backdated_insert_keeps_coordinates_bounded():
    """A batch with old timestamps must not flush recently refreshed memories into an old segment."""
    d = 16
    E, imp, t = random_memories(8, d=d, seed=3, horizon=10.0)
    st = FoldedHNSW(d, seg_hours=48.0, buffer_size=4, ef=64, M=16)
    st.add(E[:4], imp[:4], t[:4])                                   # flushed into a segment near t = 10
    st.retrieve(E[0], 500.0, 2, PROFILES["paper"])                  # two memories refreshed at t = 500
    st.add(E[4:6], imp[4:6], np.array([1.0, 2.0]))                  # backdated batch fills the buffer
    for i in range(st.n):
        sid = st.loc[i]
        if sid >= 0:
            assert st.t_last[i] - st.seg_of[sid]["anchor"] <= st.seg_hours + 1e-9

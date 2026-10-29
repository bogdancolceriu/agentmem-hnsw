"""Tests for general forgetting curves: the fit, the bookkeeping of uses and the folded store."""

import numpy as np
import pytest

from agentmem import (ACT_R, FADEMEM, PROFILES, CurveExact, CurveExactFit, CurveFolded, CurveOverfetch,
                      ExactStore, Weights, exponential, fit_expsum, hyperbolic)
from agentmem.curve_stores import pareto_max


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


def test_exponential_is_one_exact_term():
    f = fit_expsum(exponential(0.995), 8)
    assert f.R == 1 and np.isclose(f.w[0], 1.0) and np.isclose(f.lam[0], -np.log(0.995))
    ages = np.linspace(0, 5000, 101)
    np.testing.assert_allclose(f(ages), 0.995 ** ages, rtol=1e-12)


@pytest.mark.parametrize("curve", [ACT_R, hyperbolic(24.0), FADEMEM])
def test_eight_exponentials_fit_within_one_percent(curve):
    f = fit_expsum(curve, 8)
    rel, rng = f.errors()
    assert rel < 0.01 and rng < 0.005, (rel, rng)
    assert np.all(f.w >= 0)


def test_use_bookkeeping_matches_direct_sums():
    """y_ir updated in O(R) per use equals the direct sum over the whole history."""
    rng = np.random.default_rng(0)
    E, imp, t = random_memories(200, seed=1, horizon=100.0)
    st = CurveExactFit(E.shape[1], curve=ACT_R, R=8)
    st.add(E, imp, t)
    hist = [[ti] for ti in t]
    for step in range(60):
        T = 100.0 + step * 3.7
        ids = rng.choice(st.n, size=10, replace=False)
        st.touch(ids, T)
        for i in ids:
            hist[i].append(T)
    T = 400.0
    direct_fit = np.array([st.fit(T - np.asarray(h)).sum() for h in hist])
    direct_true = np.array([ACT_R(T - np.asarray(h)).sum() for h in hist])
    np.testing.assert_allclose(st.D_fit(T), direct_fit, rtol=1e-10)
    np.testing.assert_allclose(st.D_true(T), direct_true, rtol=1e-10)


def test_bulk_load_matches_incremental_uses():
    rng = np.random.default_rng(2)
    E, imp, t = random_memories(100, seed=2, horizon=50.0)
    a = CurveExactFit(E.shape[1])
    a.add(E, imp, t)
    hist = [[ti] for ti in t]
    for step in range(30):
        T = 50.0 + step
        ids = rng.choice(100, size=5, replace=False)
        a.touch(ids, T)
        for i in ids:
            hist[i].append(T)
    b = CurveExactFit(E.shape[1])
    b.load(E, imp, [np.asarray(h) for h in hist], T_now=90.0)
    np.testing.assert_allclose(a.y[:100], b.y[:100], rtol=1e-10)
    np.testing.assert_allclose(a.t_last[:100], b.t_last[:100])
    np.testing.assert_allclose(a.D_true(90.0), b.D_true(90.0), rtol=1e-12)


@pytest.mark.parametrize("anchor_shift", [0.0, 5.0, 300.0])
def test_folded_inner_product_is_the_fitted_score(anchor_shift):
    rng = np.random.default_rng(3)
    E, imp, t = random_memories(300, seed=3, horizon=200.0)
    st = CurveFolded(E.shape[1], recent_hours=1e9)            # everything stays in the recent part
    st.add(E, imp, t)
    st.touch(rng.choice(300, size=50, replace=False), 210.0)
    ids = np.arange(300)
    anchor = float(st.t_last[ids].max()) + anchor_shift
    X = st._vectors(ids, anchor).astype(np.float64)
    a, b, c = 0.7, 1.3, 2.1
    for T in (anchor, anchor + 1.0, anchor + 50.0):
        q = unit(rng, E.shape[1])
        g = np.exp(-st.lam * (T - anchor))
        Q = np.concatenate([c * q, a * g / st.s, [b / st.s]])
        ref = a * st.D_fit(T) + b * st.imp[:300] + c * (E @ q)
        np.testing.assert_allclose(X @ Q, ref, rtol=1e-4, atol=1e-5)


def test_paper_score_is_the_exponential_last_use_case():
    """With an exponential curve and only the last use, the curve stores rank like the paper's score."""
    E, imp, t = random_memories(1500, seed=4)
    d = E.shape[1]
    ref, cur = ExactStore(d), CurveExact(d, curve=exponential(0.995), frequency=False)
    ref.add(E, imp, t)
    cur.add(E, imp, t)
    rng = np.random.default_rng(5)
    for step in range(100):
        q, T = unit(rng, d), 500.0 + step
        a = ref.retrieve(q, T, 10, PROFILES["paper"])
        b = cur.retrieve(q, T, 10, PROFILES["paper"])
        assert list(a) == list(b)


@pytest.mark.parametrize("profile", ["paper", "raw"])
def test_folded_matches_exact_fit_through_refresh_aging_and_compaction(profile):
    """A long sequence of arrivals and retrievals: same trajectory as the exact scan, bounded coordinates."""
    E, imp, t = random_memories(4000, horizon=1500.0, seed=6)
    d = E.shape[1]
    exact = CurveExactFit(d)
    folded = CurveFolded(d, recent_hours=24.0, seg_hours=12.0, max_closed=3, ef=400, M=32, norm_ef=200)
    rng = np.random.default_rng(7)
    w = PROFILES[profile]
    agree = []
    for i in range(0, 4000, 100):
        T = float(t[i + 99])
        exact.add(E[i:i + 100], imp[i:i + 100], t[i:i + 100])
        folded.add(E[i:i + 100], imp[i:i + 100], t[i:i + 100])
        for j in range(4):
            q = unit(rng, d)
            a = exact.retrieve(q, T + j, 10, w)
            b = folded.retrieve(q, T + j, 10, w)
            agree.append(len(set(a) & set(b)) / 10)
        assert folded.coordinates_bounded()
        assert len(folded.segments) <= folded.max_closed + 2
    assert folded.stats["compactions"] > 0
    assert np.mean(agree) > 0.99, np.mean(agree)
    np.testing.assert_allclose(exact.y[: exact.n], folded.y[: folded.n], rtol=1e-9)


def test_overfetch_with_everything_is_exact():
    E, imp, t = random_memories(600, seed=8)
    d = E.shape[1]
    exact, ofr = CurveExactFit(d), CurveOverfetch(d, fetch=600, ef=600)
    exact.add(E, imp, t)
    ofr.add(E, imp, t)
    rng = np.random.default_rng(9)
    for w in (PROFILES["paper"], Weights(1.0, 0.5, 2.0)):
        for _ in range(20):
            q = unit(rng, d)
            assert (set(exact.retrieve(q, 600.0, 10, w, refresh=False))
                    == set(ofr.retrieve(q, 600.0, 10, w, refresh=False)))


def test_log_of_the_sum_only_with_the_exact_scan():
    """ACT-R's logarithm: the exact scan ranks by ln D, a folded store refuses it."""
    E, imp, t = random_memories(500, seed=14)
    d = E.shape[1]
    st = CurveExact(d, log=True)
    st.add(E, imp, t)
    st.touch(np.arange(0, 500, 7), 520.0)
    T = 530.0
    D = np.bincount(st.acc_mem[: st.n_acc], weights=ACT_R(T - st.acc_t[: st.n_acc]), minlength=st.n)
    np.testing.assert_allclose(st.D_true(T), np.log(D), rtol=1e-12)
    q = unit(np.random.default_rng(15), d)
    ref = CurveExact.combine(np.log(D), imp, (E @ q).astype(np.float64), PROFILES["paper"])
    np.testing.assert_allclose(st.exact(q, T, PROFILES["paper"]), ref, rtol=1e-12)
    for cls in (CurveFolded, CurveExactFit, CurveOverfetch):
        with pytest.raises(ValueError):
            cls(d, log=True)


def test_segments_need_a_long_enough_recent_window():
    with pytest.raises(ValueError):
        CurveFolded(8, recent_hours=6.0, seg_hours=12.0)


def test_pareto_fronts_give_the_exact_range():
    rng = np.random.default_rng(10)
    Z = rng.random((3000, 6)) ** 3
    hi, lo = pareto_max(Z), pareto_max(-Z)
    for _ in range(50):
        g = rng.random(6) + 1e-3
        D = Z @ g
        assert np.isclose(D.max(), (Z[hi] @ g).max()) and np.isclose(D.min(), (Z[lo] @ g).min())
    Zs = Z[:300]
    brute = {i for i in range(300)
             if not any(np.all(Zs[j] >= Zs[i]) and np.any(Zs[j] > Zs[i]) for j in range(300))}
    assert set(pareto_max(Zs).tolist()) == brute


def test_folded_with_searched_ranges_and_fronts():
    """Same trajectory as the exact scan when the ranges come from fronts and +-q searches."""
    E, imp, t = random_memories(3000, horizon=900.0, seed=12)
    d = E.shape[1]
    exact = CurveExactFit(d)
    folded = CurveFolded(d, recent_hours=24.0, seg_hours=24.0, max_closed=3, ef=400, M=32, norm_ef=400,
                         exact_ranges_below=0)
    rng = np.random.default_rng(13)
    agree = []
    for i in range(0, 3000, 100):
        T = float(t[i + 99])
        exact.add(E[i:i + 100], imp[i:i + 100], t[i:i + 100])
        folded.add(E[i:i + 100], imp[i:i + 100], t[i:i + 100])
        for j in range(4):
            q = unit(rng, d)
            a = exact.retrieve(q, T + j, 10, PROFILES["paper"])
            b = folded.retrieve(q, T + j, 10, PROFILES["paper"])
            agree.append(len(set(a) & set(b)) / 10)
    assert any(seg["fronts"] is not None for seg in folded.segments)
    assert np.mean(agree) > 0.99, np.mean(agree)

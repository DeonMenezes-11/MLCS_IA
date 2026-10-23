"""Correctness checks for the privacy mechanisms.

These are the 'controlled examples / correctness checks' the rubric asks for:
each mechanism is verified to be unbiased and, where it has a closed-form
variance, to match it empirically.
"""
import numpy as np
import pandas as pd
import pytest

from privpipe import dp, ldp, ppdm, compressive


# --------------------------------------------------------------------------- #
# Central DP primitives
# --------------------------------------------------------------------------- #
def test_laplace_scale_matches_theory():
    rng = np.random.default_rng(0)
    eps, sens = 0.5, 1.0
    draws = dp.laplace_mech(np.zeros(200000), sens, eps, rng)
    expected_std = np.sqrt(2) * sens / eps
    assert abs(draws.std() - expected_std) / expected_std < 0.03


def test_dp_histogram_unbiased():
    counts = np.array([100.0, 300.0, 50.0])
    reps = np.array([dp.dp_histogram(counts, 1.0, s) for s in range(400)])
    assert np.allclose(reps.mean(axis=0), counts, rtol=0, atol=6)


# --------------------------------------------------------------------------- #
# LDP frequency oracles
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("eps", [0.5, 1.0, 2.0])
def test_grr_unbiased_and_variance(eps):
    rng = np.random.default_rng(1)
    d, n = 5, 60000
    true = rng.integers(0, d, size=n)
    grr = ldp.GRR(d, eps, seed=2)
    est = grr.estimate_freq(grr.perturb(true))
    truth = np.bincount(true, minlength=d) / n
    # unbiased: estimate close to truth
    assert np.max(np.abs(est - truth)) < 0.02
    # variance tracks closed form (order of magnitude)
    reps = np.array([ldp.GRR(d, eps, seed=s).estimate_freq(
        ldp.GRR(d, eps, seed=s).perturb(true))[0] for s in range(40)])
    emp_var = reps.var()
    theo_var = grr.var_per_value(n)
    assert 0.3 < emp_var / theo_var < 3.0


def test_oue_unbiased():
    rng = np.random.default_rng(3)
    d, n = 6, 50000
    true = rng.integers(0, d, size=n)
    oue = ldp.UnaryEncoding(d, 1.5, optimized=True, seed=4)
    est = oue.estimate_freq(oue.perturb(true))
    truth = np.bincount(true, minlength=d) / n
    assert np.max(np.abs(est - truth)) < 0.02


# --------------------------------------------------------------------------- #
# LDP numeric mean oracles
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mech", ["duchi", "piecewise"])
def test_numeric_mean_unbiased(mech):
    rng = np.random.default_rng(5)
    x = rng.uniform(-1, 1, size=100000)
    out = ldp.duchi_mean(x, 1.0, seed=6) if mech == "duchi" else ldp.piecewise(x, 1.0, seed=6)
    assert abs(out.mean() - x.mean()) < 0.02


def test_piecewise_lower_variance_than_duchi_highexp():
    rng = np.random.default_rng(7)
    x = rng.uniform(-1, 1, size=100000)
    eps = 4.0
    d = ldp.duchi_mean(x, eps, seed=8)
    p = ldp.piecewise(x, eps, seed=8)
    # both unbiased; PM should have smaller spread of the per-user output at high eps
    assert p.var() < d.var()


# --------------------------------------------------------------------------- #
# k-anonymity family
# --------------------------------------------------------------------------- #
def test_mondrian_satisfies_k():
    rng = np.random.default_rng(9)
    n = 2000
    df = pd.DataFrame({
        "age": rng.integers(18, 70, n),
        "sex": rng.choice(["Male", "Female"], n),
        "race": rng.choice(["A", "B", "C"], n),
        "marital_status": rng.choice(["M", "S"], n),
        "education": rng.choice(["HS", "BS", "MS"], n),
        "native_country": rng.choice(["US", "IN"], n),
        "occupation": rng.choice(["x", "y", "z", "w"], n),
    })
    qi = ["age", "sex", "race", "marital_status", "education", "native_country"]
    k = 10
    anon = ppdm.mondrian(df, qi, k=k)
    assert ppdm.check_k_anonymity(anon, qi) >= k


def test_l_diversity_and_t_closeness_on_hand_table():
    # one homogeneous class (eq 0, all 'a') and one diverse class (eq 1)
    df = pd.DataFrame({
        "eq_class": [0, 0, 0, 1, 1, 1],
        "sex": ["a", "a", "a", "a", "b", "b"],
    })
    ld = ppdm.l_diversity(df, "sex")
    assert ld["min_l"] == 1 and ld["n_homogeneous_classes"] == 1
    hom = ppdm.homogeneity_attack(df, "sex")
    assert hom["disclosed_records"] == 3
    t = ppdm.t_closeness(df, "sex")
    assert 0.0 <= t <= 1.0


# --------------------------------------------------------------------------- #
# Output modification
# --------------------------------------------------------------------------- #
def test_label_only_removes_confidence():
    proba = np.array([[0.7, 0.3], [0.1, 0.9]])
    out = ppdm.label_only(proba)
    assert set(np.unique(out)) <= {0.0, 1.0}
    assert np.array_equal(out.argmax(1), proba.argmax(1))


# --------------------------------------------------------------------------- #
# Compressive privacy
# --------------------------------------------------------------------------- #
def test_dca_preserves_more_utility_than_random():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(10)
    n, d = 1500, 20
    y = rng.integers(0, 2, n)
    # informative signal aligned with y on first 3 dims
    X = rng.normal(size=(n, d))
    X[:, :3] += y[:, None] * 1.5
    p = 3
    Wd = compressive.dca(X, y, p, rho=1.0)
    Wr = compressive.random_projection(X, p, seed=1)
    def acc(W):
        Z = compressive.project(X, W)
        clf = LogisticRegression(max_iter=500).fit(Z[:1000], y[:1000])
        return clf.score(Z[1000:], y[1000:])
    assert acc(Wd) >= acc(Wr)

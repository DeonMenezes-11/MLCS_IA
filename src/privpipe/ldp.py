"""Local differential privacy (Module 3).

Frequency oracles for categorical attributes (GRR, SUE, OUE, OLH) and mean
oracles for numeric attributes (Laplace, Duchi 1-bit, Piecewise Mechanism),
plus an LDP Naive Bayes classifier built on GRR-perturbed features.

Every estimator is unbiased by construction; the unit tests check that the
empirical variance tracks the closed-form variance.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Categorical frequency oracles (3.1, 3.2)
# --------------------------------------------------------------------------- #
class GRR:
    """Generalised Randomised Response over a domain of size ``d``."""

    def __init__(self, d: int, eps: float, seed: int = 0):
        self.d = d
        self.eps = eps
        self.p = np.exp(eps) / (np.exp(eps) + d - 1)
        self.q = (1 - self.p) / (d - 1)
        self.rng = np.random.default_rng(seed)

    def perturb(self, values):
        values = np.asarray(values, dtype=int)
        out = values.copy()
        flip = self.rng.random(len(values)) >= self.p
        rand = self.rng.integers(0, self.d, size=len(values))
        out[flip] = rand[flip]
        return out

    def estimate_freq(self, perturbed):
        n = len(perturbed)
        counts = np.bincount(np.asarray(perturbed, int), minlength=self.d).astype(float)
        return (counts / n - self.q) / (self.p - self.q)

    def var_per_value(self, n: int) -> float:
        """Closed-form variance of each estimated frequency (Wang et al. 2017)."""
        return self.q * (1 - self.q) / (n * (self.p - self.q) ** 2)


class UnaryEncoding:
    """Symmetric (SUE) and Optimised (OUE) unary encoding frequency oracles."""

    def __init__(self, d: int, eps: float, optimized: bool = True, seed: int = 0):
        self.d = d
        self.eps = eps
        if optimized:  # OUE: p=1/2, q=1/(e^eps+1)
            self.p = 0.5
            self.q = 1.0 / (np.exp(eps) + 1)
        else:  # SUE: symmetric
            self.p = np.exp(eps / 2) / (np.exp(eps / 2) + 1)
            self.q = 1.0 / (np.exp(eps / 2) + 1)
        self.rng = np.random.default_rng(seed)

    def perturb(self, values):
        values = np.asarray(values, dtype=int)
        n = len(values)
        B = np.zeros((n, self.d), dtype=int)
        B[np.arange(n), values] = 1
        keep1 = self.rng.random((n, self.d)) < self.p
        keep0 = self.rng.random((n, self.d)) < self.q
        return np.where(B == 1, keep1.astype(int), keep0.astype(int))

    def estimate_freq(self, perturbed):
        n = perturbed.shape[0]
        counts = perturbed.sum(axis=0).astype(float)
        return (counts / n - self.q) / (self.p - self.q)

    def var_per_value(self, n: int) -> float:
        return self.q * (1 - self.q) / (n * (self.p - self.q) ** 2)


class OLH:
    """Optimised Local Hashing: hash the value into ``g = e^eps + 1`` buckets and
    apply GRR on the bucket. Low communication, good accuracy at larger eps."""

    def __init__(self, d: int, eps: float, seed: int = 0):
        self.d = d
        self.eps = eps
        self.g = max(2, int(round(np.exp(eps) + 1)))
        self.p = np.exp(eps) / (np.exp(eps) + self.g - 1)
        self.q = 1.0 / self.g
        self.rng = np.random.default_rng(seed)

    def _hash(self, value, seed):
        # simple universal-ish hash into g buckets
        return (np.uint64(2654435761) * np.uint64(int(value) + 1)
                + np.uint64(seed * 40503 + 1)) % np.uint64(self.g)

    def perturb(self, values):
        values = np.asarray(values, dtype=int)
        seeds = self.rng.integers(1, 2**31 - 1, size=len(values))
        reports = np.array([int(self._hash(v, s)) for v, s in zip(values, seeds)])
        flip = self.rng.random(len(values)) >= self.p
        rand = self.rng.integers(0, self.g, size=len(values))
        reports[flip] = rand[flip]
        return reports, seeds

    def estimate_freq(self, reports, seeds):
        n = len(reports)
        est = np.zeros(self.d)
        for v in range(self.d):
            support = np.array([int(self._hash(v, s)) for s in seeds])
            matches = (reports == support).mean()
            est[v] = (matches - 1.0 / self.g) / (self.p - self.q)
        return est


# --------------------------------------------------------------------------- #
# Numeric mean oracles (3.2, 3.3)
# --------------------------------------------------------------------------- #
def laplace_ldp(values, eps: float, lo: float = -1.0, hi: float = 1.0, seed: int = 0):
    """Per-user Laplace on a value scaled to [-1,1]; sensitivity = range = 2."""
    rng = np.random.default_rng(seed)
    v = np.clip(np.asarray(values, float), lo, hi)
    v = 2 * (v - lo) / (hi - lo) - 1
    out = v + rng.laplace(0.0, 2.0 / eps, size=v.shape)
    return out  # unbiased estimate of the scaled value; mean -> scaled mean


def duchi_mean(values, eps: float, seed: int = 0):
    """Duchi et al. 1-bit mechanism for a value in [-1,1]. Output in {-B, +B}."""
    rng = np.random.default_rng(seed)
    x = np.clip(np.asarray(values, float), -1, 1)
    B = (np.exp(eps) + 1) / (np.exp(eps) - 1)
    p_pos = ((np.exp(eps) - 1) / (2 * np.exp(eps) + 2)) * x + 0.5
    out = np.where(rng.random(x.shape) < p_pos, B, -B)
    return out  # E[out] = x


def piecewise(values, eps: float, seed: int = 0):
    """Piecewise Mechanism (Wang et al. 2019) for a value in [-1,1]. Lower
    variance than Duchi at moderate/high eps. Output in [-C, C], unbiased."""
    rng = np.random.default_rng(seed)
    x = np.clip(np.asarray(values, float), -1, 1)
    z = np.exp(eps / 2)
    C = (z + 1) / (z - 1)
    l = ((C + 1) / 2) * x - (C - 1) / 2
    r = l + (C - 1)
    out = np.empty_like(x)
    u = rng.random(x.shape)
    inside = u < (z / (z + 1))
    # inside [l, r]
    out[inside] = l[inside] + rng.random(inside.sum()) * (r[inside] - l[inside])
    # outside: choose left [-C,l] or right [r,C] proportional to length
    idx = np.where(~inside)[0]
    left_len = l[idx] + C
    right_len = C - r[idx]
    pick_left = rng.random(len(idx)) < left_len / (left_len + right_len)
    li = idx[pick_left]
    ri = idx[~pick_left]
    out[li] = -C + rng.random(len(li)) * (l[li] + C)
    out[ri] = r[ri] + rng.random(len(ri)) * (C - r[ri])
    return out  # E[out] = x


# --------------------------------------------------------------------------- #
# LDP Naive Bayes (3.4)
# --------------------------------------------------------------------------- #
class LDPNaiveBayes:
    """Naive Bayes whose class-conditional categorical distributions are
    reconstructed from GRR-perturbed features. Each user splits the budget over
    the features they report; the server holds the (public) class labels."""

    def __init__(self, eps: float = 2.0, n_bins: int = 8, seed: int = 0):
        self.eps = eps
        self.n_bins = n_bins
        self.seed = seed

    def _bin(self, X, fit=False):
        if fit:
            self.edges_ = []
            self.card_ = []
            for j in range(X.shape[1]):
                qs = np.quantile(X[:, j], np.linspace(0, 1, self.n_bins + 1))
                qs[0], qs[-1] = -np.inf, np.inf
                qs = np.unique(qs)  # -inf/inf stay distinct -> card >= 2 always
                self.edges_.append(qs)
                self.card_.append(len(qs) - 1)
        B = np.zeros_like(X, dtype=int)
        for j in range(X.shape[1]):
            B[:, j] = np.clip(np.digitize(X[:, j], self.edges_[j][1:-1]), 0,
                              self.card_[j] - 1)
        return B

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        B = self._bin(X, fit=True)
        n_feat = X.shape[1]
        eps_feat = self.eps / n_feat
        self.cond_ = []  # per feature: [n_classes, card]
        for j in range(n_feat):
            d = self.card_[j]
            grr = GRR(d, eps_feat, seed=self.seed + j)
            rep = grr.perturb(B[:, j])
            table = np.zeros((len(self.classes_), d))
            for ci, c in enumerate(self.classes_):
                freq = grr.estimate_freq(rep[y == c])
                freq = np.clip(freq, 1e-3, None)
                table[ci] = freq / freq.sum()
            self.cond_.append(np.log(table))
        self.log_prior_ = np.log(
            np.array([(y == c).mean() for c in self.classes_]))
        return self

    def predict_proba(self, X):
        B = self._bin(np.asarray(X, float))
        logp = np.tile(self.log_prior_, (len(B), 1))
        for j, table in enumerate(self.cond_):
            logp = logp + table[:, B[:, j]].T
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

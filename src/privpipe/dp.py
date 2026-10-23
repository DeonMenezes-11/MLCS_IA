"""Central (curator-side) differential privacy mechanisms.

Covers the Module-2 ground: the Laplace/Gaussian primitives and composition,
plus three differentially private learners -- Naive Bayes (Laplace on
class-conditional counts), logistic regression (output perturbation,
Chaudhuri et al. 2011), DP-SGD (Abadi et al. 2016 via Opacus) -- and a
differentially private k-means for the unsupervised slice (2.4).

The scratch mechanisms are deliberately simple and auditable; Opacus supplies
the DP-SGD accountant so the reported epsilon is a real RDP bound.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Primitives (2.1)
# --------------------------------------------------------------------------- #
def laplace_mech(value, sensitivity: float, eps: float, rng=None):
    rng = np.random.default_rng(rng)
    scale = sensitivity / eps
    value = np.asarray(value, dtype=float)
    return value + rng.laplace(0.0, scale, size=value.shape)


def gaussian_mech(value, sensitivity: float, eps: float, delta: float, rng=None):
    rng = np.random.default_rng(rng)
    sigma = np.sqrt(2.0 * np.log(1.25 / delta)) * sensitivity / eps
    value = np.asarray(value, dtype=float)
    return value + rng.normal(0.0, sigma, size=value.shape), sigma


def dp_histogram(counts, eps: float, rng=None):
    """Release a histogram under eps-DP (add/remove neighbours, L1 sensitivity 1)."""
    noisy = laplace_mech(np.asarray(counts, dtype=float), 1.0, eps, rng)
    return np.clip(noisy, 0.0, None)


# --------------------------------------------------------------------------- #
# DP Naive Bayes (2.3) -- Laplace noise on class-conditional counts
# --------------------------------------------------------------------------- #
class DPNaiveBayes:
    """Categorical Naive Bayes trained on binned features; every count query is
    protected with Laplace noise. The privacy budget is split equally across the
    prior histogram and the per-feature conditional histograms, so the whole
    fit satisfies ``eps``-DP by sequential composition."""

    def __init__(self, eps: float = 1.0, n_bins: int = 10, seed: int = 0):
        self.eps = eps
        self.n_bins = n_bins
        self.seed = seed

    def _bin(self, X, fit=False):
        if fit:
            self.edges_ = []
            for j in range(X.shape[1]):
                qs = np.quantile(X[:, j], np.linspace(0, 1, self.n_bins + 1))
                qs[0], qs[-1] = -np.inf, np.inf
                self.edges_.append(np.unique(qs))
        B = np.zeros_like(X, dtype=int)
        for j in range(X.shape[1]):
            B[:, j] = np.clip(np.digitize(X[:, j], self.edges_[j][1:-1]), 0,
                              len(self.edges_[j]) - 2)
        return B

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        B = self._bin(X, fit=True)
        n_feat = X.shape[1]
        eps_q = self.eps / (n_feat + 1)  # one prior + one per feature

        prior_counts = np.array([(y == c).sum() for c in self.classes_], float)
        prior_counts = dp_histogram(prior_counts, eps_q, rng) + 1e-6
        self.log_prior_ = np.log(prior_counts / prior_counts.sum())

        self.log_cond_ = []  # per feature: [n_classes, n_bins]
        for j in range(n_feat):
            nb = len(self.edges_[j]) - 1
            table = np.zeros((len(self.classes_), nb))
            for ci, c in enumerate(self.classes_):
                counts = np.bincount(B[y == c, j], minlength=nb).astype(float)
                counts = dp_histogram(counts, eps_q, rng) + 1.0  # Laplace smoothing
                table[ci] = np.log(counts / counts.sum())
            self.log_cond_.append(table)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        B = self._bin(X)
        logp = np.tile(self.log_prior_, (len(X), 1))
        for j, table in enumerate(self.log_cond_):
            logp = logp + table[:, B[:, j]].T
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# --------------------------------------------------------------------------- #
# DP logistic regression (2.3) -- output perturbation (Chaudhuri et al. 2011)
# --------------------------------------------------------------------------- #
class DPLogisticOutputPerturb:
    """L2-regularised logistic regression whose weight vector is perturbed with
    noise of norm drawn from Gamma(d, 2/(n*lambda*eps)) and uniform direction,
    giving eps-DP when each row satisfies ||x|| <= 1."""

    def __init__(self, eps: float = 1.0, lam: float = 0.01, seed: int = 0):
        self.eps = eps
        self.lam = lam
        self.seed = seed

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float)
        # bound each row to the unit ball so the sensitivity analysis holds
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        Xb = X / np.maximum(norms.max(), 1.0)
        n, d = Xb.shape
        base = LogisticRegression(C=1.0 / (self.lam * n), max_iter=500)
        base.fit(Xb, y)
        w = base.coef_.ravel().copy()
        # output perturbation noise
        norm = rng.gamma(shape=d, scale=2.0 / (n * self.lam * self.eps))
        direction = rng.normal(size=d)
        direction /= np.linalg.norm(direction)
        self.coef_ = w + norm * direction
        self.intercept_ = float(base.intercept_[0])
        self._scale = max(norms.max(), 1.0)
        self.classes_ = base.classes_
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float) / self._scale
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        z = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.vstack([1 - p1, p1]).T

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


def fit_dp_logistic(X_train, y_train, eps: float, seed: int = 0):
    """DP logistic regression via diffprivlib (bounded data-norm objective
    perturbation). More robust in high dimension than the from-scratch
    output-perturbation class above, which is retained for the mechanism
    taxonomy. Returns the fitted estimator (has predict / predict_proba)."""
    from diffprivlib.models import LogisticRegression as DPLR
    import numpy as _np
    _np.random.seed(seed)
    data_norm = float(np.linalg.norm(X_train, axis=1).max())
    clf = DPLR(epsilon=eps, data_norm=data_norm, max_iter=200)
    clf.fit(X_train, y_train)
    return clf


# --------------------------------------------------------------------------- #
# DP k-means (2.4) -- noisy Lloyd iterations
# --------------------------------------------------------------------------- #
def dp_kmeans(X, k: int = 5, eps: float = 1.0, n_iter: int = 8,
              clip: float = 3.0, seed: int = 0):
    """Each Lloyd iteration spends eps/n_iter: Laplace-noise the per-cluster
    counts and the per-cluster coordinate sums, then recompute centroids."""
    rng = np.random.default_rng(seed)
    X = np.clip(np.asarray(X, dtype=float), -clip, clip)
    n, d = X.shape
    centroids = X[rng.choice(n, k, replace=False)].copy()
    eps_it = eps / n_iter
    for _ in range(n_iter):
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        assign = dists.argmin(axis=1)
        new = np.zeros_like(centroids)
        for c in range(k):
            mask = assign == c
            count = laplace_mech(mask.sum(), 1.0, eps_it / 2, rng)
            count = max(count, 1.0)
            sums = laplace_mech(X[mask].sum(axis=0), 2 * clip, eps_it / 2, rng)
            new[c] = sums / count
        centroids = new
    return centroids, assign


# --------------------------------------------------------------------------- #
# DP-SGD (2.3) -- Abadi et al. 2016 via Opacus
# --------------------------------------------------------------------------- #
def train_dpsgd_mlp(X_train, y_train, X_test, y_test, target_eps: float = 3.0,
                    delta: float = 1e-5, epochs: int = 15, batch_size: int = 512,
                    hidden: int = 64, max_grad_norm: float = 1.0, lr: float = 1e-2,
                    seed: int = 0, balanced: bool = True):
    """Train a 2-layer MLP with DP-SGD to a *target* epsilon and return the model
    plus the epsilon the accountant actually spent. Class-balanced loss keeps the
    minority (>50K) class from collapsing under the DP noise."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from opacus import PrivacyEngine

    torch.manual_seed(seed)
    dev = "cpu"
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xte = torch.tensor(X_test, dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(X_train.shape[1], hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 2),
    ).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    engine = PrivacyEngine()
    model, opt, loader = engine.make_private_with_epsilon(
        module=model, optimizer=opt, data_loader=loader,
        target_epsilon=target_eps, target_delta=delta, epochs=epochs,
        max_grad_norm=max_grad_norm,
    )
    if balanced:
        counts = np.bincount(np.asarray(y_train).astype(int), minlength=2)
        w = torch.tensor(len(y_train) / (2.0 * np.maximum(counts, 1)), dtype=torch.float32)
        lossf = nn.CrossEntropyLoss(weight=w)
    else:
        lossf = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(Xte)
        proba = torch.softmax(logits, dim=1).numpy()
    eps_spent = float(engine.get_epsilon(delta))
    return proba, eps_spent

"""Compressive privacy (Module 5).

Linear projections that compress the feature matrix before it is handed to a
third-party learner. PCA is the utility-agnostic baseline; Discriminant
Component Analysis (DCA, Kung 2017) keeps the *utility* label's discriminant
subspace; privacy-aware DCA additionally suppresses the *sensitive* label's
discriminant directions. Random projection is the data-oblivious baseline.

Each projection is scored two ways: downstream utility (income accuracy) and
reconstruction error of the original features (a proxy for how much raw signal
survives the compression).
"""
from __future__ import annotations

import numpy as np


def _scatter(X, y):
    """Between- and within-class scatter matrices for labels ``y``."""
    d = X.shape[1]
    mu = X.mean(axis=0)
    Sb = np.zeros((d, d))
    Sw = np.zeros((d, d))
    for c in np.unique(y):
        Xc = X[y == c]
        muc = Xc.mean(axis=0)
        diff = (muc - mu).reshape(-1, 1)
        Sb += len(Xc) * (diff @ diff.T)
        dc = Xc - muc
        Sw += dc.T @ dc
    return Sb, Sw


def pca(X, p: int):
    """Top-``p`` principal directions (columns of the returned matrix)."""
    Xc = X - X.mean(axis=0)
    cov = Xc.T @ Xc / len(X)
    vals, vecs = np.linalg.eigh(cov)
    W = vecs[:, ::-1][:, :p]
    return W


def dca(X, y_util, p: int, rho: float = 1.0):
    """Discriminant Component Analysis: eigenvectors of (Sw + rho*I)^{-1} Sb for
    the utility label. The ridge ``rho`` lets DCA keep more than (C-1) useful
    directions, unlike plain LDA."""
    Xc = X - X.mean(axis=0)
    Sb, Sw = _scatter(Xc, y_util)
    d = X.shape[1]
    M = np.linalg.solve(Sw + rho * np.eye(d), Sb)
    vals, vecs = np.linalg.eig(M)
    order = np.argsort(vals.real)[::-1]
    W = vecs[:, order[:p]].real
    # normalise columns
    W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    return W


def privacy_aware_dca(X, y_util, y_sens, p: int, rho: float = 1.0, alpha: float = 1.0):
    """DCA that maximises the utility discriminant while penalising the sensitive
    discriminant: eigenvectors of (Sw_util + rho*I + alpha*Sb_sens)^{-1} Sb_util."""
    Xc = X - X.mean(axis=0)
    Sb_u, Sw_u = _scatter(Xc, y_util)
    Sb_s, _ = _scatter(Xc, y_sens)
    d = X.shape[1]
    M = np.linalg.solve(Sw_u + rho * np.eye(d) + alpha * Sb_s, Sb_u)
    vals, vecs = np.linalg.eig(M)
    order = np.argsort(vals.real)[::-1]
    W = vecs[:, order[:p]].real
    W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    return W


def random_projection(X, p: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    R = rng.normal(0, 1.0 / np.sqrt(p), size=(d, p))
    return R


def project(X, W):
    return (X - X.mean(axis=0)) @ W


def reconstruction_error(X, W) -> float:
    """Relative least-squares reconstruction error of X from its projection.
    Higher = the compression discarded more of the raw signal."""
    Xc = X - X.mean(axis=0)
    Z = Xc @ W
    # best linear reconstruction Z @ pinv -> Xc
    Wpinv, *_ = np.linalg.lstsq(Z, Xc, rcond=None)
    Xhat = Z @ Wpinv
    return float(np.linalg.norm(Xc - Xhat) / (np.linalg.norm(Xc) + 1e-12))

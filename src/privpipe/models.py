"""Baseline (non-private) classifiers used as the utility reference."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def make_model(name: str, seed: int = 0):
    name = name.lower()
    if name in ("lr", "logreg"):
        return LogisticRegression(max_iter=1000)
    if name in ("nb", "gnb"):
        return GaussianNB()
    if name in ("rf", "forest"):
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    if name in ("mlp", "nn"):
        return MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=300,
                             random_state=seed)
    raise ValueError(f"unknown model {name!r}")


BASELINES = ["lr", "nb", "rf", "mlp"]

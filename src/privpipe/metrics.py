"""Utility and privacy metrics shared across experiments."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
)


def utility(y_true, y_pred, y_score=None) -> dict:
    """Standard classification-utility bundle for an income model."""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_pos": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return out


def tpr_at_fpr(labels, scores, target_fpr: float = 0.01) -> float:
    """Attack TPR at a fixed low FPR -- the metric Carlini et al. (2022) argue
    membership-inference attacks should be judged on, rather than average AUC."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    neg = scores[labels == 0]
    pos = scores[labels == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    thr = np.quantile(neg, 1.0 - target_fpr)
    return float((pos >= thr).mean())


def mia_summary(member_scores, nonmember_scores) -> dict:
    """Summarise a membership-inference attack: AUC, advantage, TPR@1%FPR.

    ``*_scores`` are higher-means-more-likely-member confidence signals for
    the training ('member') and held-out ('non-member') records.
    """
    from sklearn.metrics import roc_auc_score as _auc
    s = np.concatenate([member_scores, nonmember_scores])
    y = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    auc = float(_auc(y, s)) if len(np.unique(y)) > 1 else float("nan")
    # Yeom et al. (2018) membership advantage = TPR - FPR at the optimal threshold,
    # approximated here by the balanced-accuracy form 2*AUC-1 is avoided in favour
    # of an explicit threshold at the midpoint of the pooled median.
    thr = np.median(s)
    tpr = float((member_scores >= thr).mean())
    fpr = float((nonmember_scores >= thr).mean())
    return {
        "auc": auc,
        "advantage": tpr - fpr,
        "tpr_at_1pct_fpr": tpr_at_fpr(y, s, 0.01),
    }


def mean_std(values) -> tuple:
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))

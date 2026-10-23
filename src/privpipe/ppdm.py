"""Privacy-preserving data mining (Module 4).

Mondrian k-anonymity with l-diversity and t-closeness checks on the published
table (4.3, 4.4), a homogeneity-attack demonstration, association-rule hiding
(4.2/4.5), and output-modification defences for the deployed model (4.5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import NUMERIC_COLS


# --------------------------------------------------------------------------- #
# Mondrian k-anonymity (4.3)
# --------------------------------------------------------------------------- #
def _choose_dim(part, qi_cols):
    """Pick the QI with the widest normalised spread to split on."""
    best, best_width = None, -1.0
    for c in qi_cols:
        if c in NUMERIC_COLS:
            width = part[c].max() - part[c].min()
            width = width / (abs(part[c].mean()) + 1.0)
        else:
            width = part[c].nunique() - 1
        if width > best_width:
            best, best_width = c, width
    return best, best_width


def mondrian(df, qi_cols, k: int = 5):
    """Greedy median-cut Mondrian. Returns a copy of ``df`` with generalised QI
    columns (numeric -> "lo-hi" range, categorical -> value set) and an
    ``eq_class`` id; every class holds at least ``k`` records."""
    df = df.reset_index(drop=True).copy()
    partitions = [df.index.to_numpy()]
    finished = []

    def allowable(idx, dim):
        part = df.loc[idx]
        if dim in NUMERIC_COLS:
            med = part[dim].median()
            left = idx[part[dim] <= med]
            right = idx[part[dim] > med]
        else:
            vals = sorted(part[dim].unique())
            half = vals[: len(vals) // 2]
            mask = part[dim].isin(half)
            left, right = idx[mask.to_numpy()], idx[(~mask).to_numpy()]
        return left, right

    while partitions:
        idx = partitions.pop()
        part = df.loc[idx]
        dim, width = _choose_dim(part, qi_cols)
        split_ok = False
        if width > 0:
            left, right = allowable(idx, dim)
            if len(left) >= k and len(right) >= k:
                partitions.extend([left, right])
                split_ok = True
        if not split_ok:
            finished.append(idx)

    out = df.copy()
    out["eq_class"] = -1
    for cid, idx in enumerate(finished):
        out.loc[idx, "eq_class"] = cid
        for c in qi_cols:
            if c in NUMERIC_COLS:
                lo, hi = df.loc[idx, c].min(), df.loc[idx, c].max()
                out.loc[idx, c] = f"{lo}-{hi}"
            else:
                vals = ",".join(sorted(map(str, df.loc[idx, c].unique())))
                out.loc[idx, c] = vals
    return out


def check_k_anonymity(df_anon, qi_cols) -> int:
    return int(df_anon.groupby(list(qi_cols)).size().min())


def l_diversity(df_anon, sensitive: str) -> dict:
    """Distinct-l: the minimum number of distinct sensitive values in any class."""
    g = df_anon.groupby("eq_class")[sensitive].nunique()
    homogeneous = int((g == 1).sum())
    return {"min_l": int(g.min()), "n_homogeneous_classes": homogeneous,
            "n_classes": int(df_anon["eq_class"].nunique())}


def t_closeness(df_anon, sensitive: str) -> float:
    """Max total-variation distance between each class's sensitive distribution
    and the global one (a categorical surrogate for EMD)."""
    global_dist = df_anon[sensitive].value_counts(normalize=True)
    cats = global_dist.index
    gvec = global_dist.reindex(cats).fillna(0).to_numpy()
    worst = 0.0
    for _, grp in df_anon.groupby("eq_class"):
        lvec = grp[sensitive].value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
        tv = 0.5 * np.abs(lvec - gvec).sum()
        worst = max(worst, tv)
    return float(worst)


def homogeneity_attack(df_anon, sensitive: str) -> dict:
    """Records in a class with a single sensitive value are disclosed outright."""
    g = df_anon.groupby("eq_class")[sensitive].nunique()
    bad_classes = g.index[g == 1]
    disclosed = int(df_anon["eq_class"].isin(bad_classes).sum())
    return {"disclosed_records": disclosed,
            "frac_disclosed": float(disclosed / len(df_anon))}


# --------------------------------------------------------------------------- #
# Association-rule hiding (4.2 / 4.5)
# --------------------------------------------------------------------------- #
def rule_support_confidence(trans, antecedent, consequent):
    """Support and confidence of {antecedent} -> {consequent} over boolean
    transaction columns in a DataFrame ``trans``."""
    a = trans[list(antecedent)].all(axis=1)
    ac = a & trans[consequent]
    support = float(ac.mean())
    conf = float(ac.sum() / max(a.sum(), 1))
    return support, conf


def hide_rule(trans, antecedent, consequent, target_conf: float, seed: int = 0):
    """Distort the minimum number of transactions (flip the consequent to 0 on
    rows supporting the rule) until confidence drops below ``target_conf``."""
    rng = np.random.default_rng(seed)
    t = trans.copy()
    a = t[list(antecedent)].all(axis=1)
    supporting = np.where((a & t[consequent]).to_numpy())[0]
    rng.shuffle(supporting)
    _, conf = rule_support_confidence(t, antecedent, consequent)
    flipped = 0
    for i in supporting:
        if conf < target_conf:
            break
        t.iloc[i, t.columns.get_loc(consequent)] = False
        _, conf = rule_support_confidence(t, antecedent, consequent)
        flipped += 1
    return t, {"flipped": flipped, "final_confidence": conf}


# --------------------------------------------------------------------------- #
# Output modification (4.5) -- defence for the deployed model
# --------------------------------------------------------------------------- #
def label_only(proba):
    """Release only the hard label: removes the confidence signal MIA exploits."""
    hard = proba.argmax(axis=1)
    out = np.zeros_like(proba)
    out[np.arange(len(hard)), hard] = 1.0
    return out


def rounded_confidence(proba, decimals: int = 1):
    """Release coarse (rounded) confidences, re-normalised to a distribution."""
    r = np.round(proba, decimals)
    r = np.clip(r, 1e-6, None)
    return r / r.sum(axis=1, keepdims=True)

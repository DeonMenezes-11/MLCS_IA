"""Privacy attacks used to *measure* what each defence actually buys.

Three attack families, each tied to a pipeline exposure point:

* membership inference (loss-threshold and shadow-model) -- the deployed-model API,
* attribute inference -- the "just drop the sensitive column" fallacy,
* quasi-identifier linkage -- the published-table exposure.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .metrics import mia_summary


# --------------------------------------------------------------------------- #
# Membership inference
# --------------------------------------------------------------------------- #
def _true_label_confidence(model, X, y) -> np.ndarray:
    """P(model assigns the record's true label). Higher ~ more member-like."""
    proba = model.predict_proba(X)
    y = np.asarray(y).astype(int)
    return proba[np.arange(len(y)), y]


def loss_threshold_mia(model, X_train, y_train, X_test, y_test, rng=None) -> dict:
    """Yeom-style attack: a record is guessed 'member' when the model is
    confident on its true label. We balance the member/non-member pools."""
    rng = np.random.default_rng(None if rng is None else rng)
    n = min(len(X_train), len(X_test))
    tr = rng.choice(len(X_train), n, replace=False)
    te = rng.choice(len(X_test), n, replace=False)
    members = _true_label_confidence(model, X_train[tr], y_train[tr])
    nonmembers = _true_label_confidence(model, X_test[te], y_test[te])
    return mia_summary(members, nonmembers)


def shadow_mia(target_model, model_factory, X_pool, y_pool,
               X_train, y_train, X_test, y_test,
               n_shadow: int = 8, seed: int = 0) -> dict:
    """Shokri-style shadow-model attack.

    ``target_model`` is the already-trained victim; ``model_factory()`` returns a
    fresh untrained classifier of the same family. Shadow models are trained on
    random halves of a pool; the attack model learns to separate 'in' from 'out'
    confidence vectors, then is applied to the target's own members (train) and
    non-members (test).
    """
    rng = np.random.default_rng(seed)
    atk_X, atk_y = [], []
    n_half = len(X_pool) // 2
    for _ in range(n_shadow):
        perm = rng.permutation(len(X_pool))
        in_idx, out_idx = perm[:n_half], perm[n_half:2 * n_half]
        sm = model_factory()
        sm.fit(X_pool[in_idx], y_pool[in_idx])
        for idx, lab in [(in_idx, 1), (out_idx, 0)]:
            proba = sm.predict_proba(X_pool[idx])
            feats = np.sort(proba, axis=1)[:, ::-1]  # sorted confidences
            atk_X.append(feats)
            atk_y.append(np.full(len(idx), lab))
    atk_X = np.vstack(atk_X)
    atk_y = np.concatenate(atk_y)
    attack = LogisticRegression(max_iter=1000)
    attack.fit(atk_X, atk_y)

    def sorted_conf(X):
        p = target_model.predict_proba(X)
        return np.sort(p, axis=1)[:, ::-1]

    members = attack.predict_proba(sorted_conf(X_train))[:, 1]
    nonmembers = attack.predict_proba(sorted_conf(X_test))[:, 1]
    return mia_summary(members, nonmembers)


# --------------------------------------------------------------------------- #
# Attribute inference
# --------------------------------------------------------------------------- #
def attribute_inference(df_train, df_test, feature_cols, sensitive: str,
                        seed: int = 0) -> dict:
    """Can an adversary reconstruct the sensitive attribute from the *released*
    (non-sensitive) features? Accuracy above the majority baseline = leakage."""
    from .data import CATEGORICAL_COLS, NUMERIC_COLS

    def enc(df):
        parts = []
        for c in feature_cols:
            if c in NUMERIC_COLS:
                parts.append(df[c].to_numpy(dtype=float).reshape(-1, 1))
            else:
                cats = sorted(set(df_train[c]) | set(df_test[c]))
                for v in cats:
                    parts.append((df[c].to_numpy() == v).astype(float).reshape(-1, 1))
        return np.hstack(parts)

    Xtr, Xte = enc(df_train), enc(df_test)
    ytr = df_train[sensitive].astype("category").cat.codes.to_numpy()
    # align test codes to train categories
    cats = df_train[sensitive].astype("category").cat.categories
    yte = df_test[sensitive].map({c: i for i, c in enumerate(cats)}).fillna(-1).astype(int).to_numpy()
    clf = RandomForestClassifier(n_estimators=120, random_state=seed, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = float((pred == yte).mean())
    majority = float(max(np.bincount(ytr)) / len(ytr))
    return {"attack_accuracy": acc, "majority_baseline": majority,
            "leakage": acc - majority}


# --------------------------------------------------------------------------- #
# Quasi-identifier linkage
# --------------------------------------------------------------------------- #
def qi_uniqueness(df, qi_cols, age_bin: int = 1) -> dict:
    """Re-identification risk of a released table: share of records that are
    unique on their quasi-identifiers, and the achieved k (min class size)."""
    work = df[qi_cols].copy()
    if "age" in work and age_bin > 1:
        work["age"] = (work["age"] // age_bin) * age_bin
    sizes = work.groupby(list(qi_cols)).size()
    rec_class = work.merge(sizes.rename("sz"), left_on=list(qi_cols), right_index=True)["sz"]
    return {
        "frac_unique": float((rec_class == 1).mean()),
        "frac_le_k2": float((rec_class <= 2).mean()),
        "achieved_k": int(sizes.min()),
        "n_classes": int(len(sizes)),
    }

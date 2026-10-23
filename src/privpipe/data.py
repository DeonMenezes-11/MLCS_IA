"""Loading and preprocessing for the UCI Adult (Census Income) dataset.

The module keeps two representations side by side:

* a *clean table* (one row per person, human-readable categorical values) that the
  anonymisation and local-DP code operates on, and
* a *modelling matrix* (one-hot + standardised) that the classifiers and the
  membership-inference attacks consume.

Keeping both lets the privacy mechanisms act at the layer where they naturally
live (k-anonymity on quasi-identifiers, LDP on raw attributes, compressive
privacy on the feature matrix) while every downstream model sees a consistent
encoding.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]

# fnlwgt is a census sampling weight, not a personal attribute -> excluded from modelling.
DROP_COLUMNS = ["fnlwgt"]

NUMERIC_COLS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_COLS = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]

# Quasi-identifiers used by the linkage attack and the k-anonymity family.
QI_COLS = ["age", "sex", "race", "marital_status", "education", "native_country"]
# Sensitive attributes.
SENSITIVE_ATTR = "sex"          # attribute-inference target
SENSITIVE_DIVERSITY = "occupation"  # l-diversity / t-closeness sensitive column
TARGET = "income"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_clean(raw_dir: str) -> pd.DataFrame:
    """Return the cleaned Adult table (train+test combined).

    Cleaning follows the convention used throughout the Adult literature:
    strip whitespace, treat '?' as missing, drop rows with any missing value,
    and map the income label to {0,1}.
    """
    frames = []
    for name, split in [("adult.data", "train"), ("adult.test", "test")]:
        path = os.path.join(raw_dir, name)
        # adult.test has a bogus first line and a trailing '.' on the label.
        skip = 1 if name == "adult.test" else 0
        df = pd.read_csv(
            path, header=None, names=RAW_COLUMNS, skiprows=skip,
            na_values=["?"], skipinitialspace=True, engine="python",
        )
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # The income column occasionally carries a trailing period (from adult.test).
    df[TARGET] = (
        df[TARGET].astype(str).str.replace(".", "", regex=False).str.strip()
    )
    df[TARGET] = (df[TARGET] == ">50K").astype(int)

    # Strip stray whitespace on all object columns.
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()

    df = df.dropna().reset_index(drop=True)
    df = df.drop(columns=DROP_COLUMNS)
    return df


@dataclass
class Dataset:
    """A reproducible train/test split with an aligned modelling matrix."""

    df_train: pd.DataFrame
    df_test: pd.DataFrame
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list
    scaler_mean: np.ndarray
    scaler_std: np.ndarray

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def _onehot(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """One-hot encode categoricals on the union of categories; standardise numerics
    using *train* statistics only (no leakage)."""
    cats = {}
    for c in CATEGORICAL_COLS:
        cats[c] = sorted(set(df_train[c]) | set(df_test[c]))

    def encode(df):
        parts = []
        names = []
        num = df[NUMERIC_COLS].to_numpy(dtype=float)
        parts.append(num)
        names.extend(NUMERIC_COLS)
        for c in CATEGORICAL_COLS:
            for v in cats[c]:
                parts.append((df[c].to_numpy() == v).astype(float).reshape(-1, 1))
                names.append(f"{c}={v}")
        return np.hstack(parts), names

    Xtr, names = encode(df_train)
    Xte, _ = encode(df_test)

    n_num = len(NUMERIC_COLS)
    mean = Xtr[:, :n_num].mean(axis=0)
    std = Xtr[:, :n_num].std(axis=0) + 1e-8
    Xtr[:, :n_num] = (Xtr[:, :n_num] - mean) / std
    Xte[:, :n_num] = (Xte[:, :n_num] - mean) / std
    return Xtr, Xte, names, mean, std


def build(raw_dir: str, seed: int = 42, test_size: float = 0.3) -> Dataset:
    """Clean, split (stratified) and encode the dataset deterministically."""
    df = load_clean(raw_dir)
    rng = np.random.default_rng(seed)
    # Stratified split by the income label for a stable positive rate.
    idx_pos = df.index[df[TARGET] == 1].to_numpy()
    idx_neg = df.index[df[TARGET] == 0].to_numpy()
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    n_te_pos = int(len(idx_pos) * test_size)
    n_te_neg = int(len(idx_neg) * test_size)
    test_idx = np.concatenate([idx_pos[:n_te_pos], idx_neg[:n_te_neg]])
    train_idx = np.concatenate([idx_pos[n_te_pos:], idx_neg[n_te_neg:]])
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)

    df_train = df.loc[train_idx].reset_index(drop=True)
    df_test = df.loc[test_idx].reset_index(drop=True)

    Xtr, Xte, names, mean, std = _onehot(df_train, df_test)
    return Dataset(
        df_train=df_train, df_test=df_test,
        X_train=Xtr, X_test=Xte,
        y_train=df_train[TARGET].to_numpy(), y_test=df_test[TARGET].to_numpy(),
        feature_names=names, scaler_mean=mean, scaler_std=std,
    )


def write_processed(raw_dir: str, out_dir: str, seed: int = 42) -> dict:
    """Persist the cleaned splits and record a manifest with sizes and hashes."""
    os.makedirs(out_dir, exist_ok=True)
    ds = build(raw_dir, seed=seed)
    train_path = os.path.join(out_dir, "adult_train_clean.csv")
    test_path = os.path.join(out_dir, "adult_test_clean.csv")
    ds.df_train.to_csv(train_path, index=False)
    ds.df_test.to_csv(test_path, index=False)
    manifest = {
        "seed": seed,
        "n_train": int(len(ds.df_train)),
        "n_test": int(len(ds.df_test)),
        "n_features_encoded": int(ds.n_features),
        "positive_rate_train": float(ds.y_train.mean()),
        "positive_rate_test": float(ds.y_test.mean()),
        "files": {
            "adult_train_clean.csv": _sha256(train_path),
            "adult_test_clean.csv": _sha256(test_path),
        },
    }
    with open(os.path.join(out_dir, "processed_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


if __name__ == "__main__":
    import sys
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw = os.path.join(here, "data", "raw")
    out = os.path.join(here, "data", "processed")
    print(json.dumps(write_processed(raw, out), indent=2))

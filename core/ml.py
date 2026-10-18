"""Klasifikasi fasies/litologi dengan Random Forest, validasi per blok kedalaman."""
from __future__ import annotations

import numpy as np
import pandas as pd


def block_split(n: int, test_fraction: float, block_size: int, seed: int = 42):
    """Bagi indeks 0..n-1 menjadi train/test per blok menerus.

    Sampel bersebelahan pada log sangat mirip; split acak per baris membuat
    akurasi tampak terlalu bagus. Di sini seluruh blok menerus masuk ke
    train atau test.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("Porsi test harus di antara 0 dan 1.")
    if block_size < 1:
        raise ValueError("Ukuran blok minimal 1.")
    n_blocks = int(np.ceil(n / block_size))
    if n_blocks < 2:
        raise ValueError("Data terlalu pendek untuk dibagi menjadi blok.")
    n_test = min(n_blocks - 1, max(1, int(round(n_blocks * test_fraction))))
    rng = np.random.default_rng(seed)
    test_blocks = set(rng.permutation(n_blocks)[:n_test].tolist())
    block_id = np.arange(n) // block_size
    is_test = np.array([b in test_blocks for b in block_id])
    return np.where(~is_test)[0], np.where(is_test)[0]


def build_dataset(df: pd.DataFrame, features: list[str], target: pd.Series):
    """Gabungkan fitur dan target; buang baris dengan NaN. Mengembalikan (X, y, idx)."""
    if not features:
        raise ValueError("Pilih minimal satu fitur.")
    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = pd.Series(target.to_numpy(), index=df.index)
    ok = X.notna().all(axis=1) & y.notna() & np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    return X[ok], y[ok], np.where(ok.to_numpy())[0]


def train_evaluate(X: pd.DataFrame, y: pd.Series, test_fraction=0.3,
                   block_size=50, n_estimators=200, max_depth=None, seed=42):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, classification_report)

    if y.nunique() < 2:
        raise ValueError("Target hanya punya satu kelas; tidak ada yang bisa dipelajari.")
    train_i, test_i = block_split(len(X), test_fraction, block_size, seed)
    Xtr, ytr = X.iloc[train_i], y.iloc[train_i]
    Xte, yte = X.iloc[test_i], y.iloc[test_i]
    if ytr.nunique() < 2:
        raise ValueError("Data latih hanya punya satu kelas. Perkecil ukuran blok.")
    unseen = sorted(set(yte.unique()) - set(ytr.unique()), key=str)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   class_weight="balanced", random_state=seed, n_jobs=1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    labels = sorted(set(y.unique()), key=str)
    cm = confusion_matrix(yte, pred, labels=labels)
    report = classification_report(yte, pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "model": model,
        "labels": labels,
        "accuracy": float(accuracy_score(yte, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "confusion": pd.DataFrame(cm, index=[f"Aktual {l}" for l in labels],
                                  columns=[f"Prediksi {l}" for l in labels]),
        "report": pd.DataFrame(report).T,
        "importance": pd.Series(model.feature_importances_, index=X.columns)
                        .sort_values(ascending=False),
        "n_train": int(len(train_i)), "n_test": int(len(test_i)),
        "train_idx": train_i, "test_idx": test_i,
        "unseen_classes": unseen,
        "class_counts": y.value_counts().sort_index(),
    }

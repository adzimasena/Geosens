"""Analisis fasies elektrofasies dari log sumur (clustering tanpa label)."""
from __future__ import annotations

import numpy as np
import pandas as pd

DEPTH_CANDIDATES = ["DEPTH", "DEPT", "MD", "DEPTH.M", "DEPT.M"]
DEFAULT_LOGS = ["GR", "RHOB", "NPHI", "RT"]
LOG_SCALE_CURVES = {"RT", "RES", "ILD", "LLD", "RDEEP"}  # resistivitas: pakai log10


def prepare_logs(df: pd.DataFrame):
    """Rapikan nama kolom, ubah ke numerik, urutkan menurut kedalaman.

    Mengembalikan (DataFrame, nama_kolom_kedalaman) atau (None, None).
    """
    if df is None or df.empty:
        return None, None
    out = df.copy()
    out.columns = [str(c).strip().upper() for c in out.columns]
    out = out.loc[:, ~out.columns.duplicated()]
    depth = next((c for c in DEPTH_CANDIDATES if c in out.columns), None)
    if depth is None:
        return None, None
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=[depth]).sort_values(depth)
    out = out.drop_duplicates(subset=[depth]).reset_index(drop=True)
    if out.empty:
        return None, None
    return out, depth


def is_constant(values: np.ndarray, rel_tol: float = 1e-9) -> bool:
    """Kurva dianggap konstan bila simpangan bakunya hanya sebatas galat pembulatan."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return True
    scale = max(1.0, float(np.max(np.abs(v))))
    return bool(np.std(v) <= rel_tol * scale)


def build_feature_matrix(df: pd.DataFrame, features: list[str]):
    """Matriks fitur ter-standardisasi. Resistivitas diubah ke log10.

    Mengembalikan (X, valid_mask). Baris dengan NaN/inf atau resistivitas <= 0
    ditandai tidak valid.
    """
    cols = []
    for f in features:
        v = df[f].to_numpy(dtype=float).copy()
        if f in LOG_SCALE_CURVES:
            v = np.where(v > 0, np.log10(np.where(v > 0, v, 1.0)), np.nan)
        cols.append(v)
    X = np.column_stack(cols)
    valid = np.all(np.isfinite(X), axis=1)
    if valid.sum() == 0:
        return X, valid
    for j, f in enumerate(features):
        col = X[valid, j]
        if is_constant(col):
            raise ValueError(f"Kurva {f} konstan pada data valid; tidak bisa dipakai.")
        X[:, j] = (X[:, j] - col.mean()) / col.std()
    return X, valid


def cluster_facies(df, depth_col, features, n_facies, seed=42, order_by=None):
    """KMeans pada fitur ter-standardisasi.

    Label diurutkan menurut rata-rata kurva `order_by` (default fitur pertama)
    agar penomoran stabil. Baris tak valid diberi label -1 ("No data").
    """
    from sklearn.cluster import KMeans

    if not features:
        raise ValueError("Pilih minimal satu kurva.")
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Kurva tidak ada: {', '.join(missing)}")
    X, valid = build_feature_matrix(df, features)
    n_valid = int(valid.sum())
    if n_valid < n_facies:
        raise ValueError("Jumlah sampel valid lebih sedikit dari jumlah fasies.")
    if n_facies < 2:
        raise ValueError("Jumlah fasies minimal 2.")

    km = KMeans(n_clusters=n_facies, n_init=10, random_state=seed).fit(X[valid])
    raw = km.labels_
    key = order_by if order_by in df.columns else features[0]
    means = [np.nanmean(df.loc[valid, key].to_numpy()[raw == k]) for k in range(n_facies)]
    rank = {old: new for new, old in enumerate(np.argsort(means), start=1)}
    labels = np.full(len(df), -1, dtype=int)
    labels[valid] = [rank[k] for k in raw]
    res = df[[depth_col]].copy()
    res["FACIES"] = labels
    return res


def sample_thickness(depth: np.ndarray) -> np.ndarray:
    """Ketebalan tiap sampel: batas di titik tengah kedalaman bersebelahan."""
    d = np.asarray(depth, dtype=float)
    if d.size == 0:
        return d
    if d.size == 1:
        return np.array([0.0])
    mid = (d[:-1] + d[1:]) / 2.0
    top = np.concatenate([[d[0] - (mid[0] - d[0])], mid])
    base = np.concatenate([mid, [d[-1] + (d[-1] - mid[-1])]])
    return base - top


def facies_summary(df, res, depth_col, features) -> pd.DataFrame:
    """Ringkasan per fasies: jumlah sampel, ketebalan, proporsi, rata-rata kurva."""
    th = sample_thickness(df[depth_col].to_numpy())
    tmp = df.copy()
    tmp["FACIES"] = res["FACIES"].to_numpy()
    tmp["_TH"] = th
    total = tmp.loc[tmp["FACIES"] > 0, "_TH"].sum()
    rows = []
    for k in sorted(tmp.loc[tmp["FACIES"] > 0, "FACIES"].unique()):
        sub = tmp[tmp["FACIES"] == k]
        row = {"Fasies": int(k), "Sampel": len(sub),
               "Ketebalan": float(sub["_TH"].sum()),
               "Proporsi (%)": float(100 * sub["_TH"].sum() / total) if total > 0 else np.nan}
        for f in features:
            row[f"{f} rata-rata"] = float(sub[f].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def facies_intervals(res: pd.DataFrame, depth_col: str) -> pd.DataFrame:
    """Interval menerus untuk tiap fasies (top/base dari batas titik tengah)."""
    d = res[depth_col].to_numpy(dtype=float)
    lab = res["FACIES"].to_numpy()
    if d.size == 0:
        return pd.DataFrame(columns=["Fasies", "Top", "Base", "Ketebalan", "Sampel"])
    if d.size == 1:
        return pd.DataFrame([{"Fasies": int(lab[0]), "Top": d[0], "Base": d[0],
                              "Ketebalan": 0.0, "Sampel": 1}])
    mid = (d[:-1] + d[1:]) / 2.0
    tops = np.concatenate([[d[0] - (mid[0] - d[0])], mid])
    bases = np.concatenate([mid, [d[-1] + (d[-1] - mid[-1])]])
    rows, start = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[start]:
            rows.append({"Fasies": int(lab[start]), "Top": float(tops[start]),
                         "Base": float(bases[i - 1]),
                         "Ketebalan": float(bases[i - 1] - tops[start]),
                         "Sampel": i - start})
            start = i
    return pd.DataFrame(rows)

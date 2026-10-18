
from pathlib import Path
import os
import tempfile

import joblib
from huggingface_hub import hf_hub_download
import lasio
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit


# =========================================================
# 1. KONFIGURASI
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    Path.home()
    / "Downloads"
    / "Force_2020_all_wells_train_test_blind_hidden_final"
)

MODEL_DIR = PROJECT_DIR / "models"

# Gunakan nama baru agar file lama tetap aman.
MODEL_PATH = MODEL_DIR / "lithology_model_v2.joblib"

# Model publik yang digunakan saat aplikasi berjalan di cloud.
HF_REPO_ID = "adzimasena/geosens-lithology-model"
HF_FILENAME = "lithology_model_v2.joblib"

LABEL_CURVE = "FORCE_2020_LITHOFACIES_LITHOLOGY"

FEATURE_CURVES = [
    "GR", "RHOB", "NPHI", "DTC", "RDEP",
    "RMED", "RSHA", "SP", "CALI",
]

MISSING_VALUES = [-999.25, -9999, -999.0]

LITHOLOGY_NAMES = {
    30000: "Sandstone",
    65000: "Shale",
    65030: "Sandstone/Shale",
    70000: "Limestone",
    70032: "Chalk",
    74000: "Dolomite",
    80000: "Marl",
    86000: "Anhydrite",
    88000: "Halite",
    90000: "Coal",
    93000: "Basement",
    99000: "Tuff",
}


# =========================================================
# 2. MEMBACA DAN MEMBERSIHKAN DATA LAS
# =========================================================

def load_las_dataset(data_dir=DATA_DIR):
    data_dir = Path(data_dir).expanduser()

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {data_dir}"
        )

    las_files = sorted(data_dir.glob("*.las"))

    if not las_files:
        raise FileNotFoundError(
            f"Tidak ada file LAS di: {data_dir}"
        )

    frames = []

    for path in las_files:
        try:
            las = lasio.read(
                str(path),
                ignore_header_errors=True,
            )

            curve_lookup = {
                curve.mnemonic.strip().upper(): curve.mnemonic
                for curve in las.curves
            }

            if LABEL_CURVE not in curve_lookup:
                continue

            frame = pd.DataFrame(index=range(len(las.index)))

            for feature in FEATURE_CURVES:
                mnemonic = curve_lookup.get(feature)

                if mnemonic is not None:
                    frame[feature] = np.asarray(
                        las[mnemonic],
                        dtype=float,
                    )

            if frame.empty:
                continue

            frame["LITHOLOGY"] = np.asarray(
                las[curve_lookup[LABEL_CURVE]],
                dtype=float,
            )

            frame["WELL"] = path.stem
            frames.append(frame)

        except Exception as exc:
            print(f"Gagal membaca {path.name}: {exc}")

    if not frames:
        raise ValueError(
            f"Tidak ditemukan data dengan label {LABEL_CURVE}."
        )

    data = pd.concat(frames, ignore_index=True)

    data = data.replace(
        MISSING_VALUES + [np.inf, -np.inf],
        np.nan,
    )

    # Gunakan hanya kode label yang dikenali.
    data = data[
        data["LITHOLOGY"].isin(LITHOLOGY_NAMES.keys())
    ].copy()

    features = [
        feature
        for feature in FEATURE_CURVES
        if feature in data.columns
        and data[feature].notna().any()
    ]

    if data.empty or not features:
        raise ValueError(
            "Data log atau label litologi tidak mencukupi."
        )

    # Hilangkan sampel tanpa data log sama sekali.
    data = data.dropna(subset=features, how="all")

    # Hilangkan fitur yang konstan atau seluruhnya kosong.
    features = [
        feature
        for feature in features
        if data[feature].nunique(dropna=True) > 1
    ]

    if not features:
        raise ValueError(
            "Tidak ada kurva log yang memiliki variasi."
        )

    # Imputasi nilai kosong menggunakan median dataset.
    data[features] = data[features].fillna(
        data[features].median()
    )

    data = data.dropna(
        subset=["LITHOLOGY", "WELL"]
    ).reset_index(drop=True)

    data["LITHOLOGY"] = data["LITHOLOGY"].astype(int)

    if data["WELL"].nunique() < 2:
        raise ValueError(
            "Diperlukan minimal dua sumur untuk training/testing."
        )

    return data, features


# =========================================================
# 3. MEMUAT MODEL YANG SUDAH ADA
# =========================================================

def load_saved_model(model_path=MODEL_PATH):
    """Memuat model lokal; jika tidak ada, unduh dari Hugging Face."""
    model_path = Path(model_path)

    if not model_path.is_file():
        # Jangan mengunduh model saat memvalidasi file sementara hasil training.
        if model_path != MODEL_PATH:
            return None

        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
        )
        model_path = Path(downloaded_path)

    saved = joblib.load(model_path)

    if not isinstance(saved, dict):
        raise ValueError("Format file model tidak valid.")

    required = {"model", "features", "metrics"}

    if not required.issubset(saved):
        raise ValueError("Komponen file model tidak lengkap.")

    model = saved["model"]
    features = saved["features"]
    metrics = saved["metrics"]

    if not hasattr(model, "predict"):
        raise ValueError("Model tidak memiliki fungsi predict.")

    if not isinstance(features, list) or not features:
        raise ValueError("Daftar fitur model tidak valid.")

    if not isinstance(metrics, dict):
        raise ValueError("Metadata model tidak valid.")

    return model, features, metrics


# =========================================================
# 4. MELATIH DAN MENYIMPAN MODEL
# =========================================================

@st.cache_resource
def train_lithology_model(data_dir=DATA_DIR):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Prioritaskan model lokal atau model publik di Hugging Face.
    try:
        saved = load_saved_model(MODEL_PATH)

        if saved is not None:
            model, features, metrics = saved
            return model, features, pd.DataFrame(), metrics

    except Exception as exc:
        # Jika dataset lokal tersedia, tetap izinkan training lokal.
        if not Path(data_dir).expanduser().is_dir():
            raise RuntimeError(
                "Model litologi tidak dapat dimuat dari Hugging Face. "
                "Pastikan repository dan file model tersedia, serta "
                "versi scikit-learn kompatibel."
            ) from exc

        print(
            f"Model tersimpan gagal dimuat: "
            f"{type(exc).__name__}: {exc}"
        )
        print("Dataset lokal ditemukan; mencoba melatih model ulang.")

    # Baca dataset.
    data, features = load_las_dataset(data_dir)

    # Pisahkan berdasarkan sumur untuk mengurangi data leakage.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(
            data[features],
            data["LITHOLOGY"],
            groups=data["WELL"],
        )
    )

    train = data.iloc[train_idx]
    test = data.iloc[test_idx]

    # Validasi bahwa data training memiliki beberapa kelas.
    if train["LITHOLOGY"].nunique() < 2:
        raise ValueError(
            "Data training hanya memiliki satu kelas litologi."
        )

    # Random Forest dengan kompleksitas yang dibatasi.
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train[features],
        train["LITHOLOGY"],
    )

    # Evaluasi.
    predictions = model.predict(test[features])

    labels = sorted(
        set(train["LITHOLOGY"].unique())
        | set(test["LITHOLOGY"].unique())
    )

    metrics = {
        "accuracy": accuracy_score(
            test["LITHOLOGY"],
            predictions,
        ),
        "report": classification_report(
            test["LITHOLOGY"],
            predictions,
            labels=labels,
            target_names=[
                LITHOLOGY_NAMES.get(code, str(code))
                for code in labels
            ],
            zero_division=0,
        ),
        "train_wells": int(train["WELL"].nunique()),
        "test_wells": int(test["WELL"].nunique()),
        "train_samples": int(len(train)),
        "test_samples": int(len(test)),
        "features": features.copy(),
        "model_type": "Random Forest",
        "n_estimators": 100,
        "max_depth": 20,
    }

    # Simpan ke file sementara, bukan langsung ke file utama.
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            prefix="lithology_v2_",
            suffix=".tmp",
            dir=MODEL_DIR,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        joblib.dump(
            {
                "model": model,
                "features": features,
                "metrics": metrics,
            },
            temp_path,
            compress=3,
        )

        # Pastikan file baru bisa dimuat kembali.
        check = load_saved_model(temp_path)

        if check is None:
            raise RuntimeError(
                "File model tidak ditemukan setelah penyimpanan."
            )

        # Ganti file utama hanya setelah validasi berhasil.
        os.replace(temp_path, MODEL_PATH)
        temp_path = None

    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    return model, features, data, metrics


# =========================================================
# 5. PREDIKSI LITOLOGI PADA DATA LOG BARU
# =========================================================

def predict_lithology(model, features, log_data):
    if log_data.empty:
        return pd.Series(
            index=log_data.index,
            dtype="object",
            name="Predicted Lithology",
        )

    missing = [
        feature
        for feature in features
        if feature not in log_data.columns
    ]

    if missing:
        raise ValueError(
            "Kurva log belum tersedia: " + ", ".join(missing)
        )

    X = log_data[features].copy()

    for feature in features:
        X[feature] = pd.to_numeric(
            X[feature],
            errors="coerce",
        )

    X = X.replace(
        MISSING_VALUES + [np.inf, -np.inf],
        np.nan,
    )

    valid = X.notna().any(axis=1)

    # Imputasi mengikuti median data input.
    # Jika satu kurva seluruhnya kosong, gunakan 0 sebagai
    # nilai fallback agar model tetap dapat memproses input.
    X = X.fillna(X.median()).fillna(0.0)

    result = pd.Series(
        index=log_data.index,
        dtype="object",
        name="Predicted Lithology",
    )

    result.loc[~valid] = "No valid log data"

    if valid.any():
        codes = model.predict(X.loc[valid])

        result.loc[valid] = [
            LITHOLOGY_NAMES.get(int(code), "Unknown")
            for code in codes
        ]

    return result

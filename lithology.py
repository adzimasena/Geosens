"""views/lithology.py

Halaman prediksi litologi (ML). Tampilan multi-track memakai mesin yang sama dengan
halaman Well Log (core/plotting.py), ditambah satu kolom litologi berwarna.
Dipanggil dari app.py lewat show_lithology().

Catatan transparansi:
- Hasil adalah PREDIKSI model machine learning, bukan ground truth.
- Data asli dari LAS tidak diubah; pengurutan kedalaman hanya dilakukan pada salinan.
- Aplikasi tidak mengonversi satuan secara otomatis.
"""

from __future__ import annotations

import html
import io
import itertools

import lasio
import numpy as np
import pandas as pd
import streamlit as st

from core.lithology import LITHOLOGY_NAMES, predict_lithology, train_lithology_model
from core.plotting import render_viewer

NONE_OPTION = "(tidak ada)"

# Alias hanya nilai awal pemetaan; pengguna dapat mengubahnya.
FEATURE_ALIASES = {
    "GR": ["GR", "GAMMA", "GAMMA_RAY"],
    "RHOB": ["RHOB", "RHOZ", "DEN", "DENSITY"],
    "NPHI": ["NPHI", "TNPH", "NPOR"],
    "DTC": ["DTC", "DT", "AC"],
    "RDEP": ["RDEP", "ILD", "LLD"],
    "RMED": ["RMED", "ILM"],
    "RSHA": ["RSHA"],
    "SP": ["SP"],
    "CALI": ["CALI", "CAL"],
}

LITH_COLORS = {
    "sandstone": "#F2C94C", "shale": "#7F8C7A", "sandstone/shale": "#C9B35C",
    "marl": "#76B7B2", "dolomite": "#AF7AC5", "limestone": "#5DADE2",
    "chalk": "#D5DBDB", "halite": "#F5B7B1", "anhydrite": "#C39BD3",
    "tuff": "#E59866", "coal": "#2C3E50", "basement": "#A04000",
}
FALLBACK_COLORS = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#17BECF"]


@st.cache_resource(show_spinner=False)
def _get_model():
    """Latih model sekali per proses server (hasil di-cache)."""
    return train_lithology_model()


def _auto_default(feature: str, curves: dict[str, np.ndarray]) -> str | None:
    for alias in FEATURE_ALIASES[feature]:
        for mnemonic in curves:
            if mnemonic.upper() == alias:
                return mnemonic
    return None


def _assign_colors(names: list[str]) -> dict[str, str]:
    fallback = itertools.cycle(FALLBACK_COLORS)
    return {n: LITH_COLORS.get(n.strip().lower()) or next(fallback) for n in names}


def _thickness_table(view: pd.DataFrame, depth_unit: str) -> pd.DataFrame:
    """Ketebalan per kelas.

    Metode: tiap sampel diberi ketebalan sebesar jarak ke sampel berikutnya;
    sampel terakhir memakai langkah median. Jarak sampel tidak harus seragam.
    """
    depth = view["DEPTH"].to_numpy(dtype=float)
    steps = np.diff(depth)
    positive = steps[steps > 0]
    last = float(np.median(positive)) if positive.size else np.nan
    dz = np.append(steps, last)
    tmp = pd.DataFrame({"cls": view["PREDICTED_LITHOLOGY"].to_numpy(), "dz": dz}).dropna(subset=["cls"])
    if tmp.empty:
        return pd.DataFrame(columns=["Litologi (prediksi ML)", "Sampel", "% sampel", f"Ketebalan ({depth_unit})"])
    out = tmp.groupby("cls").agg(Sampel=("dz", "size"), tebal=("dz", "sum")).reset_index()
    out["% sampel"] = (100 * out["Sampel"] / out["Sampel"].sum()).round(1)
    out["tebal"] = out["tebal"].round(2)
    out = out.rename(columns={"cls": "Litologi (prediksi ML)", "tebal": f"Ketebalan ({depth_unit})"})
    return out.sort_values("Sampel", ascending=False).reset_index(drop=True)


def show_lithology() -> None:
    st.title("Lithology Prediction")
    st.write("Prediksi litologi dari well log menggunakan model machine learning.")
    st.info(
        "Hasil di halaman ini adalah **prediksi model ML, bukan ground truth**. Model dilatih pada dataset "
        "FORCE 2020 (lihat core/lithology.py) dan belum divalidasi untuk sumur Anda. "
        "Tidak semua jenis litologi dapat dibedakan dengan pasti dari well log.",
        icon="ℹ️",
    )

    uploaded = st.file_uploader("Unggah file LAS", type=["las"], key="lithology_las_upload")
    if uploaded is None:
        st.info("Unggah file LAS untuk memulai prediksi litologi.")
        return

    # --- Baca LAS (data asli tidak diubah) ---
    try:
        las = lasio.read(io.StringIO(uploaded.getvalue().decode("utf-8", errors="replace")))
    except Exception as exc:
        st.error(f"Gagal membaca file LAS: {exc}")
        return

    depth = np.asarray(las.index, dtype=float)
    if depth.size == 0 or np.isnan(depth).all():
        st.error("Kedalaman pada file LAS kosong atau tidak valid.")
        return
    depth_unit = str(getattr(las, "index_unit", None) or las.curves[0].unit or "satuan?")

    curves = {c.mnemonic: np.asarray(c.data, dtype=float) for c in list(las.curves)[1:]}
    if not curves:
        st.error("File LAS tidak memiliki kurva selain kedalaman.")
        return

    file_id = f"{uploaded.name}_{uploaded.size}"

    # --- Pemetaan kurva (pengguna dapat mengoreksi) ---
    options = [NONE_OPTION] + list(curves)
    mapping: dict[str, str] = {}
    with st.expander("Pemetaan kurva LAS ke kurva model (periksa sebelum prediksi)", expanded=True):
        st.caption(
            "Nama kurva berbeda antar sumur. Pilihan awal ditebak dari nama, jadi **periksa** terutama "
            "resistivitas. Aplikasi tidak mengonversi satuan otomatis."
        )
        cols = st.columns(3)
        for i, feature in enumerate(FEATURE_ALIASES):
            default = _auto_default(feature, curves)
            idx = options.index(default) if default in options else 0
            choice = cols[i % 3].selectbox(feature, options, index=idx, key=f"lith_map_{file_id}_{feature}")
            if choice != NONE_OPTION:
                mapping[feature] = choice

    if not mapping:
        st.error("Belum ada kurva yang dipetakan.")
        return
    if len(set(mapping.values())) < len(mapping):
        st.warning("Satu kurva LAS dipetakan ke lebih dari satu kurva model. Pastikan ini disengaja.")

    log_data = pd.DataFrame({"DEPTH": depth, **{f: curves[m] for f, m in mapping.items()}})
    log_data = log_data.dropna(subset=["DEPTH"]).sort_values("DEPTH").reset_index(drop=True)
    if log_data["DEPTH"].duplicated().any():
        st.warning("Ada kedalaman duplikat pada file ini. Data tidak dihapus, tetapi hasil perlu diperiksa.")
    if "NPHI" in log_data and log_data["NPHI"].dropna().quantile(0.99) > 1.5:
        st.warning("NPHI tampak dalam persen (persentil 99 > 1.5). Aplikasi tidak mengonversi satuan otomatis.")

    d_lo, d_hi = float(log_data["DEPTH"].min()), float(log_data["DEPTH"].max())
    m1, m2, m3 = st.columns(3)
    m1.metric("Jumlah sampel", f"{len(log_data):,}")
    m2.metric("Rentang kedalaman", f"{d_lo:g} – {d_hi:g} {depth_unit}")
    m3.metric("Kurva dipetakan", len(mapping))

    # --- Jalankan prediksi ---
    if st.button("Jalankan Prediksi Litologi", type="primary"):
        st.session_state.pop("lith_result", None)
        with st.spinner("Memuat dataset FORCE 2020 dan melatih model (pertama kali bisa lama)..."):
            try:
                model, features, _training, metrics = _get_model()
            except Exception as exc:
                st.error(f"Gagal menyiapkan model: {exc}")
                model = None
        if model is not None:
            missing = [f for f in features if f not in log_data.columns]
            if missing:
                st.error("Kurva yang dibutuhkan model belum dipetakan: " + ", ".join(missing))
            else:
                try:
                    predictions = predict_lithology(model, features, log_data.copy())
                    pred = pd.Series(np.asarray(predictions, dtype=object))
                    if len(pred) != len(log_data):
                        raise ValueError(
                            f"Jumlah prediksi ({len(pred)}) tidak sama dengan jumlah sampel ({len(log_data)})."
                        )
                    pred.index = log_data.index
                    if isinstance(LITHOLOGY_NAMES, dict):
                        pred = pred.map(lambda v: LITHOLOGY_NAMES.get(v, v))
                    pred = pred.where(pred.isna(), pred.astype(str))
                except Exception as exc:
                    st.error(f"Prediksi gagal: {exc}")
                else:
                    result_df = log_data.copy()
                    result_df["PREDICTED_LITHOLOGY"] = pred
                    st.session_state["lith_result"] = {
                        "file_id": file_id, "df": result_df, "features": list(features),
                        "metrics": metrics, "depth_unit": depth_unit,
                    }

    result = st.session_state.get("lith_result")
    if not result or result["file_id"] != file_id:
        return

    df = result["df"]
    st.success("Prediksi selesai. Label di bawah adalah **prediksi ML**, bukan ground truth.")
    n_nan = int(df["PREDICTED_LITHOLOGY"].isna().sum())
    if n_nan:
        st.warning(f"{n_nan:,} sampel tidak diprediksi (kemungkinan ada fitur yang kosong). Tidak ada nilai yang diisi otomatis.")

    class_names = sorted(df["PREDICTED_LITHOLOGY"].dropna().unique().tolist())
    if not class_names:
        st.error("Tidak ada sampel yang berhasil diprediksi.")
        return
    colors = _assign_colors(class_names)

    with st.expander("Metrik model dari proses pelatihan"):
        st.caption("Metrik ini dari data latih/uji model, bukan validasi pada sumur yang Anda unggah.")
        st.write(result["metrics"])

    legend = " ".join(
        f'<span style="display:inline-block;margin-right:14px;">'
        f'<span style="display:inline-block;width:12px;height:12px;background:{colors[n]};'
        f'border:1px solid #888;margin-right:5px;vertical-align:middle;"></span>{html.escape(n)}</span>'
        for n in class_names
    )
    st.markdown(legend, unsafe_allow_html=True)

    # --- Tampilan multi-track yang sama dengan halaman Well Log ---
    rng = render_viewer(
        df, "DEPTH", prefix="lv", signature_extra=file_id,
        litho={"col": "PREDICTED_LITHOLOGY", "classes": class_names, "colors": colors, "title": "LITHOLOGY (ML)"},
    )
    top, bottom = rng if rng else (d_lo, d_hi)
    sub = df[(df["DEPTH"] >= top) & (df["DEPTH"] <= bottom)]
    if sub.empty:
        return

    st.subheader("Statistik litologi (prediksi ML)")
    st.dataframe(_thickness_table(sub, result["depth_unit"]), hide_index=True, width="stretch")
    st.caption(
        "Ketebalan = jarak ke sampel berikutnya dijumlahkan per kelas (sampel terakhir memakai langkah median). "
        "Persentase dihitung dari sampel yang berhasil diprediksi pada interval terpilih."
    )

    st.subheader("Hasil per kedalaman")
    st.dataframe(sub.head(1000), hide_index=True, width="stretch")
    if len(sub) > 1000:
        st.caption("Tabel menampilkan 1.000 baris pertama. Unduhan berisi seluruh interval terpilih.")
    st.download_button(
        "⬇️ Unduh hasil (CSV, prediksi ML)",
        data=sub.to_csv(index=False).encode("utf-8"),
        file_name="lithology_ml_prediction.csv",
        mime="text/csv",
    )
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import facies as F
from core import ml as M


def show_ml(df):
    st.subheader("Machine Learning: klasifikasi fasies/litologi")
    st.write(
        "Melatih Random Forest untuk memprediksi sebuah label dari kurva log. "
        "Validasi memakai blok kedalaman menerus, bukan baris acak, supaya skor "
        "tidak terlalu optimistis."
    )

    data, depth = F.prepare_logs(df)
    if data is None:
        st.error("Kolom kedalaman (DEPTH, DEPT, atau MD) tidak ditemukan atau kosong.")
        return

    # Sumber label
    sources = []
    fr = st.session_state.get("facies_result")
    if fr is not None and len(fr["data"]) == len(data) and fr["data"][depth].equals(data[depth]):
        sources.append("Hasil menu Facies")
    sources.append("Kolom label di data")
    src = st.radio("Sumber label (target)", sources)

    if src == "Hasil menu Facies":
        y_all = fr["labels"]["FACIES"].where(fr["labels"]["FACIES"] > 0)
        st.warning(
            "Label dari menu Facies dibuat dari kurva yang sama. Model hanya akan "
            "meniru K-Means dan skor tinggi tidak membuktikan ketepatan geologi. "
            "Label dari data inti atau interpretasi geolog lebih bermakna."
        )
    else:
        raw = df.copy()
        raw.columns = [str(c).strip().upper() for c in raw.columns]
        raw = raw.loc[:, ~raw.columns.duplicated()]
        label_cols = [c for c in raw.columns if c != depth
                      and (not pd.api.types.is_float_dtype(raw[c])
                           or raw[c].nunique(dropna=True) <= 20)]
        if not label_cols:
            st.error("Tidak ada kolom yang cocok sebagai label (kolom kategori atau berkelas sedikit).")
            return
        tcol = st.selectbox("Kolom label", label_cols)
        # selaraskan dengan urutan data bersih
        raw[depth] = pd.to_numeric(raw[depth], errors="coerce")
        raw = raw.dropna(subset=[depth]).sort_values(depth).drop_duplicates(subset=[depth])
        y_all = raw[tcol].reset_index(drop=True)
        y_all = y_all.where(y_all.astype(str).str.strip() != "")

    tcol = locals().get("tcol")
    feat_opts = [c for c in data.columns if c != depth and c != tcol
                 and data[c].notna().sum() > 1]
    features = st.multiselect("Fitur (kurva)", feat_opts,
                              default=[c for c in F.DEFAULT_LOGS if c in feat_opts])

    c1, c2, c3 = st.columns(3)
    test_frac = c1.slider("Porsi data uji", 0.1, 0.5, 0.3, 0.05)
    block = c2.number_input("Ukuran blok (sampel)", min_value=5, value=50, step=5)
    trees = c3.number_input("Jumlah pohon", min_value=10, max_value=500, value=200, step=10)
    seed = st.number_input("Seed acak", min_value=0, value=42, step=1, key="ml_seed")

    if st.button("Latih dan evaluasi model"):
        try:
            X, y, idx = M.build_dataset(data, features, y_all)
            if len(X) < 2 * int(block):
                st.error("Sampel valid terlalu sedikit untuk ukuran blok ini.")
                return
            res = M.train_evaluate(X, y, float(test_frac), int(block), int(trees),
                                   seed=int(seed))
        except ValueError as e:
            st.error(str(e))
            return
        res["features"] = list(features)
        res["target_source"] = src
        res["params"] = {"porsi_uji": test_frac, "ukuran_blok": int(block),
                         "pohon": int(trees), "seed": int(seed)}
        st.session_state.ml_result = res

    res = st.session_state.get("ml_result")
    if res is None:
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Akurasi (uji)", f"{res['accuracy']:.1%}")
    m2.metric("Akurasi seimbang", f"{res['balanced_accuracy']:.1%}")
    m3.metric("Sampel latih / uji", f"{res['n_train']} / {res['n_test']}")
    if res["unseen_classes"]:
        st.warning("Kelas di data uji yang tidak ada di data latih (tidak mungkin diprediksi): "
                   + ", ".join(map(str, res["unseen_classes"])))
    st.write("Jumlah sampel per kelas")
    st.dataframe(res["class_counts"].rename("Sampel").to_frame(), use_container_width=True)
    st.write("Matriks konfusi (data uji)")
    st.dataframe(res["confusion"], use_container_width=True)
    st.write("Laporan per kelas")
    st.dataframe(res["report"], use_container_width=True)

    imp = res["importance"]
    fig = go.Figure(go.Bar(x=imp.values[::-1], y=imp.index[::-1], orientation="h"))
    fig.update_layout(title="Pentingnya fitur", height=300)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Skor berlaku untuk sumur ini saja. Model yang dilatih pada satu sumur "
               "belum tentu berlaku di sumur lain; uji pada sumur terpisah sebelum dipakai.")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import facies as F

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2",
           "#9D755D", "#72B7B2", "#EECA3B"]


def show_facies(df):
    st.subheader("Analisis Fasies (elektrofasies)")
    st.write(
        "Mengelompokkan sampel log yang mirip menjadi beberapa elektrofasies "
        "dengan K-Means. Kelompok ini murni statistik dan belum tentu sama dengan "
        "fasies geologi; interpretasikan bersama data inti, cutting, atau deskripsi."
    )

    data, depth = F.prepare_logs(df)
    if data is None:
        st.error("Kolom kedalaman (DEPTH, DEPT, atau MD) tidak ditemukan atau kosong.")
        return

    usable = [c for c in data.columns if c != depth and data[c].notna().sum() > 1
              and not F.is_constant(data[c].to_numpy())]
    defaults = [c for c in F.DEFAULT_LOGS if c in usable]
    features = st.multiselect("Kurva yang dipakai", usable, default=defaults)
    c1, c2 = st.columns(2)
    k = c1.number_input("Jumlah fasies", min_value=2, max_value=8, value=3, step=1)
    seed = c2.number_input("Seed acak", min_value=0, value=42, step=1)
    st.caption("Resistivitas (RT dan sejenisnya) otomatis diubah ke log10 sebelum "
               "dikelompokkan. Fasies diberi nomor menurut rata-rata kurva pertama "
               "yang dipilih, dari kecil ke besar.")

    if st.button("Hitung fasies"):
        try:
            res = F.cluster_facies(data, depth, features, int(k), seed=int(seed),
                                   order_by=features[0] if features else None)
        except ValueError as e:
            st.error(str(e))
            return
        st.session_state.facies_result = {
            "labels": res, "data": data, "depth": depth,
            "features": list(features), "k": int(k), "seed": int(seed),
        }

    r = st.session_state.get("facies_result")
    if r is None:
        return
    if len(r["data"]) != len(data) or not r["data"][depth].equals(data[depth]):
        st.warning("Data aktif berbeda dari saat fasies dihitung. Klik Hitung fasies lagi.")

    res, d, feats = r["labels"], r["depth"], r["features"]
    n_nodata = int((res["FACIES"] < 0).sum())
    if n_nodata:
        st.info(f"{n_nodata} sampel tidak dapat dikelompokkan (data kosong atau RT ≤ 0).")

    fig = go.Figure()
    for k_ in sorted(res["FACIES"].unique()):
        sub = res[res["FACIES"] == k_]
        name = "No data" if k_ < 0 else f"Fasies {k_}"
        color = "#BBBBBB" if k_ < 0 else PALETTE[(int(k_) - 1) % len(PALETTE)]
        fig.add_trace(go.Scatter(x=[0] * len(sub), y=sub[d], mode="markers", name=name,
                                 marker=dict(color=color, size=6, symbol="square")))
    fig.update_yaxes(autorange="reversed", title_text="Depth")
    fig.update_xaxes(visible=False)
    fig.update_layout(title="Kolom fasies", height=600, width=300)
    left, right = st.columns([1, 3])
    left.plotly_chart(fig, use_container_width=True)

    summ = F.facies_summary(r["data"], res, d, feats)
    right.write("Ringkasan per fasies")
    right.dataframe(summ, use_container_width=True)
    st.caption("Ketebalan sepanjang lubang bor (MD), satuan sama dengan kolom kedalaman.")

    st.write("Interval menerus")
    iv = F.facies_intervals(res, d)
    st.dataframe(iv, use_container_width=True)

    out = r["data"].copy()
    out["FACIES"] = res["FACIES"].to_numpy()
    st.download_button("Unduh data + fasies (CSV)",
                       out.to_csv(index=False).encode("utf-8"),
                       file_name="facies_result.csv", mime="text/csv")

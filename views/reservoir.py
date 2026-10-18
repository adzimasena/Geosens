"""views/reservoir.py

Halaman Reservoir Evaluation. Perhitungan ada di core/reservoir.py.
Memakai hasil Petrophysics terakhir dari st.session_state["petro_result"].

Semua hasil adalah ESTIMASI berbasis log, bukan nilai terukur. Cut-off tidak punya nilai bawaan.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.reservoir import (
    RESERVOIR_DOCS,
    ReservoirError,
    ReservoirParams,
    build_intervals,
    compute_reservoir,
)


def _f(x, nd: int = 2) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:,.{nd}f}"


def _show_docs() -> None:
    st.subheader("Rumus, asumsi, dan batasan")
    st.caption("Tidak ada cut-off universal. Semua nilai cut-off berasal dari Anda dan sebaiknya "
               "dikalibrasi dengan core atau uji produksi.")
    for doc in RESERVOIR_DOCS:
        with st.expander(doc["name"]):
            for eq in doc["equations"]:
                st.latex(eq)
            st.markdown(f"**Variabel dan satuan:** {doc['variables']}")
            st.markdown(f"**Asumsi:** {doc['assumptions']}")
            st.markdown(f"**Batasan:** {doc['limitations']}")
            st.markdown(f"**Referensi:** {doc['reference']}")


def _plot(t: pd.DataFrame, depth_col: str) -> None:
    tracks = [c for c in ("VSH", "PHIE", "SW") if c in t.columns]
    if not tracks:
        return
    fig = make_subplots(rows=1, cols=len(tracks), shared_yaxes=True, subplot_titles=tracks,
                        horizontal_spacing=0.04)
    for i, c in enumerate(tracks, start=1):
        cur = t[[depth_col, c]].dropna()
        fig.add_trace(go.Scatter(x=cur[c], y=cur[depth_col], mode="lines", name=c, line=dict(width=1.5),
                                 hovertemplate=c + ": %{x}<br>Depth: %{y}<extra></extra>"), row=1, col=i)
        if c == "VSH":
            fig.update_xaxes(range=[0, 1], row=1, col=i)
    for _, r in build_intervals(t, "RES_FLAG").iterrows():
        fig.add_hrect(y0=r["Top"], y1=r["Base"], fillcolor="#2e8b57", opacity=0.15, line_width=0,
                      row="all", col="all")
    for _, r in build_intervals(t, "PAY_FLAG").iterrows():
        fig.add_hrect(y0=r["Top"], y1=r["Base"], fillcolor="#e67e00", opacity=0.25, line_width=0,
                      row="all", col="all")
    fig.update_yaxes(title_text="Depth", autorange="reversed", row=1, col=1)
    fig.update_layout(height=650, showlegend=False, title="Log dengan interval net reservoir (hijau) dan net pay (oranye)")
    st.plotly_chart(fig, width="stretch")


def show_reservoir() -> None:
    st.write("Tentukan net reservoir, net pay, ketebalan, dan kelas kualitas dari hasil Petrophysics "
             "dengan cut-off yang Anda tentukan.")
    st.info(
        "Semua hasil adalah **estimasi berbasis log**. Cut-off **tidak punya nilai bawaan**: kolom yang "
        "dikosongkan berarti kriteria itu tidak dipakai, dan bila tidak ada cut-off sama sekali, "
        "net reservoir ditandai 'cannot calculate'.",
        icon="ℹ️",
    )

    petro = st.session_state.get("petro_result")
    if not petro:
        st.warning("Belum ada hasil Petrophysics. Buka menu Petrophysics, isi parameter, lalu klik "
                   "'Hitung petrofisika'.")
        _show_docs()
        return

    depth_col = petro["depth_col"]
    ptab: pd.DataFrame = petro["out"].table
    dmin = float(ptab[depth_col].min())
    dmax = float(ptab[depth_col].max())
    sig = (len(ptab), dmin, dmax, tuple(ptab.columns))
    h = hashlib.md5(repr(sig).encode()).hexdigest()[:8]
    st.caption(f"Memakai hasil Petrophysics terakhir: {len(ptab):,} sampel, kedalaman {dmin:,.2f} sampai {dmax:,.2f}. "
               "Kalau data diganti, hitung ulang Petrophysics dulu.")

    # --- 1. Zona ---
    with st.expander("1. Zona evaluasi", expanded=True):
        use_zone = st.checkbox("Batasi ke interval kedalaman tertentu", value=False, key=f"rv_usezone_{h}")
        zt = zb = None
        if use_zone:
            c1, c2 = st.columns(2)
            zt = c1.number_input("Top zona", value=dmin, format="%.2f", key=f"rv_zt_{h}")
            zb = c2.number_input("Base zona", value=dmax, format="%.2f", key=f"rv_zb_{h}")
        st.caption("Satuan kedalaman mengikuti data (m atau ft). Ketebalan dihitung sepanjang lubang bor, "
                   "bukan ketebalan vertikal sejati.")

    # --- 2. Cut-off reservoir ---
    with st.expander("2. Cut-off net reservoir", expanded=True):
        st.caption("Kosongkan kolom untuk tidak memakai kriteria itu. Sampel lolos jika SEMUA kriteria yang diisi "
                   "terpenuhi. Nilai cut-off bergantung formasi dan lapangan Anda; aplikasi tidak menyarankan angka.")
        c1, c2 = st.columns(2)
        vsh_max = c1.number_input("Vsh maksimum (fraction)", value=None, min_value=0.0, max_value=1.0,
                                  format="%.3f", key=f"rv_vsh_{h}")
        phie_min = c2.number_input("PHIE minimum (fraction)", value=None, min_value=0.0, max_value=0.999,
                                   format="%.3f", key=f"rv_phie_{h}")

    # --- 3. Net pay ---
    with st.expander("3. Cut-off net pay (opsional)", expanded=True):
        sw_max = st.number_input("Sw maksimum (fraction)", value=None, min_value=0.001, max_value=1.0,
                                 format="%.3f", key=f"rv_sw_{h}",
                                 help="Net pay = net reservoir dengan Sw <= nilai ini. Kosongkan jika tidak diminta.")
        st.caption("Net pay dari log belum menyatakan zona produktif; itu butuh data uji fluida dan produksi.")

    # --- 4. Kelas kualitas ---
    with st.expander("4. Kelas kualitas reservoir berdasarkan PHIE (opsional)"):
        st.caption("Tiga batas PHIE (fraction, naik) membentuk empat kelas: Buruk, Cukup, Baik, Sangat baik. "
                   "Kelas ini relatif dan hanya memakai PHIE. Kosongkan semuanya jika tidak diminta.")
        c1, c2, c3 = st.columns(3)
        b1 = c1.number_input("Batas 1", value=None, min_value=0.0, max_value=1.0, format="%.3f", key=f"rv_b1_{h}")
        b2 = c2.number_input("Batas 2", value=None, min_value=0.0, max_value=1.0, format="%.3f", key=f"rv_b2_{h}")
        b3 = c3.number_input("Batas 3", value=None, min_value=0.0, max_value=1.0, format="%.3f", key=f"rv_b3_{h}")

    if st.button("Hitung evaluasi reservoir", type="primary"):
        bounds = [b1, b2, b3]
        class_bounds = None
        if any(b is not None for b in bounds):
            if any(b is None for b in bounds):
                st.warning("Isi ketiga batas kelas atau kosongkan semuanya. Kelas kualitas tidak dihitung.")
            else:
                class_bounds = (float(b1), float(b2), float(b3))
        params = ReservoirParams(vsh_max=vsh_max, phie_min=phie_min, sw_max=sw_max,
                                 zone_top=zt, zone_base=zb, class_bounds=class_bounds)
        try:
            out = compute_reservoir(ptab, depth_col, params)
        except ReservoirError as exc:
            st.error(f"Perhitungan tidak dapat dijalankan: {exc}")
        except Exception as exc:  # kesalahan tak terduga ditampilkan, tidak disembunyikan
            st.error(f"Terjadi kesalahan tak terduga: {type(exc).__name__}: {exc}")
            st.exception(exc)
        else:
            st.session_state["res_result"] = {"sig": sig, "out": out, "depth_col": depth_col}

    result = st.session_state.get("res_result")
    if result and result["sig"] == sig:
        out = result["out"]
        s = out.summary
        t = out.table
        st.divider()
        st.subheader("Hasil")
        st.caption("Hasil memakai parameter saat tombol 'Hitung' terakhir ditekan. Satuan ketebalan = satuan kedalaman data.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Gross", _f(s["gross"]))
        m2.metric("Net reservoir", _f(s["net_reservoir"]))
        m3.metric("Net pay", _f(s["net_pay"]))
        m4.metric("Net/Gross", _f(s["ntg"], 3))
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("PHIE rata-rata (net)", _f(s["phie_avg"], 3))
        m6.metric("Vsh rata-rata (net)", _f(s["vsh_avg"], 3))
        m7.metric("Sw rata-rata (net, PV)", _f(s["sw_avg_pv"], 3))
        m8.metric("Tebal tak dapat dievaluasi", _f(s["unknown"]))

        st.markdown("**Status per langkah**")
        rows = [{"Langkah": k, "Status": "✅ dihitung" if v == "calculated" else f"⚪ {v}"}
                for k, v in out.status.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        if out.warnings:
            with st.expander(f"Peringatan ({len(out.warnings)})", expanded=True):
                for w in out.warnings:
                    st.warning(w)

        st.markdown("**Interval net reservoir**")
        if "RES_FLAG" not in t.columns:
            st.caption("Tidak tersedia karena net reservoir belum dapat dihitung.")
        elif out.intervals.empty:
            st.caption("Tidak ada interval yang lolos cut-off.")
        else:
            st.dataframe(out.intervals, hide_index=True, width="stretch")

        if not out.class_summary.empty:
            st.markdown("**Ketebalan per kelas kualitas (PHIE)**")
            st.dataframe(out.class_summary, hide_index=True, width="stretch")

        _plot(t, result["depth_col"])
        st.caption("Pita hijau: net reservoir. Pita oranye: net pay.")

        st.markdown("**Tabel per kedalaman**")
        st.dataframe(t, hide_index=True, width="stretch")
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Unduh tabel (CSV)", t.to_csv(index=False).encode("utf-8"),
                           file_name="reservoir_evaluation.csv", mime="text/csv")
        d2.download_button(
            "⬇️ Unduh parameter dan ringkasan (JSON)",
            json.dumps({"params": out.params, "summary": s, "status": out.status},
                       indent=2, ensure_ascii=False, default=str).encode("utf-8"),
            file_name="reservoir_summary.json", mime="application/json")
        st.caption("Kolom: THICK (tebal sel), CELL_TOP/CELL_BASE (batas sel), RES_FLAG, PAY_FLAG, "
                   "EVAL_OK (False = input cut-off kosong), PHIE_CLASS.")

    st.divider()
    _show_docs()
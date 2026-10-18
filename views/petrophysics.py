"""views/petrophysics.py

Halaman Petrophysical Analysis. Perhitungan ada di core/petrophysics.py.
Dipanggil lewat show_petrophysics(df). Jika df tidak diberikan, dibaca dari st.session_state["well_data"].

Semua hasil adalah ESTIMASI berbasis log, bukan nilai terukur.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import streamlit as st

from core.petrophysics import (
    MATRIX_PRESETS,
    METHOD_DOCS,
    PARAM_KIND,
    SATURATION_MODELS,
    TOTAL_POROSITY_METHODS,
    VSH_METHODS,
    PetroParams,
    PetrophysicsError,
    compute_petrophysics,
    nphi_unit_hint,
    percentile_gr_limits,
)
from core.plotting import render_viewer

DEPTH_NAMES = {"DEPTH", "DEPT", "MD", "DEPTH.M", "DEPT.M"}
NONE = "(tidak ada)"

# Alias hanya untuk tebakan awal; pengguna yang memastikan. Resistivitas: pilih yang DALAM (deep).
ROLE_ALIASES = {
    "GR": ["GR", "GAMMA", "GAMMA_RAY"],
    "RHOB": ["RHOB", "RHOZ", "DEN", "DENSITY"],
    "NPHI": ["NPHI", "TNPH", "NPOR", "NEUTRON"],
    "DT": ["DTC", "DT", "AC", "SONIC"],
    "RT": ["RDEP", "ILD", "LLD", "RT", "RES", "RESISTIVITY"],
}

VIEW_TRACKS = [["GR", "VSH"], ["PHIT", "PHIE"], ["SW", "BVW"], ["PERM_EST"]]
VIEW_SCALES = {
    "VSH": (0.0, 1.0, False),
    "PHIT": (0.0, 0.5, False),
    "PHIE": (0.0, 0.5, False),
    "SW": (0.0, 1.0, False),
    "BVW": (0.0, 0.3, False),
    "PERM_EST": (0.01, 10000.0, True),
}


def _auto(role: str, columns) -> str | None:
    upper = {str(c).upper(): c for c in columns}
    for alias in ROLE_ALIASES[role]:
        if alias in upper:
            return upper[alias]
    return None


def _show_docs() -> None:
    st.subheader("Rumus, asumsi, dan batasan")
    st.caption(
        "Sitasi ditulis dari pengetahuan umum dan belum diverifikasi satu per satu ke sumber aslinya. "
        "Periksa ulang sebelum dipakai di laporan resmi."
    )
    for doc in METHOD_DOCS:
        with st.expander(doc["name"]):
            for eq in doc["equations"]:
                st.latex(eq)
            st.markdown(f"**Variabel dan satuan:** {doc['variables']}")
            st.markdown(f"**Asumsi:** {doc['assumptions']}")
            st.markdown(f"**Batasan:** {doc['limitations']}")
            st.markdown(f"**Referensi:** {doc['reference']}")


def show_petrophysics(df: pd.DataFrame | None = None) -> None:
    st.title("Petrophysical Analysis")
    st.write("Hitung Vsh, porositas, saturasi air, dan permeabilitas estimasi dengan parameter yang Anda tentukan.")
    st.info(
        "Semua hasil di halaman ini adalah **estimasi berbasis log**, bukan nilai terukur. "
        "Parameter yang tidak diisi (mis. Rw) **tidak ditebak**; perhitungan yang bergantung padanya ditandai "
        "'cannot calculate'.",
        icon="ℹ️",
    )

    if df is None:
        df = st.session_state.get("well_data")
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.warning("Belum ada data sumur. Muat data di Data Management terlebih dahulu.")
        _show_docs()
        return

    # --- Salinan kerja (data asli tidak diubah) ---
    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]
    depth_col = next((c for c in work.columns if c.upper() in DEPTH_NAMES), None)
    if depth_col is None:
        st.error("Kolom kedalaman (DEPTH, DEPT, atau MD) tidak ditemukan.")
        return
    work[depth_col] = pd.to_numeric(work[depth_col], errors="coerce")
    n_before = len(work)
    work = work.dropna(subset=[depth_col])
    if len(work) != n_before:
        st.warning(f"{n_before - len(work)} baris tanpa kedalaman valid dikeluarkan dari analisis (data asli tidak diubah).")
    if not work[depth_col].is_monotonic_increasing:
        work = work.sort_values(depth_col, kind="stable")
        st.warning("Kedalaman tidak berurutan naik. Salinan data diurutkan untuk analisis (data asli tidak diubah).")
    work = work.reset_index(drop=True)
    if work.empty:
        st.error("Tidak ada baris dengan kedalaman valid.")
        return
    if work[depth_col].duplicated().any():
        st.warning("Ada kedalaman duplikat. Data tidak dihapus; hasil pada kedalaman itu perlu diperiksa.")

    sig = hashlib.md5(repr((len(work), depth_col, float(work[depth_col].min()),
                            float(work[depth_col].max()), tuple(work.columns))).encode()).hexdigest()[:8]

    # --- 1. Pemetaan kurva ---
    options = [NONE] + [c for c in work.columns if c != depth_col]
    mapping: dict[str, str] = {}
    with st.expander("1. Pemetaan kurva", expanded=True):
        st.caption("Pilihan awal ditebak dari nama. **Periksa**, terutama resistivitas (pilih yang dalam/deep). "
                   "Satuan diasumsikan: GR gAPI, RHOB g/cc, DT µs/ft, RT ohm.m.")
        cols = st.columns(5)
        for i, role in enumerate(ROLE_ALIASES):
            default = _auto(role, work.columns)
            idx = options.index(default) if default in options else 0
            choice = cols[i].selectbox(role, options, index=idx, key=f"pm_{sig}_{role}")
            if choice != NONE:
                mapping[role] = choice
    if not mapping:
        st.error("Belum ada kurva yang dipetakan.")
        return

    # --- 2. Volume shale ---
    with st.expander("2. Volume shale (Vsh)", expanded=True):
        gmin_d = gmax_d = None
        if "GR" in mapping:
            try:
                gmin_d, gmax_d, _ = percentile_gr_limits(work[mapping["GR"]], 5.0, 95.0)
            except PetrophysicsError as exc:
                st.warning(f"Nilai awal GR dari persentil tidak dapat dihitung: {exc}")
            st.caption("Nilai awal = persentil 5 dan 95 dari data (estimasi statistik, **bukan** baseline geologi pasti). "
                       "Ubah sesuai konteks geologi.")
        else:
            st.info("Kurva GR belum dipetakan: IGR dan Vsh tidak dapat dihitung.")
        c1, c2 = st.columns(2)
        gr_min = c1.number_input("GR_min (sand/clean, gAPI)", value=None if gmin_d is None else round(gmin_d, 1),
                                 format="%.1f", key=f"gmin_{sig}_{mapping.get('GR')}")
        gr_max = c2.number_input("GR_max (shale, gAPI)", value=None if gmax_d is None else round(gmax_d, 1),
                                 format="%.1f", key=f"gmax_{sig}_{mapping.get('GR')}")
        label_to_key = {v: k for k, v in VSH_METHODS.items()}
        vsh_label = st.radio("Metode Vsh", list(VSH_METHODS.values()), index=0, horizontal=True)
        st.caption("Pilih Larionov hanya jika umur batuan diketahui (Tersier atau lebih tua). "
                   "Linear adalah pilihan paling konservatif dan tidak mengasumsikan umur.")
        c3, c4 = st.columns(2)
        clip_igr = c3.checkbox("Clip IGR ke 0–1", value=True, help="Nilai sebelum clipping tetap disimpan di kolom IGR_RAW.")
        clip_vsh = c4.checkbox("Clip Vsh ke 0–1", value=True)

    # --- 3. Porositas ---
    with st.expander("3. Porositas", expanded=True):
        preset = st.selectbox("Preset matriks (nilai awal, bisa diedit)", list(MATRIX_PRESETS))
        c1, c2, c3, c4 = st.columns(4)
        rho_ma = c1.number_input("Densitas matriks (g/cc)", value=MATRIX_PRESETS[preset]["rho_matrix"],
                                 format="%.3f", key=f"rhoma_{preset}")
        rho_f = c2.number_input("Densitas fluida (g/cc)", value=1.0, format="%.3f", key="rhof")
        dt_ma = c3.number_input("DT matriks (µs/ft)", value=MATRIX_PRESETS[preset]["dt_matrix"],
                                format="%.1f", key=f"dtma_{preset}")
        dt_f = c4.number_input("DT fluida (µs/ft)", value=189.0, format="%.1f", key="dtf")
        st.caption("Preset hanya nilai awal dari buku teks, bukan nilai pasti untuk formasi Anda.")

        nphi_unit = None
        if "NPHI" in mapping:
            hint = nphi_unit_hint(work[mapping["NPHI"]])
            hint_text = {"fraction": "tampak seperti fraction (v/v)", "percent": "tampak seperti persen (%)",
                         "ambiguous": "satuan tidak jelas dari data"}[hint]
            choice = st.selectbox("Satuan NPHI (wajib dipilih)", ["(pilih satuan)", "fraction", "percent"],
                                  key=f"nphiu_{sig}")
            st.caption(f"Petunjuk dari data: {hint_text}. Aplikasi tidak mengonversi otomatis; konfirmasi sendiri.")
            nphi_unit = None if choice == "(pilih satuan)" else choice

        m_label_to_key = {v: k for k, v in TOTAL_POROSITY_METHODS.items()}
        phit_label = st.radio("Porositas total (PHIT) memakai", list(TOTAL_POROSITY_METHODS.values()), index=0)
        shale_corr = st.checkbox("Koreksi shale: PHIE = PHIT × (1 − Vsh)", value=True,
                                 help="Jika dimatikan, PHIE disamakan dengan PHIT dan diberi peringatan.")

    # --- 4. Saturasi air ---
    with st.expander("4. Saturasi air", expanded=True):
        model_label = {k: v["label"] for k, v in SATURATION_MODELS.items()}
        sat_model = st.selectbox("Model saturasi", list(model_label), format_func=lambda k: (
            model_label[k] + ("" if SATURATION_MODELS[k]["available"] else " (belum tersedia)")))
        if not SATURATION_MODELS[sat_model]["available"]:
            st.info(f"Model ini belum tersedia: {SATURATION_MODELS[sat_model]['reason']}", icon="🚧")
        c1, c2, c3, c4 = st.columns(4)
        rw = c1.number_input("Rw (ohm.m) — wajib diisi", value=None, min_value=0.0, format="%.4f", key=f"rw_{sig}",
                             help="Resistivitas air formasi pada suhu formasi. Aplikasi tidak menebak Rw.")
        a = c2.number_input("a (tortuosity)", value=1.0, min_value=0.0, format="%.3f", key="arch_a")
        m = c3.number_input("m (cementation)", value=2.0, min_value=0.0, format="%.3f", key="arch_m")
        n = c4.number_input("n (saturation exp.)", value=2.0, min_value=0.0, format="%.3f", key="arch_n")
        st.caption("a, m, n di atas adalah contoh nilai awal yang dapat diubah. Idealnya dikalibrasi dengan data core.")

    # --- 5. Permeabilitas ---
    with st.expander("5. Permeabilitas estimasi (opsional)"):
        perm_label = st.radio("Metode", ["Tidak dihitung", "Timur", "Coates"], horizontal=True)
        perm_method = {"Tidak dihitung": "none", "Timur": "timur", "Coates": "coates"}[perm_label]
        swirr = st.number_input("Swirr (fraction, saturasi air irreducible)", value=None, min_value=0.0, max_value=1.0,
                                format="%.3f", key=f"swirr_{sig}")
        swirr_from_sw = st.checkbox("Pakai Sw hasil hitungan sebagai proksi Swirr", value=False)
        if swirr_from_sw:
            st.warning("Hanya masuk akal pada zona yang benar-benar pada saturasi air irreducible.")
        st.caption("Hasilnya adalah **estimated permeability**, bukan pengukuran. Tanpa core permeability, "
                   "hasil belum divalidasi untuk sumur ini.")

    if st.button("Hitung petrofisika", type="primary"):
        params = PetroParams(
            gr_min=gr_min, gr_max=gr_max, clip_igr=clip_igr, vsh_method=label_to_key[vsh_label], clip_vsh=clip_vsh,
            rho_matrix=rho_ma, rho_fluid=rho_f, dt_matrix=dt_ma, dt_fluid=dt_f, nphi_unit=nphi_unit,
            phit_method=m_label_to_key[phit_label], shale_correction=shale_corr, sat_model=sat_model,
            rw=rw, a=a, m=m, n=n, perm_method=perm_method, swirr=swirr, swirr_from_sw=swirr_from_sw,
        )
        try:
            out = compute_petrophysics(work, depth_col, mapping, params)
        except PetrophysicsError as exc:
            st.error(f"Perhitungan tidak dapat dijalankan: {exc}")
        except Exception as exc:  # kesalahan tak terduga ditampilkan, tidak disembunyikan
            st.error(f"Terjadi kesalahan tak terduga: {type(exc).__name__}: {exc}")
            st.exception(exc)
        else:
            st.session_state["petro_result"] = {"sig": sig, "out": out, "depth_col": depth_col}

    result = st.session_state.get("petro_result")
    if result and result["sig"] == sig:
        out = result["out"]
        t = out.table
        st.divider()
        st.subheader("Hasil")
        st.caption("Hasil memakai parameter saat tombol 'Hitung' terakhir ditekan. Ubah parameter lalu hitung ulang.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Jumlah sampel", f"{len(t):,}")
        m2.metric("PHIE valid", f"{int(t['PHIE'].notna().sum()):,}" if "PHIE" in t else "—")
        m3.metric("Sw valid", f"{int(t['SW'].notna().sum()):,}" if "SW" in t else "—")

        st.markdown("**Status per parameter**")
        rows = [{"Parameter": k, "Jenis": PARAM_KIND.get(k, ""),
                 "Status": "✅ dihitung" if v == "calculated" else f"⚪ {v}"} for k, v in out.status.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        if out.warnings:
            with st.expander(f"Peringatan ({len(out.warnings)})", expanded=True):
                for w in out.warnings:
                    st.warning(w)

        view = pd.DataFrame({result["depth_col"]: t[result["depth_col"]]})
        if "GR_log" in t:
            view["GR"] = t["GR_log"]
        for c in ("VSH", "PHIT", "PHIE", "SW", "BVW", "PERM_EST"):
            if c in t:
                view[c] = t[c]
        st.markdown("**Tampilan multi-track**")
        render_viewer(view, result["depth_col"], prefix="pv", default_tracks=VIEW_TRACKS, scale_overrides=VIEW_SCALES)
        st.caption("Sw di atas 1 terpotong di grafik (skala 0–1) tetapi tetap tersimpan di tabel dan ditandai SW>1.")

        st.markdown("**Tabel per kedalaman**")
        st.dataframe(t, hide_index=True, width="stretch")
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Unduh tabel (CSV)", t.to_csv(index=False).encode("utf-8"),
                           file_name="petrophysics_estimates.csv", mime="text/csv")
        d2.download_button(
            "⬇️ Unduh parameter (JSON)",
            json.dumps({"params": out.params, "curve_map": out.curve_map, "status": out.status},
                       indent=2, ensure_ascii=False, default=str).encode("utf-8"),
            file_name="petrophysics_parameters.json", mime="application/json")
        st.caption("Flag di kolom QC_FLAGS: INPUT_NA (ada input kosong), IGR_OUT, *_OUT (porositas di luar 0–1), "
                   "PHIE<=0, RT<=0, SW>1.")

    st.divider()
    _show_docs()
"""core/plotting.py

Mesin tampilan multi-track bersama untuk halaman Well Log dan Lithology.
Satu track dapat memuat beberapa kurva, masing-masing dengan skala sendiri
(dinormalisasi ke lebar track). Hover menampilkan nilai asli.

Data asli tidak diubah: pengurutan dan konversi numerik hanya pada salinan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DEPTH_NAMES = {"DEPTH", "DEPT", "MD", "DEPTH.M", "DEPT.M"}

ALIASES = {
    "RHOZ": "RHOB", "DEN": "RHOB", "DENSITY": "RHOB",
    "TNPH": "NPHI", "NPOR": "NPHI", "NEUTRON": "NPHI",
    "ILD": "RT", "LLD": "RT", "RES": "RT", "RESISTIVITY": "RT",
    "RDEP": "RT", "RMED": "RT", "RSHA": "RT",
    "DTC": "DT", "AC": "DT", "SONIC": "DT",
    "CAL": "CALI", "GAMMA": "GR", "GAMMA_RAY": "GR",
}

# Skala awal (min kiri, max kanan, log). Hanya nilai awal tampilan, dapat diubah pengguna.
PRESETS = {
    "GR": (0.0, 150.0, False),
    "RHOB": (1.95, 2.95, False),
    "NPHI": (0.45, -0.15, False),
    "RT": (0.2, 2000.0, True),
    "DT": (140.0, 40.0, False),
    "CALI": (6.0, 16.0, False),
    "PEF": (0.0, 10.0, False),
    "DRHO": (-0.25, 0.25, False),
}

UNITS = {
    "GR": "API", "RHOB": "g/cm³", "NPHI": "v/v", "RT": "ohm.m",
    "DT": "us/ft", "CALI": "in", "PEF": "b/e", "DRHO": "g/cm³", "SP": "mV",
}

# Palet Okabe-Ito (ramah buta warna)
COLORS = ["#009E73", "#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9"]

BG_PAPER = "#FFFFFF"
BG_PLOT = "#FBFCFD"
GRID = "#E3E8EC"
AXIS = "#B8C2CC"
TEXT = "#44525F"
TITLE_BAR = "#2F3E4E"


# ------------------------------------------------------------------ utilitas
def canon(name: str) -> str:
    upper = str(name).upper()
    return ALIASES.get(upper, upper)


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def default_scale(canon_name: str, series: pd.Series) -> tuple[float, float, bool]:
    s = series.dropna()
    if canon_name == "NPHI" and not s.empty and s.quantile(0.99) > 1.5:
        return 45.0, -15.0, False  # NPHI tampaknya dalam persen
    if canon_name in PRESETS:
        return PRESETS[canon_name]
    if s.empty:
        return 0.0, 1.0, False
    lo, hi = float(s.quantile(0.01)), float(s.quantile(0.99))
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    return lo, hi, False


def normalize(values: pd.Series, lo: float, hi: float, log: bool) -> np.ndarray:
    v = values.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if log:
            v = np.where(v > 0, v, np.nan)
            return (np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        return (v - lo) / (hi - lo)


def _fmt(x: float) -> str:
    return f"{x:g}"


def _text(fig, col, x, y, text, color, size=11, position="middle center", bold=False):
    """Teks di area header (baris 1)."""
    fig.add_trace(go.Scatter(
        x=[x], y=[y], mode="text", text=[f"<b>{text}</b>" if bold else text],
        textposition=position, textfont=dict(color=color, size=size),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=col)


# ----------------------------------------------------------------- callbacks
def _reset(prefix: str) -> None:
    for key in list(st.session_state.keys()):
        if key.startswith(prefix + "_"):
            del st.session_state[key]


def _add_track(prefix: str) -> None:
    st.session_state[f"{prefix}_n_tracks"] += 1


def _remove_track(prefix: str) -> None:
    n = st.session_state[f"{prefix}_n_tracks"]
    if n > 1:
        st.session_state.pop(f"{prefix}_trk_{n - 1}", None)
        st.session_state[f"{prefix}_n_tracks"] = n - 1


def _fit_all(prefix: str) -> None:
    ss = st.session_state
    ss[f"{prefix}_top_in"] = ss[f"{prefix}_dmin"]
    ss[f"{prefix}_bot_in"] = ss[f"{prefix}_dmax"]
    ss[f"{prefix}_top"] = ss[f"{prefix}_dmin"]
    ss[f"{prefix}_bot"] = ss[f"{prefix}_dmax"]


# --------------------------------------------------------------- viewer utama
def render_viewer(
    df: pd.DataFrame,
    depth_col: str,
    prefix: str,
    litho: dict | None = None,
    signature_extra: str = "",
    default_tracks: list[list[str]] | None = None,
    scale_overrides: dict[str, tuple[float, float, bool]] | None = None,
) -> tuple[float, float] | None:
    """Tampilkan kontrol dan grafik multi-track.

    litho (opsional): {"col": nama kolom label, "classes": [nama kelas],
                       "colors": {kelas: warna}, "title": judul kolom}
    Mengembalikan (top, bottom) kedalaman terpilih, atau None jika data tidak dapat ditampilkan.
    """
    if df is None or df.empty:
        st.warning("Belum ada data sumur. Unggah data terlebih dahulu.")
        return None

    work = df.copy()  # data asli tidak diubah
    work.columns = [str(c).strip() for c in work.columns]
    if depth_col not in work.columns:
        st.error(f"Kolom kedalaman {depth_col!r} tidak ditemukan.")
        return None

    work[depth_col] = pd.to_numeric(work[depth_col], errors="coerce")
    work = work.dropna(subset=[depth_col]).sort_values(depth_col)

    curve_cols = [
        c for c in work.select_dtypes(include="number").columns
        if c != depth_col and work[c].notna().any()
    ]
    if work.empty or not curve_cols:
        st.warning("Tidak ditemukan kurva numerik yang valid.")
        return None

    dmin, dmax = float(work[depth_col].min()), float(work[depth_col].max())
    if dmin >= dmax:
        st.warning("Rentang kedalaman hanya satu titik, grafik tidak dapat dibuat.")
        return None

    ss = st.session_state
    signature = (depth_col, tuple(curve_cols), dmin, dmax, len(work), signature_extra)
    if ss.get(f"{prefix}_sig") != signature:
        _reset(prefix)
        ss[f"{prefix}_sig"] = signature

    ss.setdefault(f"{prefix}_n_tracks", 4)
    ss.setdefault(f"{prefix}_dmin", dmin)
    ss.setdefault(f"{prefix}_dmax", dmax)
    ss.setdefault(f"{prefix}_top", dmin)
    ss.setdefault(f"{prefix}_bot", dmax)
    ss.setdefault(f"{prefix}_top_in", dmin)
    ss.setdefault(f"{prefix}_bot_in", dmax)
    ss.setdefault(f"{prefix}_shade", True)

    by_canon: dict[str, str] = {}
    for c in curve_cols:
        by_canon.setdefault(canon(c), c)
    defaults = default_tracks or [["GR", "RHOB", "NPHI"], ["RT"], ["DT"], []]
    for i in range(ss[f"{prefix}_n_tracks"]):
        key = f"{prefix}_trk_{i}"
        if key not in ss:
            wanted = defaults[i] if i < len(defaults) else []
            ss[key] = [by_canon[w] for w in wanted if w in by_canon]

    # --- Kontrol kedalaman ---
    left, right = st.columns([5, 1])
    with left:
        with st.form(f"{prefix}_depth_form"):
            c1, c2, c3 = st.columns([2, 2, 1])
            top_in = c1.number_input("Top", min_value=dmin, max_value=dmax, key=f"{prefix}_top_in", format="%.1f")
            bot_in = c2.number_input("Bottom", min_value=dmin, max_value=dmax, key=f"{prefix}_bot_in", format="%.1f")
            c3.write("")
            applied = c3.form_submit_button("Apply")
        if applied:
            if top_in < bot_in:
                ss[f"{prefix}_top"], ss[f"{prefix}_bot"] = float(top_in), float(bot_in)
            else:
                st.error("Top harus lebih kecil dari Bottom.")
    with right:
        right.write("")
        right.write("")
        st.button("Fit all", on_click=_fit_all, args=(prefix,), key=f"{prefix}_fit_btn")

    top, bottom = float(ss[f"{prefix}_top"]), float(ss[f"{prefix}_bot"])

    # --- Pengaturan track ---
    b1, b2, b3 = st.columns([1, 1, 2])
    b1.button("➕ Tambah track", on_click=_add_track, args=(prefix,), key=f"{prefix}_add_btn")
    b2.button("➖ Hapus track terakhir", on_click=_remove_track, args=(prefix,), key=f"{prefix}_rm_btn")
    b3.checkbox("Arsir kurva pertama tiap track", key=f"{prefix}_shade")

    n_tracks = ss[f"{prefix}_n_tracks"]
    track_cols = st.columns(min(n_tracks, 4))
    for i in range(n_tracks):
        with track_cols[i % len(track_cols)]:
            st.multiselect(f"Track {i + 1:02d}", options=curve_cols, key=f"{prefix}_trk_{i}")

    tracks = [list(ss[f"{prefix}_trk_{i}"]) for i in range(n_tracks)]
    used = list(dict.fromkeys(c for t in tracks for c in t))

    # --- Skala kurva ---
    for c in used:
        if scale_overrides and c in scale_overrides:
            lo0, hi0, log0 = scale_overrides[c]
        else:
            lo0, hi0, log0 = default_scale(canon(c), work[c])
        ss.setdefault(f"{prefix}_lo_{c}", float(lo0))
        ss.setdefault(f"{prefix}_hi_{c}", float(hi0))
        ss.setdefault(f"{prefix}_log_{c}", bool(log0))

    if used:
        with st.expander("Skala kurva (bisa diedit)"):
            st.caption(
                "Min di kiri, max di kanan. Isi min lebih besar dari max untuk membalik sumbu "
                "(mis. NPHI 0.45 ke -0.15). Skala log membutuhkan nilai > 0. "
                "Skala awal hanya nilai tampilan, bukan standar."
            )
            for c in used:
                a, b, d, e = st.columns([2, 2, 2, 1])
                a.markdown(f"**{c}** ({UNITS.get(canon(c), 'satuan?')})")
                b.number_input("Min (kiri)", key=f"{prefix}_lo_{c}", format="%.4g")
                d.number_input("Max (kanan)", key=f"{prefix}_hi_{c}", format="%.4g")
                e.checkbox("Log", key=f"{prefix}_log_{c}")

    if not used and litho is None:
        st.info("Pilih minimal satu kurva di salah satu track.")
        return top, bottom

    sub = work[(work[depth_col] >= top) & (work[depth_col] <= bottom)]
    if sub.empty:
        st.warning("Tidak ada data pada rentang kedalaman ini.")
        return top, bottom

    # --- Susun grafik ---
    n_max = max([len(t) for t in tracks] + [1])
    header_px = 50 + 52 * n_max
    body_px = 700
    total = header_px + body_px
    h_top = n_max + 1
    n_cols = n_tracks + (1 if litho else 0)

    fig = make_subplots(
        rows=2, cols=n_cols, shared_yaxes=True,
        horizontal_spacing=0.0, vertical_spacing=0.0,
        row_heights=[header_px / total, body_px / total],
        column_widths=[1] * n_tracks + ([0.6] if litho else []),
    )

    skipped: list[str] = []
    for ti, curves in enumerate(tracks):
        col = ti + 1
        fig.add_shape(type="rect", x0=0.0, x1=1.0, y0=n_max + 0.05, y1=h_top,
                      fillcolor=TITLE_BAR, line_width=0, row=1, col=col)
        _text(fig, col, 0.5, n_max + 0.5, f"TRACK {ti + 1:02d}  ·  {len(curves)} kurva", "#FFFFFF", 11, bold=True)

        for j, curve in enumerate(curves):
            lo = float(ss[f"{prefix}_lo_{curve}"])
            hi = float(ss[f"{prefix}_hi_{curve}"])
            log = bool(ss[f"{prefix}_log_{curve}"])
            if lo == hi or (log and (lo <= 0 or hi <= 0)):
                skipped.append(curve)
                continue

            color = COLORS[j % len(COLORS)]
            unit = UNITS.get(canon(curve), "")
            base = n_max - 1 - j

            fig.add_shape(type="rect", x0=0.03, x1=0.97, y0=base + 0.08, y1=base + 0.92,
                          fillcolor=rgba(color, 0.10), line=dict(color=rgba(color, 0.45), width=1),
                          row=1, col=col)
            fig.add_shape(type="rect", x0=0.03, x1=0.065, y0=base + 0.08, y1=base + 0.92,
                          fillcolor=color, line_width=0, row=1, col=col)
            _text(fig, col, 0.5, base + 0.66, f"{curve}  ({unit})" if unit else curve, TEXT, 11, bold=True)
            _text(fig, col, 0.10, base + 0.30, _fmt(lo), color, 10, "middle right")
            _text(fig, col, 0.94, base + 0.30, _fmt(hi), color, 10, "middle left")
            _text(fig, col, 0.5, base + 0.30, "◂━━━━━▸", color, 10)

            raw = pd.to_numeric(sub[curve], errors="coerce")
            shade = bool(ss.get(f"{prefix}_shade", True)) and j == 0
            fig.add_trace(go.Scatter(
                x=normalize(raw, lo, hi, log), y=sub[depth_col], mode="lines",
                line=dict(color=color, width=1.6), connectgaps=False,
                fill="tozerox" if shade else None,
                fillcolor=rgba(color, 0.16) if shade else None,
                customdata=raw.to_numpy(dtype=float), name=curve, showlegend=False,
                hovertemplate=f"{curve}: %{{customdata:.4g}} {unit}<br>{depth_col}: %{{y:.2f}}<extra></extra>",
            ), row=2, col=col)

        fig.update_xaxes(visible=False, range=[0, 1], row=1, col=col)
        fig.update_xaxes(
            range=[0, 1], dtick=0.2, showgrid=True, gridcolor=GRID, showticklabels=False,
            zeroline=False, showline=True, linecolor=AXIS, mirror=True, row=2, col=col,
        )

    # --- Kolom litologi (opsional) ---
    if litho:
        lcol = n_cols
        classes = list(litho["classes"])
        colors = litho["colors"]
        fig.add_shape(type="rect", x0=0, x1=1, y0=n_max + 0.05, y1=h_top,
                      fillcolor=TITLE_BAR, line_width=0, row=1, col=lcol)
        _text(fig, lcol, 0.5, n_max + 0.5, litho.get("title", "LITHOLOGY"), "#FFFFFF", 11, bold=True)

        names = sub[litho["col"]]
        codes = pd.Categorical(names, categories=classes).codes.astype(float)
        codes[codes < 0] = np.nan
        label = names.fillna("(tidak diprediksi)").astype(str).to_numpy()
        n = max(len(classes), 1)
        colorscale = []
        for k, name in enumerate(classes):
            colorscale += [[k / n, colors[name]], [(k + 1) / n, colors[name]]]
        if not colorscale:
            colorscale = [[0, "#BDBDBD"], [1, "#BDBDBD"]]

        fig.add_trace(go.Heatmap(
            x=[0, 1], y=sub[depth_col].to_numpy(),
            z=np.column_stack([codes, codes]), text=np.column_stack([label, label]),
            colorscale=colorscale, zmin=-0.5, zmax=n - 0.5, showscale=False,
            hovertemplate=f"%{{text}}<br>{depth_col}: %{{y:.2f}}<extra></extra>",
        ), row=2, col=lcol)
        fig.update_xaxes(visible=False, range=[0, 1], row=1, col=lcol)
        fig.update_xaxes(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False,
                         showline=True, linecolor=AXIS, mirror=True, row=2, col=lcol)

    fig.update_yaxes(visible=False, range=[0, h_top], row=1)
    fig.update_yaxes(
        range=[bottom, top], showgrid=True, gridcolor=GRID, zeroline=False,
        showline=True, linecolor=AXIS, tickfont=dict(color=TEXT, size=10), row=2,
    )
    fig.update_yaxes(title_text=depth_col, title_font=dict(color=TEXT, size=11), row=2, col=1)
    fig.update_layout(
        height=total + 30, margin=dict(l=60, r=10, t=10, b=10),
        paper_bgcolor=BG_PAPER, plot_bgcolor=BG_PLOT,
        font=dict(color=TEXT), hovermode="closest", showlegend=False,
    )

    if skipped:
        st.warning(
            "Kurva tidak digambar karena skalanya tidak valid (min = max, atau skala log dengan nilai ≤ 0): "
            + ", ".join(dict.fromkeys(skipped))
        )

    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    st.caption(
        "Tiap kurva dinormalisasi ke lebar track sesuai skalanya (lihat kartu header). "
        "Hover menampilkan nilai asli. Celah pada garis berarti nilai kosong."
    )
    return top, bottom
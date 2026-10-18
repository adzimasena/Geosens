"""core/reservoir.py

Evaluasi reservoir dari tabel hasil Petrophysics: net reservoir, net pay, ketebalan,
dan kelas kualitas. Semua cut-off berasal dari pengguna; tidak ada nilai bawaan.
Semua hasil adalah ESTIMASI berbasis log, bukan nilai terukur.

Langkah yang tidak dapat dihitung dilaporkan di `status`, dan kolomnya tidak dibuat.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


class ReservoirError(Exception):
    """Input atau parameter evaluasi reservoir tidak valid."""


CLASS_LABELS = ["Buruk", "Cukup", "Baik", "Sangat baik"]


@dataclass
class ReservoirParams:
    vsh_max: float | None = None      # net reservoir: Vsh <= vsh_max (kosong = tidak dipakai)
    phie_min: float | None = None     # net reservoir: PHIE >= phie_min (kosong = tidak dipakai)
    sw_max: float | None = None       # net pay: Sw <= sw_max, hanya pada net reservoir
    zone_top: float | None = None     # batas atas zona evaluasi (kosong = seluruh data)
    zone_base: float | None = None    # batas bawah zona evaluasi
    class_bounds: tuple[float, float, float] | None = None  # batas PHIE antar kelas, naik


@dataclass
class ReservoirOutput:
    table: pd.DataFrame
    summary: dict
    intervals: pd.DataFrame
    class_summary: pd.DataFrame
    status: dict[str, str]
    warnings: list[str]
    params: dict


# ------------------------------------------------------------------ util
def validate_params(p: ReservoirParams) -> None:
    if p.vsh_max is not None and not (0.0 <= p.vsh_max <= 1.0):
        raise ReservoirError("Cut-off Vsh harus antara 0 dan 1.")
    if p.phie_min is not None and not (0.0 <= p.phie_min < 1.0):
        raise ReservoirError("Cut-off PHIE harus antara 0 dan 1 (fraction).")
    if p.sw_max is not None and not (0.0 < p.sw_max <= 1.0):
        raise ReservoirError("Cut-off Sw harus lebih dari 0 dan paling besar 1.")
    if (p.zone_top is None) != (p.zone_base is None):
        raise ReservoirError("Isi kedua batas zona (top dan base) atau kosongkan keduanya.")
    if p.zone_top is not None and not (np.isfinite(p.zone_top) and np.isfinite(p.zone_base)
                                       and p.zone_top < p.zone_base):
        raise ReservoirError("Top zona harus lebih kecil dari base zona.")
    if p.class_bounds is not None:
        b = p.class_bounds
        if len(b) != 3 or not (0.0 < b[0] < b[1] < b[2] < 1.0):
            raise ReservoirError("Batas kelas PHIE harus tiga angka naik dan berada di antara 0 dan 1.")


def sample_thickness(depth, zone_top: float | None = None,
                     zone_base: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ketebalan tiap sampel dari kedalaman AKTUAL (tidak mengasumsikan langkah seragam).

    Batas antar sampel = titik tengah kedalaman bersebelahan. Sampel pertama dan terakhir
    diperluas setengah langkah ke luar. Jika zona diberikan, sel dipotong ke zona itu.
    Mengembalikan (tebal, top_sel, base_sel).
    """
    d = pd.to_numeric(pd.Series(depth), errors="coerce").to_numpy(dtype=float)
    if d.size < 2:
        raise ReservoirError("Dibutuhkan minimal 2 sampel untuk menghitung ketebalan.")
    if np.isnan(d).any():
        raise ReservoirError("Kedalaman mengandung nilai kosong atau non-numerik.")
    if (np.diff(d) < 0).any():
        raise ReservoirError("Kedalaman harus berurutan naik.")
    mid = (d[:-1] + d[1:]) / 2.0
    top = np.empty_like(d)
    base = np.empty_like(d)
    top[1:] = mid
    top[0] = d[0] - (d[1] - d[0]) / 2.0
    base[:-1] = mid
    base[-1] = d[-1] + (d[-1] - d[-2]) / 2.0
    if zone_top is not None and zone_base is not None:
        top = np.maximum(top, zone_top)
        base = np.minimum(base, zone_base)
    thick = np.clip(base - top, 0.0, None)
    return thick, top, base


def _wavg(values, weights) -> float | None:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not m.any():
        return None
    return float(np.sum(v[m] * w[m]) / np.sum(w[m]))


def build_intervals(table: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    """Interval menerus dari sampel dengan flag True. Rata-rata dibobot ketebalan;
    Sw dibobot volume pori (tebal × PHIE)."""
    base_cols = ["No", "Top", "Base", "Tebal"]
    if flag_col not in table.columns:
        return pd.DataFrame(columns=base_cols)
    f = table[flag_col].to_numpy(dtype=bool)
    th = table["THICK"].to_numpy(dtype=float)
    ctop = table["CELL_TOP"].to_numpy(dtype=float)
    cbase = table["CELL_BASE"].to_numpy(dtype=float)
    phie = table["PHIE"].to_numpy(dtype=float) if "PHIE" in table else None
    vsh = table["VSH"].to_numpy(dtype=float) if "VSH" in table else None
    sw = table["SW"].to_numpy(dtype=float) if "SW" in table else None
    pay = table["PAY_FLAG"].to_numpy(dtype=bool) if (flag_col == "RES_FLAG" and "PAY_FLAG" in table) else None

    cols = list(base_cols)
    if pay is not None:
        cols.append("Tebal pay")
    if phie is not None:
        cols.append("PHIE rata-rata")
    if vsh is not None:
        cols.append("Vsh rata-rata")
    if sw is not None and phie is not None:
        cols.append("Sw (PV-weighted)")

    rows = []
    n = len(f)
    i = 0
    while i < n:
        if not f[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and f[j + 1]:
            j += 1
        sl = slice(i, j + 1)
        row = {"No": len(rows) + 1, "Top": float(ctop[i]), "Base": float(cbase[j]), "Tebal": float(th[sl].sum())}
        if pay is not None:
            row["Tebal pay"] = float(th[sl][pay[sl]].sum())
        if phie is not None:
            row["PHIE rata-rata"] = _wavg(phie[sl], th[sl])
        if vsh is not None:
            row["Vsh rata-rata"] = _wavg(vsh[sl], th[sl])
        if sw is not None and phie is not None:
            row["Sw (PV-weighted)"] = _wavg(sw[sl], th[sl] * phie[sl])
        rows.append(row)
        i = j + 1
    return pd.DataFrame(rows, columns=cols)


# ------------------------------------------------------------------ pipeline
def compute_reservoir(table: pd.DataFrame, depth_col: str, params: ReservoirParams) -> ReservoirOutput:
    """Hitung net reservoir, net pay, ketebalan, dan kelas kualitas dari tabel Petrophysics."""
    if depth_col not in table.columns:
        raise ReservoirError(f"Kolom kedalaman {depth_col!r} tidak ada di tabel.")
    validate_params(params)
    p = params
    status: dict[str, str] = {}
    warnings: list[str] = []

    t = table.copy()
    t[depth_col] = pd.to_numeric(t[depth_col], errors="coerce")
    n0 = len(t)
    t = t.dropna(subset=[depth_col])
    if len(t) != n0:
        warnings.append(f"[DEPTH] {n0 - len(t)} baris tanpa kedalaman valid dikeluarkan.")
    if len(t) < 2:
        raise ReservoirError("Dibutuhkan minimal 2 sampel dengan kedalaman valid.")
    if not t[depth_col].is_monotonic_increasing:
        t = t.sort_values(depth_col, kind="stable")
        warnings.append("[DEPTH] Kedalaman tidak berurutan naik; data diurutkan untuk perhitungan ketebalan.")
    t = t.reset_index(drop=True)
    d = t[depth_col].to_numpy(dtype=float)

    ndup = int(pd.Series(d).duplicated().sum())
    if ndup:
        warnings.append(f"[DEPTH] Ada {ndup} kedalaman duplikat; ketebalan sampel duplikat menjadi sangat kecil atau nol.")
    steps = np.diff(d)
    pos = steps[steps > 0]
    if pos.size:
        ngap = int((steps > 3.0 * np.median(pos)).sum())
        if ngap:
            warnings.append(f"[DEPTH] Ada {ngap} celah kedalaman lebih dari 3× langkah median. Tiap sampel di tepi celah "
                            "mendapat setengah celah sebagai ketebalan, dan interval net yang melintasi celah terhitung menerus. "
                            "Periksa apakah ada data hilang.")

    th, ctop, cbase = sample_thickness(d, p.zone_top, p.zone_base)
    gross = float(th.sum())
    if gross <= 0:
        raise ReservoirError("Zona evaluasi tidak beririsan dengan rentang kedalaman data.")
    in_zone = th > 0

    out = pd.DataFrame({depth_col: d, "CELL_TOP": ctop, "CELL_BASE": cbase, "THICK": th})
    for c in ("VSH", "PHIE", "SW"):
        if c in t.columns:
            out[c] = pd.to_numeric(t[c], errors="coerce")

    summary: dict = {"gross": gross, "net_reservoir": None, "net_pay": None, "unknown": None,
                     "ntg": None, "pay_to_gross": None, "phie_avg": None, "vsh_avg": None,
                     "sw_avg_pv": None, "n_intervals": None}

    # ---- Net reservoir
    criteria = []
    if p.vsh_max is not None:
        criteria.append(("VSH", "<=", p.vsh_max))
    if p.phie_min is not None:
        criteria.append(("PHIE", ">=", p.phie_min))

    res: np.ndarray | None = None
    if not criteria:
        status["RESERVOIR"] = "cannot calculate: isi minimal satu cut-off (Vsh maks atau PHIE min)."
    else:
        missing = [c for c, _, _ in criteria if c not in out.columns]
        if missing:
            status["RESERVOIR"] = (f"cannot calculate: kolom {', '.join(missing)} tidak ada di hasil Petrophysics "
                                   "(cut-off yang diisi tidak boleh diabaikan diam-diam).")
        else:
            ok = np.ones(len(out), dtype=bool)
            known = np.ones(len(out), dtype=bool)
            for c, op, thr in criteria:
                v = out[c].to_numpy(dtype=float)
                known &= np.isfinite(v)
                with np.errstate(invalid="ignore"):
                    ok &= (v <= thr) if op == "<=" else (v >= thr)
            res = ok & known & in_zone
            unknown = float(th[in_zone & ~known].sum())
            out["EVAL_OK"] = known
            out["RES_FLAG"] = res
            status["RESERVOIR"] = "calculated"
            summary["net_reservoir"] = float(th[res].sum())
            summary["unknown"] = unknown
            summary["ntg"] = summary["net_reservoir"] / gross
            if unknown > 0:
                warnings.append(f"[RESERVOIR] {unknown:.2f} (satuan kedalaman) dari zona tidak dapat dievaluasi karena "
                                "input cut-off kosong atau bukan angka. Bagian ini TIDAK dihitung sebagai net.")
            phie_a = out["PHIE"].to_numpy(dtype=float) if "PHIE" in out else None
            if phie_a is not None:
                summary["phie_avg"] = _wavg(phie_a[res], th[res])
            if "VSH" in out:
                summary["vsh_avg"] = _wavg(out["VSH"].to_numpy(dtype=float)[res], th[res])
            if phie_a is not None and "SW" in out:
                summary["sw_avg_pv"] = _wavg(out["SW"].to_numpy(dtype=float)[res], (th * phie_a)[res])

    # ---- Net pay
    pay: np.ndarray | None = None
    if p.sw_max is None:
        status["PAY"] = "tidak diminta"
    elif res is None:
        status["PAY"] = "cannot calculate: net reservoir tidak tersedia."
    elif "SW" not in out.columns:
        status["PAY"] = "cannot calculate: kolom SW tidak ada (apakah Rw sudah diisi di Petrophysics?)."
    else:
        sw = out["SW"].to_numpy(dtype=float)
        pay_known = np.isfinite(sw)
        with np.errstate(invalid="ignore"):
            pay = res & pay_known & (sw <= p.sw_max)
        out["PAY_FLAG"] = pay
        status["PAY"] = "calculated"
        summary["net_pay"] = float(th[pay].sum())
        summary["pay_to_gross"] = summary["net_pay"] / gross
        unk_pay = float(th[res & ~pay_known].sum())
        if unk_pay > 0:
            warnings.append(f"[PAY] {unk_pay:.2f} (satuan kedalaman) net reservoir tidak punya Sw valid, jadi tidak dihitung sebagai pay.")

    # ---- Kelas kualitas (relatif, berdasarkan PHIE saja)
    class_df = pd.DataFrame(columns=["Kelas", "Rentang PHIE", "Gross", "Net reservoir"])
    if p.class_bounds is None:
        status["PHIE_CLASS"] = "tidak diminta"
    elif "PHIE" not in out.columns:
        status["PHIE_CLASS"] = "cannot calculate: kolom PHIE tidak ada di hasil Petrophysics."
    else:
        b = p.class_bounds
        cls = pd.cut(out["PHIE"], bins=[-np.inf, b[0], b[1], b[2], np.inf], labels=CLASS_LABELS, right=False)
        out["PHIE_CLASS"] = cls.astype(object)
        ranges = [f"< {b[0]:g}", f"{b[0]:g} sampai < {b[1]:g}", f"{b[1]:g} sampai < {b[2]:g}", f">= {b[2]:g}"]
        rows = []
        for label, rng in zip(CLASS_LABELS, ranges):
            m = ((out["PHIE_CLASS"] == label).to_numpy(dtype=bool)) & in_zone
            rows.append({"Kelas": label, "Rentang PHIE": rng, "Gross": float(th[m].sum()),
                         "Net reservoir": float(th[m & res].sum()) if res is not None else None})
        class_df = pd.DataFrame(rows)
        status["PHIE_CLASS"] = "calculated"
        warnings.append("[PHIE_CLASS] Kelas ini relatif dan hanya berdasarkan PHIE dengan batas dari pengguna; "
                        "bukan klasifikasi baku dan tidak memakai permeabilitas.")

    intervals = build_intervals(out, "RES_FLAG") if res is not None else pd.DataFrame()
    if res is not None:
        summary["n_intervals"] = int(len(intervals))

    warnings.append("[THICKNESS] Ketebalan dihitung sepanjang lubang bor dari kolom kedalaman (MD jika itu isinya). "
                    "Pada sumur miring ini bukan ketebalan vertikal sejati (TVT).")

    return ReservoirOutput(table=out, summary=summary, intervals=intervals, class_summary=class_df,
                           status=status, warnings=warnings, params=asdict(p))


# ------------------------------------------------------------------ dokumentasi metode
RESERVOIR_DOCS = [
    {
        "name": "Ketebalan dari kedalaman aktual",
        "equations": [r"h_i = \dfrac{d_{i+1} - d_{i-1}}{2}"],
        "variables": "d_i: kedalaman sampel ke-i (m atau ft, sesuai data). h_i: ketebalan sampel. "
                     "Sampel pertama dan terakhir diperluas setengah langkah ke luar.",
        "assumptions": "Setiap sampel mewakili sel dari titik tengah ke titik tengah sampel bersebelahan, "
                       "sehingga interval tidak seragam ditangani tanpa mengasumsikan langkah tetap.",
        "limitations": "Ketebalan sepanjang lubang bor, bukan ketebalan vertikal sejati pada sumur miring. "
                       "Celah data besar membuat sampel di tepinya mendapat ketebalan besar.",
        "reference": "Definisi geometris; tanpa sitasi khusus.",
    },
    {
        "name": "Net reservoir dan Net/Gross",
        "equations": [r"\text{Net reservoir} = \sum_{i \in \text{lolos cut-off}} h_i",
                      r"N/G = \dfrac{\text{Net reservoir}}{\text{Gross}}"],
        "variables": "Gross: total ketebalan zona. Sampel lolos jika semua cut-off yang diisi terpenuhi "
                     "(Vsh <= Vsh maks, PHIE >= PHIE min).",
        "assumptions": "Cut-off ditentukan pengguna dan mewakili reservoir yang berarti untuk lapangan ini.",
        "limitations": "Tidak ada cut-off universal; nilai biasanya bergantung formasi dan idealnya dikalibrasi "
                       "dengan core atau uji produksi. Sampel dengan input kosong tidak dihitung dan dilaporkan "
                       "sebagai tebal tidak dapat dievaluasi.",
        "reference": "Definisi umum; tanpa sitasi khusus.",
    },
    {
        "name": "Net pay",
        "equations": [r"\text{Net pay} = \sum_{i \in \text{net reservoir},\, S_{w,i} \le S_{w,max}} h_i"],
        "variables": "Sw maks: cut-off saturasi air dari pengguna.",
        "assumptions": "Net pay adalah bagian net reservoir dengan Sw di bawah atau sama dengan cut-off.",
        "limitations": "Bergantung pada Rw, a, m, n dan model Sw di Petrophysics. Net pay dari log belum "
                       "menyatakan zona produktif; itu butuh data uji fluida dan produksi.",
        "reference": "Definisi umum; tanpa sitasi khusus.",
    },
    {
        "name": "Rata-rata berbobot",
        "equations": [r"\bar{\phi}_E = \dfrac{\sum h_i \phi_{E,i}}{\sum h_i}",
                      r"\bar{S}_w = \dfrac{\sum h_i \phi_{E,i} S_{w,i}}{\sum h_i \phi_{E,i}}"],
        "variables": "Penjumlahan hanya pada net reservoir. PHIE dan Vsh dibobot ketebalan; Sw dibobot volume pori.",
        "assumptions": "Bobot ketebalan dari kedalaman aktual.",
        "limitations": "Nilai rata-rata menyembunyikan variasi vertikal; lihat tabel interval.",
        "reference": "Praktik umum evaluasi formasi; tanpa sitasi khusus.",
    },
    {
        "name": "Kelas kualitas reservoir (relatif)",
        "equations": [r"\text{kelas} = f(\phi_E;\ b_1 < b_2 < b_3)"],
        "variables": "b1, b2, b3: batas PHIE (fraction) dari pengguna; empat kelas: Buruk, Cukup, Baik, Sangat baik.",
        "assumptions": "Kualitas diwakili oleh PHIE saja.",
        "limitations": "Bukan klasifikasi baku. Tidak memakai permeabilitas, litologi, atau ukuran pori. "
                       "Nilai batas bergantung lapangan.",
        "reference": "Skema kelas ditentukan pengguna; tanpa sitasi khusus.",
    },
]
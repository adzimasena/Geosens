"""core/petrophysics.py

Perhitungan petrofisika sebagai fungsi murni (tanpa Streamlit) agar dapat diuji.

Prinsip:
- Input divalidasi; parameter yang hilang TIDAK ditebak.
- Nilai kosong (NaN) tetap NaN; nilai tidak valid secara domain menjadi NaN dengan peringatan.
- Nilai di luar rentang fisik DIPERTAHANKAN dan diberi peringatan (kecuali pengguna memilih clipping).
- Semua hasil adalah estimasi berbasis log, bukan nilai terukur.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


class PetrophysicsError(ValueError):
    """Input tidak valid atau tidak cukup untuk perhitungan."""


@dataclass
class CalcResult:
    """Hasil satu perhitungan: nilai per sampel dan daftar peringatan."""

    values: pd.Series
    warnings: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ konstanta
VSH_METHODS = {
    "linear": "Linear (Vsh = IGR)",
    "larionov_tertiary": "Larionov — batuan Tersier",
    "larionov_older": "Larionov — batuan lebih tua",
}

TOTAL_POROSITY_METHODS = {
    "density": "Porositas densitas (PHID)",
    "neutron": "Porositas neutron (PHIN)",
    "sonic": "Porositas sonic Wyllie (PHIS)",
    "density_neutron_mean": "Densitas-neutron: rata-rata aritmetika",
    "density_neutron_rms": "Densitas-neutron: rata-rata kuadratik (RMS)",
}

# Nilai awal yang dapat diedit. BUKAN nilai pasti untuk semua formasi.
MATRIX_PRESETS = {
    "Sandstone": {"rho_matrix": 2.65, "dt_matrix": 55.5},
    "Limestone": {"rho_matrix": 2.71, "dt_matrix": 47.5},
    "Dolomite": {"rho_matrix": 2.87, "dt_matrix": 43.5},
}

SATURATION_MODELS = {
    "archie": {"label": "Archie", "available": True, "reason": ""},
    "simandoux": {
        "label": "Simandoux (shaly sand)",
        "available": False,
        "reason": "bentuk persamaan Simandoux berbeda antar sumber dan belum diverifikasi; "
                  "tidak ditampilkan angka daripada memakai bentuk yang belum pasti.",
    },
    "indonesian": {
        "label": "Indonesian (shaly sand)",
        "available": False,
        "reason": "konvensi parameter (Rsh, eksponen) belum diverifikasi dengan referensi; "
                  "tidak ditampilkan angka daripada memakai bentuk yang belum pasti.",
    },
}

PARAM_KIND = {
    "IGR": "Calculated parameter",
    "VSH": "Estimated parameter (dari GR)",
    "PHID": "Calculated parameter",
    "PHIN": "Calculated parameter",
    "PHIS": "Estimated parameter (model Wyllie)",
    "PHIT": "Calculated parameter",
    "PHIE": "Calculated parameter (koreksi shale sederhana)",
    "SW": "Estimated parameter (model saturasi)",
    "BVW": "Calculated parameter",
    "PERM_EST": "Estimated permeability (empiris, belum dikalibrasi)",
}


# ------------------------------------------------------------------ utilitas
def _scalar(value, name: str, positive: bool = False) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise PetrophysicsError(f"{name} harus berupa angka (nilai: {value!r}).") from exc
    if not np.isfinite(v):
        raise PetrophysicsError(f"{name} harus berhingga (bukan NaN atau tak hingga).")
    if positive and v <= 0:
        raise PetrophysicsError(f"{name} harus > 0 (nilai: {v:g}).")
    return v


def _prep(x, name: str) -> tuple[pd.Series, list[str]]:
    """Ubah input ke Series float. Non-numerik dan inf menjadi NaN, dengan peringatan."""
    raw = x if isinstance(x, pd.Series) else pd.Series(x)
    if len(raw) == 0:
        raise PetrophysicsError(f"{name}: data kosong.")
    num = pd.to_numeric(raw, errors="coerce").astype(float)
    warns: list[str] = []
    n_bad = int((raw.notna() & num.isna()).sum())
    if n_bad:
        warns.append(f"{n_bad} nilai {name} non-numerik dianggap kosong (NaN).")
    inf = np.isinf(num)
    if inf.any():
        warns.append(f"{int(inf.sum())} nilai {name} tak hingga dianggap kosong (NaN).")
        num = num.mask(inf)
    if num.notna().sum() == 0:
        raise PetrophysicsError(f"{name}: tidak ada nilai valid.")
    return num, warns


def _align(*series: pd.Series) -> tuple[list[np.ndarray], pd.Index]:
    """Samakan panjang (panjang 1 di-broadcast). Panjang berbeda lainnya -> error."""
    n = max(len(s) for s in series)
    ref = next(s for s in series if len(s) == n)
    arrays: list[np.ndarray] = []
    for s in series:
        if len(s) == n:
            arrays.append(s.to_numpy(dtype=float))
        elif len(s) == 1:
            arrays.append(np.full(n, float(s.iloc[0])))
        else:
            raise PetrophysicsError("Panjang data input tidak sama; tidak dapat dihitung bersama.")
    return arrays, ref.index


def _range_warning(s: pd.Series, lo: float, hi: float, name: str) -> list[str]:
    n = int(((s < lo) | (s > hi)).sum())
    if not n:
        return []
    return [f"{n} nilai {name} di luar rentang {lo:g}–{hi:g}; nilai dipertahankan (tidak dihapus). "
            "Periksa data dan parameter."]


# ------------------------------------------------------------------ IGR dan Vsh
def percentile_gr_limits(gr, low: float = 5.0, high: float = 95.0) -> tuple[float, float, list[str]]:
    """GR_min dan GR_max dari persentil data (estimasi statistik, bukan baseline geologi)."""
    lo = _scalar(low, "persentil bawah")
    hi = _scalar(high, "persentil atas")
    if not (0 <= lo < hi <= 100):
        raise PetrophysicsError("Persentil harus memenuhi 0 ≤ bawah < atas ≤ 100.")
    s, w = _prep(gr, "GR")
    valid = s.dropna()
    if len(valid) < 10:
        raise PetrophysicsError("Minimal 10 sampel GR valid untuk menghitung persentil.")
    gmin, gmax = (float(v) for v in np.percentile(valid, [lo, hi]))
    if gmax <= gmin:
        raise PetrophysicsError("Persentil menghasilkan GR_max ≤ GR_min; data GR hampir konstan.")
    w.append("GR_min dan GR_max berasal dari persentil data interval ini: estimasi statistik, "
             "bukan baseline shale atau clean yang pasti secara geologi.")
    return gmin, gmax, w


def gamma_ray_index(gr, gr_min: float, gr_max: float, clip: bool = False) -> CalcResult:
    """IGR = (GR_log - GR_min) / (GR_max - GR_min). GR dan batas dalam satuan yang sama (gAPI)."""
    gmin = _scalar(gr_min, "GR_min")
    gmax = _scalar(gr_max, "GR_max")
    if gmax <= gmin:
        raise PetrophysicsError(f"GR_max ({gmax:g}) harus lebih besar dari GR_min ({gmin:g}).")
    s, w = _prep(gr, "GR")
    n_neg = int((s < 0).sum())
    if n_neg:
        w.append(f"{n_neg} nilai GR negatif (tidak lazim); dipertahankan, periksa QC.")
    igr = ((s - gmin) / (gmax - gmin)).rename("IGR")
    n_out = int(((igr < 0) | (igr > 1)).sum())
    if n_out:
        w.append(f"{n_out} nilai IGR di luar 0–1 (GR di luar GR_min/GR_max).")
    if clip:
        igr = igr.clip(0.0, 1.0)
        w.append(f"IGR di-clip ke 0–1 ({n_out} nilai diubah).")
    return CalcResult(igr, w)


def vsh_from_igr(igr, method: str = "linear", clip: bool = True) -> CalcResult:
    """Vsh dari IGR: linear, Larionov Tersier, atau Larionov batuan lebih tua.

    Pilihan umur batuan harus datang dari pengguna; tidak ada metode yang dipilih otomatis.
    """
    if method not in VSH_METHODS:
        raise PetrophysicsError(f"Metode Vsh tidak dikenal: {method!r}.")
    s, w = _prep(igr, "IGR")
    with np.errstate(over="ignore", invalid="ignore"):
        if method == "linear":
            v = s.copy()
        elif method == "larionov_tertiary":
            v = 0.083 * (np.power(2.0, 3.7 * s) - 1.0)
        else:
            v = 0.33 * (np.power(2.0, 2.0 * s) - 1.0)
    v = pd.Series(np.asarray(v, dtype=float), index=s.index, name="VSH")
    n_out = int(((v < 0) | (v > 1)).sum())
    if n_out and not clip:
        w.append(f"{n_out} nilai Vsh di luar 0–1; dipertahankan karena clipping tidak dipilih.")
    if clip:
        if n_out:
            w.append(f"{n_out} nilai Vsh di-clip ke 0–1.")
        v = v.clip(0.0, 1.0)
    return CalcResult(v, w)


# ------------------------------------------------------------------ porositas
def density_porosity(rhob, rho_matrix: float, rho_fluid: float) -> CalcResult:
    """PHID = (rho_ma - RHOB) / (rho_ma - rho_f). Semua densitas dalam g/cc."""
    rma = _scalar(rho_matrix, "Densitas matriks", positive=True)
    rf = _scalar(rho_fluid, "Densitas fluida", positive=True)
    if rma <= rf:
        raise PetrophysicsError(
            f"Densitas matriks ({rma:g}) harus lebih besar dari densitas fluida ({rf:g}); "
            "selisihnya adalah penyebut rumus."
        )
    s, w = _prep(rhob, "RHOB")
    bad = s <= 0
    if bad.any():
        w.append(f"{int(bad.sum())} nilai RHOB ≤ 0 dianggap tidak valid (NaN).")
    s = s.mask(bad)
    phid = ((rma - s) / (rma - rf)).rename("PHID")
    w += _range_warning(phid, 0.0, 1.0, "PHID")
    return CalcResult(phid, w)


def nphi_unit_hint(nphi) -> str:
    """Petunjuk satuan NPHI untuk UI: 'fraction', 'percent', atau 'ambiguous'. TIDAK dipakai untuk konversi."""
    s = pd.to_numeric(pd.Series(nphi), errors="coerce").dropna()
    if s.empty:
        return "ambiguous"
    p99 = float(np.percentile(s, 99))
    if p99 > 1.5:
        return "percent"
    if p99 <= 1.0:
        return "fraction"
    return "ambiguous"


def neutron_porosity(nphi, unit: str) -> CalcResult:
    """Porositas neutron (PHIN) dalam fraction. Satuan harus dipilih eksplisit: 'fraction' atau 'percent'."""
    if unit not in ("fraction", "percent"):
        raise PetrophysicsError("Satuan NPHI harus dipilih eksplisit ('fraction' atau 'percent'); tidak ditebak.")
    s, w = _prep(nphi, "NPHI")
    if unit == "percent":
        s = s / 100.0
        w.append("NPHI dikonversi dari persen ke fraction (÷100) atas pilihan pengguna.")
    s = s.rename("PHIN")
    w += _range_warning(s, 0.0, 1.0, "PHIN")
    return CalcResult(s, w)


def sonic_porosity(dt, dt_matrix: float, dt_fluid: float) -> CalcResult:
    """Wyllie time-average: PHIS = (DT - DT_ma) / (DT_f - DT_ma). DT, DT_ma, DT_f harus satuan sama (mis. us/ft)."""
    dm = _scalar(dt_matrix, "DT matriks", positive=True)
    dfl = _scalar(dt_fluid, "DT fluida", positive=True)
    if dfl <= dm:
        raise PetrophysicsError(f"DT fluida ({dfl:g}) harus lebih besar dari DT matriks ({dm:g}).")
    s, w = _prep(dt, "DT")
    bad = s <= 0
    if bad.any():
        w.append(f"{int(bad.sum())} nilai DT ≤ 0 dianggap tidak valid (NaN).")
    s = s.mask(bad)
    phis = ((s - dm) / (dfl - dm)).rename("PHIS")
    w += _range_warning(phis, 0.0, 1.0, "PHIS")
    return CalcResult(phis, w)


_TOTAL_NEEDS = {
    "density": ("phid",),
    "neutron": ("phin",),
    "sonic": ("phis",),
    "density_neutron_mean": ("phid", "phin"),
    "density_neutron_rms": ("phid", "phin"),
}


def total_porosity(method: str, phid=None, phin=None, phis=None) -> CalcResult:
    """Porositas total (PHIT) dengan metode yang disebutkan eksplisit."""
    if method not in _TOTAL_NEEDS:
        raise PetrophysicsError(f"Metode porositas total tidak dikenal: {method!r}.")
    given = {"phid": phid, "phin": phin, "phis": phis}
    missing = [k.upper() for k in _TOTAL_NEEDS[method] if given[k] is None]
    if missing:
        raise PetrophysicsError(f"Input tidak cukup untuk metode {method}: {', '.join(missing)} tidak tersedia.")
    prepped: list[pd.Series] = []
    w: list[str] = []
    for key in _TOTAL_NEEDS[method]:
        s, ww = _prep(given[key], key.upper())
        prepped.append(s)
        w += ww
    arrays, idx = _align(*prepped)
    if len(arrays) == 1:
        phit = arrays[0].copy()
    elif method == "density_neutron_mean":
        phit = (arrays[0] + arrays[1]) / 2.0
    else:
        a, b = arrays
        bad = (a < 0) | (b < 0)
        if bad.any():
            w.append(f"{int(bad.sum())} sampel dengan PHID atau PHIN negatif tidak dihitung pada metode RMS (NaN).")
        a = np.where(bad, np.nan, a)
        b = np.where(bad, np.nan, b)
        phit = np.sqrt((a ** 2 + b ** 2) / 2.0)
    phit_s = pd.Series(phit, index=idx, name="PHIT")
    w += _range_warning(phit_s, 0.0, 1.0, "PHIT")
    return CalcResult(phit_s, w)


def effective_porosity(phit, vsh) -> CalcResult:
    """PHIE = PHIT × (1 - Vsh). Koreksi shale sederhana; bukan satu-satunya definisi porositas efektif."""
    pt, w1 = _prep(phit, "PHIT")
    v, w2 = _prep(vsh, "Vsh")
    (a_pt, a_v), idx = _align(pt, v)
    w = w1 + w2
    bad = (a_v < 0) | (a_v > 1)
    if bad.any():
        w.append(f"{int(bad.sum())} nilai Vsh di luar 0–1 tidak dipakai (hasil NaN). "
                 "Aktifkan clipping Vsh jika memang diinginkan.")
    a_v = np.where(bad, np.nan, a_v)
    phie = pd.Series(a_pt * (1.0 - a_v), index=idx, name="PHIE")
    w += _range_warning(phie, 0.0, 1.0, "PHIE")
    return CalcResult(phie, w)


# ------------------------------------------------------------------ saturasi, BVW
def archie_sw(rt, phie, rw: float, a: float = 1.0, m: float = 2.0, n: float = 2.0,
              clip: bool = False) -> CalcResult:
    """Archie: Sw = ((a × Rw) / (Rt × PHIE^m))^(1/n). Rt dan Rw dalam ohm.m (Rw pada suhu formasi)."""
    rw_ = _scalar(rw, "Rw", positive=True)
    a_ = _scalar(a, "a", positive=True)
    m_ = _scalar(m, "m", positive=True)
    n_ = _scalar(n, "n", positive=True)
    r, w1 = _prep(rt, "Rt")
    p, w2 = _prep(phie, "PHIE")
    (ra, pa), idx = _align(r, p)
    w = w1 + w2
    bad_r, bad_p = ra <= 0, pa <= 0
    if bad_r.any():
        w.append(f"{int(bad_r.sum())} nilai Rt ≤ 0 tidak valid (NaN).")
    if bad_p.any():
        w.append(f"{int(bad_p.sum())} nilai PHIE ≤ 0 tidak valid (NaN): Archie tidak dapat dihitung pada porositas nol.")
    ra = np.where(bad_r, np.nan, ra)
    pa = np.where(bad_p, np.nan, pa)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sw = ((a_ * rw_) / (ra * pa ** m_)) ** (1.0 / n_)
    inf = np.isinf(sw)
    if inf.any():
        sw = np.where(inf, np.nan, sw)
        w.append(f"{int(inf.sum())} hasil Sw tak hingga dianggap NaN.")
    sw_s = pd.Series(sw, index=idx, name="SW")
    n_gt1 = int((sw_s > 1).sum())
    if n_gt1:
        w.append(f"{n_gt1} nilai Sw > 1 (secara fisik tidak mungkin). Periksa Rw, a, m, n, PHIE, dan Rt.")
    if clip:
        sw_s = sw_s.clip(0.0, 1.0)
        w.append(f"Sw di-clip ke 0–1 ({n_gt1} nilai diubah). Clipping bukan validasi fisik.")
    return CalcResult(sw_s, w)


def bulk_volume_water(phie, sw) -> CalcResult:
    """BVW = PHIE × Sw. Bukan indikator tunggal produktivitas."""
    p, w1 = _prep(phie, "PHIE")
    s, w2 = _prep(sw, "Sw")
    (pa, sa), idx = _align(p, s)
    return CalcResult(pd.Series(pa * sa, index=idx, name="BVW"), w1 + w2)


# ------------------------------------------------------------------ permeabilitas
def _perm_inputs(phie, swirr):
    p, w1 = _prep(phie, "PHIE")
    s, w2 = _prep(swirr, "Swirr")
    (pa, sa), idx = _align(p, s)
    w = w1 + w2
    valid = (pa > 0) & (pa < 1) & (sa > 0) & (sa < 1)
    n_bad = int((~valid & ~np.isnan(pa) & ~np.isnan(sa)).sum())
    if n_bad:
        w.append(f"{n_bad} sampel dengan PHIE atau Swirr di luar (0, 1) tidak dihitung (NaN).")
    return np.where(valid, pa, np.nan), np.where(valid, sa, np.nan), idx, w


def permeability_timur(phie, swirr) -> CalcResult:
    """ESTIMASI permeabilitas (mD): k = 0.136 × (100·PHIE)^4.4 / (100·Swirr)^2. PHIE dan Swirr fraction."""
    pa, sa, idx, w = _perm_inputs(phie, swirr)
    k = 0.136 * (100.0 * pa) ** 4.4 / (100.0 * sa) ** 2
    return CalcResult(pd.Series(k, index=idx, name="PERM_EST"), w)


def permeability_coates(phie, swirr) -> CalcResult:
    """ESTIMASI permeabilitas (mD): k = (100 × PHIE² × (1 - Swirr) / Swirr)². PHIE dan Swirr fraction."""
    pa, sa, idx, w = _perm_inputs(phie, swirr)
    k = (100.0 * pa ** 2 * (1.0 - sa) / sa) ** 2
    return CalcResult(pd.Series(k, index=idx, name="PERM_EST"), w)


# ------------------------------------------------------------------ pipeline
@dataclass
class PetroParams:
    """Seluruh parameter satu kali perhitungan. Disimpan bersama hasil agar dapat direproduksi."""

    gr_min: float | None = None
    gr_max: float | None = None
    clip_igr: bool = True
    vsh_method: str = "linear"
    clip_vsh: bool = True
    rho_matrix: float = 2.65
    rho_fluid: float = 1.0
    dt_matrix: float = 55.5
    dt_fluid: float = 189.0
    nphi_unit: str | None = None
    phit_method: str = "density"
    shale_correction: bool = True
    sat_model: str = "archie"
    rw: float | None = None
    a: float = 1.0
    m: float = 2.0
    n: float = 2.0
    perm_method: str = "none"  # "none", "timur", "coates"
    swirr: float | None = None
    swirr_from_sw: bool = False


@dataclass
class PetroOutput:
    table: pd.DataFrame
    status: dict[str, str]
    warnings: list[str]
    params: dict
    curve_map: dict


def compute_petrophysics(df: pd.DataFrame, depth_col: str, curve_map: dict[str, str],
                         params: PetroParams) -> PetroOutput:
    """Jalankan seluruh rantai perhitungan. Langkah yang tidak dapat dihitung dilaporkan di `status`
    dan kolomnya TIDAK dibuat (tidak ada angka karangan).

    curve_map: peran -> nama kolom, mis. {"GR": "GR", "RHOB": "RHOZ", "NPHI": "NPOR", "DT": "DTC", "RT": "ILD"}.
    """
    if depth_col not in df.columns:
        raise PetrophysicsError(f"Kolom kedalaman {depth_col!r} tidak ada di data.")
    df = df.reset_index(drop=True)
    p = params
    status: dict[str, str] = {}
    warnings: list[str] = []

    table = pd.DataFrame(index=df.index)
    table[depth_col] = pd.to_numeric(df[depth_col], errors="coerce")
    depth = table[depth_col]
    if depth.isna().any():
        warnings.append(f"[DEPTH] {int(depth.isna().sum())} nilai kedalaman kosong atau non-numerik.")
    valid_depth = depth.dropna()
    if not valid_depth.is_monotonic_increasing:
        warnings.append("[DEPTH] Kedalaman tidak berurutan naik; urutkan data sebelum interpretasi.")
    if valid_depth.duplicated().any():
        warnings.append(f"[DEPTH] Ada {int(valid_depth.duplicated().sum())} kedalaman duplikat.")

    for role, name in curve_map.items():
        if name and name in df.columns:
            table[f"{role}_log"] = pd.to_numeric(df[name], errors="coerce")

    def col(role: str) -> pd.Series:
        name = curve_map.get(role)
        if not name:
            raise PetrophysicsError(f"Kurva {role} belum dipetakan.")
        if name not in df.columns:
            raise PetrophysicsError(f"Kolom {name!r} (untuk {role}) tidak ada di data.")
        return df[name]

    def need(value, message: str):
        if value is None:
            raise PetrophysicsError(message)
        return value

    def step(key: str, fn) -> CalcResult | None:
        try:
            res = fn()
        except PetrophysicsError as exc:
            status[key] = f"cannot calculate: {exc}"
            return None
        status[key] = "calculated"
        warnings.extend(f"[{key}] {w}" for w in res.warnings)
        return res

    def store(key: str, fn) -> pd.Series | None:
        res = step(key, fn)
        if res is None:
            return None
        table[key] = res.values
        return res.values

    # IGR (simpan nilai mentah dan nilai yang dipakai)
    igr = None
    igr_res = step("IGR", lambda: gamma_ray_index(
        col("GR"), need(p.gr_min, "GR_min belum ditentukan."), need(p.gr_max, "GR_max belum ditentukan."), clip=False))
    if igr_res is not None:
        table["IGR_RAW"] = igr_res.values
        igr = igr_res.values.clip(0.0, 1.0) if p.clip_igr else igr_res.values
        table["IGR"] = igr
        if p.clip_igr:
            warnings.append("[IGR] IGR di-clip ke 0–1 sesuai pilihan; nilai sebelum clipping ada di kolom IGR_RAW.")

    # Vsh
    vsh = None
    if igr is None:
        status["VSH"] = "cannot calculate: IGR tidak tersedia."
    else:
        vsh = store("VSH", lambda: vsh_from_igr(igr, p.vsh_method, clip=p.clip_vsh))

    # Porositas
    phid = store("PHID", lambda: density_porosity(col("RHOB"), p.rho_matrix, p.rho_fluid))
    phin = store("PHIN", lambda: neutron_porosity(
        col("NPHI"), need(p.nphi_unit, "Satuan NPHI belum dipilih (fraction atau percent).")))
    phis = store("PHIS", lambda: sonic_porosity(col("DT"), p.dt_matrix, p.dt_fluid))
    phit = store("PHIT", lambda: total_porosity(p.phit_method, phid=phid, phin=phin, phis=phis))

    def _phie() -> CalcResult:
        base = need(phit, "PHIT tidak tersedia.")
        if not p.shale_correction:
            return CalcResult(base.copy().rename("PHIE"),
                              ["PHIE disamakan dengan PHIT karena koreksi shale dimatikan; "
                               "ini bukan porositas efektif terkoreksi shale."])
        return effective_porosity(base, need(vsh, "Vsh tidak tersedia (butuh kurva GR) untuk koreksi shale."))

    phie = store("PHIE", _phie)

    # Saturasi air
    def _sw() -> CalcResult:
        info = SATURATION_MODELS.get(p.sat_model)
        if info is None:
            raise PetrophysicsError(f"Model saturasi tidak dikenal: {p.sat_model!r}.")
        if not info["available"]:
            raise PetrophysicsError(f"Model {info['label']} belum tersedia: {info['reason']}")
        rt = col("RT")
        return archie_sw(rt, need(phie, "PHIE tidak tersedia."),
                         need(p.rw, "Rw belum diisi. Aplikasi tidak menebak Rw."), p.a, p.m, p.n)

    sw = store("SW", _sw)
    store("BVW", lambda: bulk_volume_water(need(phie, "PHIE tidak tersedia."), need(sw, "Sw tidak tersedia.")))

    # Permeabilitas (opsional)
    if p.perm_method == "none":
        status["PERM_EST"] = "tidak diminta"
    else:
        def _perm() -> CalcResult:
            if p.perm_method not in ("timur", "coates"):
                raise PetrophysicsError(f"Metode permeabilitas tidak dikenal: {p.perm_method!r}.")
            ph = need(phie, "PHIE tidak tersedia.")
            if p.swirr_from_sw:
                swirr = need(sw, "Sw tidak tersedia untuk dipakai sebagai proksi Swirr.")
            else:
                swirr = need(p.swirr, "Swirr belum diisi. Aplikasi tidak menebak Swirr.")
            return (permeability_timur if p.perm_method == "timur" else permeability_coates)(ph, swirr)

        if store("PERM_EST", _perm) is not None and p.swirr_from_sw:
            warnings.append("[PERM_EST] Sw dipakai sebagai proksi Swirr. Ini hanya masuk akal pada zona yang benar-benar "
                            "berada pada saturasi air irreducible; di luar itu hasilnya menyesatkan.")

    # Flag kualitas data
    flags = pd.Series("", index=table.index, dtype=object)

    def flag(mask: pd.Series, label: str) -> None:
        m = mask.astype(bool)
        flags.loc[m] = flags.loc[m] + label + ";"

    log_cols = [c for c in table.columns if c.endswith("_log")]
    if log_cols:
        flag(table[log_cols].isna().any(axis=1), "INPUT_NA")
    if "IGR_RAW" in table:
        flag((table["IGR_RAW"] < 0) | (table["IGR_RAW"] > 1), "IGR_OUT")
    for c in ("PHID", "PHIN", "PHIS", "PHIT"):
        if c in table:
            flag((table[c] < 0) | (table[c] > 1), f"{c}_OUT")
    if "PHIE" in table:
        flag(table["PHIE"] <= 0, "PHIE<=0")
    if "RT_log" in table:
        flag(table["RT_log"] <= 0, "RT<=0")
    if "SW" in table:
        flag(table["SW"] > 1, "SW>1")
    table["QC_FLAGS"] = flags

    return PetroOutput(table=table, status=status, warnings=warnings, params=asdict(p), curve_map=dict(curve_map))


# ------------------------------------------------------------------ dokumentasi metode
METHOD_DOCS = [
    {
        "name": "Gamma Ray Index (IGR)",
        "equations": [r"I_{GR} = \dfrac{GR_{log} - GR_{min}}{GR_{max} - GR_{min}}"],
        "variables": "GR_log: gamma ray (gAPI). GR_min: baseline sand/clean (gAPI). GR_max: baseline shale (gAPI). IGR tanpa satuan.",
        "assumptions": "GR bertambah seiring kandungan shale. GR_min dan GR_max mewakili baseline zona bersih dan zona shale.",
        "limitations": "GR_min/GR_max dari persentil adalah estimasi statistik, bukan baseline pasti. Mineral radioaktif non-shale "
                       "(feldspar, mika, glaukonit) membuat IGR menaksir shale terlalu tinggi.",
        "reference": "Bentuk standar; tanpa sitasi khusus.",
    },
    {
        "name": "Volume shale (Vsh)",
        "equations": [
            r"\text{Linear: } V_{sh} = I_{GR}",
            r"\text{Larionov Tersier: } V_{sh} = 0.083\,(2^{3.7\,I_{GR}} - 1)",
            r"\text{Larionov batuan lebih tua: } V_{sh} = 0.33\,(2^{2\,I_{GR}} - 1)",
        ],
        "variables": "IGR tanpa satuan; Vsh fraction (0–1).",
        "assumptions": "Pilihan Larionov bergantung umur batuan, yang harus ditentukan pengguna dari konteks geologi.",
        "limitations": "Vsh dari GR adalah estimasi, bukan persen mineral lempung hasil pengukuran. "
                       "Metode linear cenderung menaksir shale lebih tinggi daripada Larionov.",
        "reference": "Larionov (1969), Borehole Radiometry (Moskow).",
    },
    {
        "name": "Porositas densitas (PHID)",
        "equations": [r"\phi_D = \dfrac{\rho_{ma} - \rho_b}{\rho_{ma} - \rho_f}"],
        "variables": "ρb: RHOB (g/cc). ρma: densitas matriks (g/cc). ρf: densitas fluida (g/cc). Hasil fraction.",
        "assumptions": "Batuan dua komponen (matriks + fluida) dengan densitas matriks dan fluida yang benar.",
        "limitations": "Preset matriks hanya nilai awal. Gas, shale, dan litologi campuran membiaskan hasil. "
                       "Nilai di luar 0–1 dipertahankan dan ditandai, bukan dihapus.",
        "reference": "Persamaan volumetrik dua komponen; tanpa sitasi khusus.",
    },
    {
        "name": "Porositas neutron (PHIN)",
        "equations": [r"\phi_N = NPHI \quad (\text{fraction; jika persen: } NPHI/100)"],
        "variables": "NPHI dalam fraction (v/v) atau persen. Satuan dipilih pengguna.",
        "assumptions": "Skala log neutron sesuai litologi kalibrasi alat.",
        "limitations": "Dipengaruhi shale, gas, dan litologi. NPHI bukan porositas efektif terkoreksi.",
        "reference": "Tanpa sitasi khusus.",
    },
    {
        "name": "Porositas sonic (Wyllie time-average)",
        "equations": [r"\phi_S = \dfrac{\Delta t_{log} - \Delta t_{ma}}{\Delta t_{f} - \Delta t_{ma}}"],
        "variables": "Δt_log, Δt_ma, Δt_f: waktu tempuh dalam satuan yang sama (mis. µs/ft).",
        "assumptions": "Batuan terkonsolidasi dan tersaturasi cairan; porositas intergranular.",
        "limitations": "Kurang akurat pada batuan lemah terkonsolidasi, shale, rekahan, dan gas.",
        "reference": "Wyllie, Gregory & Gardner (1956), Geophysics 21(1).",
    },
    {
        "name": "Porositas total dan efektif",
        "equations": [
            r"\phi_{ND} = \dfrac{\phi_D + \phi_N}{2} \quad \text{(rata-rata aritmetika)}",
            r"\phi_{ND} = \sqrt{\dfrac{\phi_D^2 + \phi_N^2}{2}} \quad \text{(RMS)}",
            r"\phi_E = \phi_T\,(1 - V_{sh})",
        ],
        "variables": "PHIT: porositas total terpilih (fraction). PHIE: porositas efektif (fraction).",
        "assumptions": "Kedua rata-rata adalah pendekatan sederhana, bukan crossplot lengkap dengan koreksi litologi.",
        "limitations": "Koreksi shale PHIT×(1−Vsh) hanya salah satu definisi. Tidak menggantikan analisis multi-mineral.",
        "reference": "Pendekatan umum; tanpa sitasi khusus.",
    },
    {
        "name": "Saturasi air (Archie)",
        "equations": [r"S_w = \left( \dfrac{a\,R_w}{R_t\,\phi_E^{\,m}} \right)^{1/n}"],
        "variables": "Rw: resistivitas air formasi (ohm.m, suhu formasi). Rt: resistivitas formasi (ohm.m). "
                     "a: tortuosity, m: cementation, n: saturation exponent.",
        "assumptions": "Batuan bersih (tanpa shale konduktif), air formasi seragam, Rt sudah terkoreksi invasi dan efek lubang bor.",
        "limitations": "Pada shaly sand Archie menaksir Sw terlalu tinggi. Rw, a, m, n harus dari pengguna dan idealnya dikalibrasi.",
        "reference": "Archie (1942), Trans. AIME 146.",
    },
    {
        "name": "Bulk Volume Water (BVW)",
        "equations": [r"BVW = \phi_E \cdot S_w"],
        "variables": "Fraction (v/v).",
        "assumptions": "PHIE dan Sw dihitung dengan metode yang sudah dipilih.",
        "limitations": "Tidak boleh dipakai sendirian untuk menyatakan zona produktif.",
        "reference": "Definisi standar.",
    },
    {
        "name": "Permeabilitas estimasi (Timur, Coates)",
        "equations": [
            r"\text{Timur: } k = \dfrac{0.136\,(100\,\phi_E)^{4.4}}{(100\,S_{wirr})^{2}}",
            r"\text{Coates: } k = \left( \dfrac{100\,\phi_E^{2}\,(1 - S_{wirr})}{S_{wirr}} \right)^{2}",
        ],
        "variables": "k: estimasi permeabilitas (mD). PHIE dan Swirr dalam fraction. Swirr: saturasi air irreducible dari pengguna.",
        "assumptions": "Hubungan empiris pasir; Swirr harus mewakili kondisi irreducible.",
        "limitations": "BUKAN permeabilitas terukur. Konstanta empiris bergantung formasi dan perlu kalibrasi dengan core. "
                       "Tanpa data core, hasil belum divalidasi untuk sumur ini.",
        "reference": "Timur (1968); Coates & Dumanoir (1974).",
    },
]
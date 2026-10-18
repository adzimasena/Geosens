import json

import pandas as pd
import streamlit as st

from core import facies as F
from core.report import build_html_report

DISCLAIMER = ("Laporan ini dihasilkan otomatis dari hasil estimasi di GeoPetroML. "
              "Ini alat bantu dan bukan pengganti interpretasi geolog atau petrofisikawan. "
              "Net pay dari log bukan zona produktif; itu memerlukan data uji fluida dan produksi.")


def _candidates():
    """Semua tabel hasil di session_state selain data mentah dan widget."""
    found = {}
    for key, val in st.session_state.items():
        if key in ("well_data",) or str(key).startswith(("_", "corr_wells")):
            continue
        if isinstance(val, pd.DataFrame) and not val.empty:
            found[str(key)] = val
        elif isinstance(val, dict):
            for k2, v2 in val.items():
                if isinstance(v2, pd.DataFrame) and not v2.empty:
                    found[f"{key}.{k2}"] = v2
    return found


def show_report(df):
    st.subheader("Report Export")
    st.write("Gabungkan hasil analisis yang sudah dihitung menjadi satu laporan HTML. "
             "Hanya hasil yang sudah dihitung pada sesi ini yang tersedia.")

    title = st.text_input("Judul laporan", value="Laporan Analisis GeoPetroML")
    well = st.text_input("Nama sumur / lapangan (opsional)")

    sections = []
    info = {"File data": st.session_state.get("uploaded_filename") or "-",
            "Jumlah baris": len(df), "Jumlah kolom": len(df.columns)}
    if well.strip():
        info = {"Sumur / lapangan": well.strip(), **info}
    sections.append(("Informasi data", info))

    data, depth = F.prepare_logs(df)
    if data is not None:
        stats = data.drop(columns=[depth]).describe().T[["count", "min", "mean", "max"]]
        stats.insert(0, "Kurva", stats.index)
        stats.insert(1, "Kedalaman", f"{data[depth].min():g} - {data[depth].max():g}")
        sections.append(("Ringkasan kurva", stats.reset_index(drop=True)))

    st.write("Hasil yang tersedia")
    cands = _candidates()
    fr = st.session_state.get("facies_result")
    if fr is not None:
        cands["Fasies: ringkasan"] = F.facies_summary(fr["data"], fr["labels"], fr["depth"], fr["features"])
        cands["Fasies: interval"] = F.facies_intervals(fr["labels"], fr["depth"])
    ml = st.session_state.get("ml_result")
    if ml is not None:
        cands["ML: matriks konfusi"] = ml["confusion"].rename_axis("Kelas").reset_index()
        cands["ML: pentingnya fitur"] = ml["importance"].rename("Penting").rename_axis("Fitur").reset_index()

    # sembunyikan kunci mentah yang sudah diwakili tabel di atas
    hidden = {"facies_result.labels", "facies_result.data", "ml_result.confusion",
              "ml_result.report", "ml_result.importance", "ml_result.class_counts"}
    cands = {k: v for k, v in cands.items() if k not in hidden}

    if not cands:
        st.info("Belum ada hasil. Jalankan Petrophysics, Reservoir Evaluation, Facies, "
                "atau Machine Learning terlebih dahulu.")
    chosen = st.multiselect("Pilih hasil yang dimasukkan", list(cands), default=list(cands))
    for name in chosen:
        sections.append((name, cands[name]))

    if ml is not None and "ML: matriks konfusi" in chosen:
        sections.append(("ML: parameter dan skor", {
            "Fitur": ", ".join(ml["features"]), "Sumber label": ml["target_source"],
            "Akurasi uji": f"{ml['accuracy']:.3f}",
            "Akurasi seimbang": f"{ml['balanced_accuracy']:.3f}", **ml["params"]}))

    notes = st.text_area("Catatan tambahan (opsional)")
    if notes.strip():
        sections.append(("Catatan", notes.strip()))

    html_doc = build_html_report(title, sections, DISCLAIMER)
    st.write(f"Isi laporan: {len(sections)} bagian.")
    st.download_button("Unduh laporan (HTML)", html_doc.encode("utf-8"),
                       file_name="laporan_geopetroml.html", mime="text/html")
    st.caption("Buka file HTML di browser; untuk PDF gunakan Cetak → Simpan sebagai PDF.")

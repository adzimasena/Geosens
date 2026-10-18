import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.Data_Loader import load_well_data
from views.lithology import show_lithology
from views.petrophysics import show_petrophysics
from views.reservoir import show_reservoir
from views.well_log import show_well_log
from views.facies import show_facies
from views.machine_learning import show_ml
from views.report_export import show_report

st.set_page_config(
    page_title="Geosens",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Menyiapkan tempat penyimpanan data selama sesi aplikasi
if "well_data" not in st.session_state:
    st.session_state.well_data = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None


# Sidebar dan navigasi
st.sidebar.title("🪨 Geosens")
st.sidebar.caption("Geological & Petrophysical Analysis")

menu = st.sidebar.radio(
    "Navigasi",
    [
        "Dashboard",
        "Data Management",
        "Well Log",
        "Lithology",
        "Facies",
        "Petrophysics",
        "Reservoir Evaluation",
        "Machine Learning",
        "Report Export",
    ]
)

st.title("Geosens")
st.caption("Geological and Petrophysical Analysis")


# Dashboard
if menu == "Dashboard":
    st.header("Dashboard")

    df = st.session_state.well_data

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Jumlah Baris Data",
        len(df) if df is not None else 0
    )

    col2.metric(
        "Jumlah Kolom",
        len(df.columns) if df is not None else 0
    )

    col3.metric(
        "Status Data",
        "Tersedia" if df is not None else "Belum tersedia"
    )

    if df is not None:
        st.write("File aktif:", st.session_state.uploaded_filename)
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("Silakan unggah data melalui menu Data Management.")


# Data Management
elif menu == "Data Management":
    st.header("Data Management")
    st.write("Unggah data sumur untuk memulai analisis.")

    uploaded_file = st.file_uploader(
        "Pilih file LAS, CSV, atau Excel",
        type=["las", "csv", "xlsx", "xls"],
        key="well_file_uploader"
    )

    if uploaded_file is not None:
        try:
            from core.Data_Loader import load_well_data

            df = load_well_data(uploaded_file)

            if df is None or df.empty:
                st.warning("File tidak berisi data yang dapat digunakan.")
            else:
                st.session_state.well_data = df
                st.session_state.uploaded_filename = uploaded_file.name

                st.success("Data berhasil dibaca dan disimpan.")

        except ImportError as e:
            st.error(
                "Modul pembaca data tidak dapat diimpor. "
                "Periksa file core/data_loader.py dan dependensinya."
            )
            st.caption(f"Detail error: {e}")

        except Exception as e:
            st.error("Gagal membaca file. Periksa format dan isi file.")
            st.caption(f"Detail error: {type(e).__name__}: {e}")

    df = st.session_state.well_data

    if df is not None:
        st.subheader("Pratinjau Data")
        st.dataframe(df.head(100), use_container_width=True)

        st.subheader("Informasi Data")
        col1, col2 = st.columns(2)
        col1.metric("Jumlah Baris", len(df))
        col2.metric("Jumlah Kolom", len(df.columns))

        st.write("Nama kolom:", list(df.columns))

        if st.button("Hapus Data Aktif"):
            st.session_state.well_data = None
            st.session_state.uploaded_filename = None
            st.rerun()



# Menu analisis
elif menu in [
    "Well Log",
    "Lithology",
    "Facies",
    "Petrophysics",
    "Reservoir Evaluation",
    "Machine Learning",
    "Report Export",
]:
    st.header(menu)

    df = st.session_state.well_data

    if df is None:
        st.warning(
            "Belum ada data sumur. "
            "Unggah file melalui Data Management terlebih dahulu."
        )

    elif menu == "Well Log":
        show_well_log(df)

    elif menu == "Lithology":
        st.header("Lithology Interpretation")
        st.write(
            "Multi-track well log untuk mendukung "
            "interpretasi litologi awal."
        )

        df = st.session_state.well_data

        if df is None:
            st.warning(
                "Unggah data sumur terlebih dahulu melalui "
                "menu Data Management."
            )

        else:
            df = df.copy()
            df.columns = [
                str(col).strip().upper()
                for col in df.columns
            ]

            # Identifikasi kolom kedalaman dan log
            depth_candidates = [
                "DEPTH", "DEPT", "MD", "DEPTH.M", "DEPT.M"
            ]

            depth_col = next(
                (c for c in depth_candidates if c in df.columns),
                None
            )

            if depth_col is None:
                st.error(
                    "Kolom kedalaman tidak ditemukan. "
                    "Gunakan DEPTH, DEPT, atau MD."
                )

            else:
                df[depth_col] = pd.to_numeric(
                    df[depth_col], errors="coerce"
                )

                log_candidates = [
                    "GR", "RHOB", "NPHI", "RT"
                ]

                available_logs = [
                    c for c in log_candidates
                    if c in df.columns
                ]

                for col in available_logs:
                    df[col] = pd.to_numeric(
                        df[col], errors="coerce"
                    )

                df = df.dropna(
                    subset=[depth_col]
                ).sort_values(depth_col)

                if df.empty:
                    st.error(
                        "Tidak ditemukan data kedalaman valid."
                    )

                elif not available_logs:
                    st.error(
                        "Tidak ditemukan log GR, RHOB, NPHI, "
                        "atau RT pada data."
                    )

                else:
                    st.subheader("1. Log yang tersedia")

                    st.write(
                        "Kurva ditemukan:",
                        ", ".join(available_logs)
                    )

                    st.subheader("2. Pengaturan tampilan")

                    selected_logs = st.multiselect(
                        "Pilih kurva untuk ditampilkan",
                        options=available_logs,
                        default=available_logs
                    )

                    depth_min = float(df[depth_col].min())
                    depth_max = float(df[depth_col].max())

                    if depth_min < depth_max:
                        depth_range = st.slider(
                            "Rentang kedalaman",
                            min_value=depth_min,
                            max_value=depth_max,
                            value=(depth_min, depth_max)
                        )

                        plot_df = df[
                            df[depth_col].between(
                                depth_range[0],
                                depth_range[1]
                            )
                        ].copy()
                    else:
                        plot_df = df.copy()

                    # Interpretasi indikator GR
                    st.subheader("3. Indikator kandungan serpih")

                    if "GR" in available_logs:
                        clean_limit = st.number_input(
                            "Batas GR rendah",
                            min_value=0.0,
                            max_value=200.0,
                            value=50.0,
                            step=5.0
                        )

                        shale_limit = st.number_input(
                            "Batas GR tinggi",
                            min_value=0.0,
                            max_value=200.0,
                            value=75.0,
                            step=5.0
                        )

                        if shale_limit < clean_limit:
                            st.error(
                                "Batas GR tinggi harus lebih besar "
                                "atau sama dengan batas GR rendah."
                            )
                            st.stop()

                        def classify_gr(value):
                            if pd.isna(value):
                                return "No data"
                            if value <= clean_limit:
                                return "GR rendah"
                            if value <= shale_limit:
                                return "GR menengah"
                            return "GR tinggi"

                        plot_df["GR_INDICATOR"] = (
                            plot_df["GR"].apply(classify_gr)
                        )

                    # Multi-track plot
                    st.subheader("4. Multi-track well log")

                    if not selected_logs:
                        st.info(
                            "Pilih minimal satu kurva untuk "
                            "menampilkan grafik."
                        )

                    elif plot_df.empty:
                        st.warning(
                            "Tidak ada data pada rentang kedalaman ini."
                        )

                    else:
                        n_tracks = len(selected_logs)
                        fig = make_subplots(
                            rows=1,
                            cols=n_tracks,
                            shared_yaxes=True,
                            subplot_titles=selected_logs,
                            horizontal_spacing=0.04
                        )

                        for i, log in enumerate(
                            selected_logs, start=1
                        ):
                            curve = plot_df[
                                [depth_col, log]
                            ].dropna()

                            fig.add_trace(
                                go.Scatter(
                                    x=curve[log],
                                    y=curve[depth_col],
                                    mode="lines",
                                    name=log,
                                    connectgaps=False,
                                    line=dict(width=1.5),
                                    hovertemplate=(
                                        log + ": %{x}<br>"
                                        "Depth: %{y}"
                                        "<extra></extra>"
                                    )
                                ),
                                row=1,
                                col=i
                            )

                            fig.update_xaxes(
                                title_text=log,
                                row=1,
                                col=i
                            )

                        fig.update_yaxes(
                            title_text="Depth",
                            autorange="reversed",
                            row=1,
                            col=1
                        )

                        fig.update_layout(
                            title="Multi-Track Log Comparison",
                            height=700,
                            showlegend=False
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    # Tabel indikator dan data pendukung
                    st.subheader("5. Tabel interpretasi awal")

                    table_cols = [depth_col] + available_logs

                    if "GR_INDICATOR" in plot_df.columns:
                        table_cols.append("GR_INDICATOR")

                    st.dataframe(
                        plot_df[table_cols],
                        use_container_width=True
                    )

                    if "GR_INDICATOR" in plot_df.columns:
                        st.caption(
                            "GR rendah, menengah, dan tinggi adalah "
                            "kategori respons GR, bukan identifikasi "
                            "litologi definitif. RHOB, NPHI, dan RT "
                            "ditampilkan sebagai data pendukung "
                            "untuk interpretasi terpadu."
                        )

                    
                    # Ekspor hasil
                    csv_data = plot_df[
                        table_cols
                    ].to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Unduh Data Interpretasi CSV",
                        data=csv_data,
                        file_name="multi_log_interpretation.csv",
                        mime="text/csv"
                    )

        st.divider()
        show_lithology()

    elif menu == "Facies":
        show_facies(df)

    elif menu == "Petrophysics":
        show_petrophysics(df)

    elif menu == "Reservoir Evaluation":
        show_reservoir()

    elif menu == "Machine Learning":
        show_ml(df)

    elif menu == "Report Export":
        show_report(df)
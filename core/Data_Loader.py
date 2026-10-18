
import io
from pathlib import Path

import lasio
import pandas as pd


def load_well_data(uploaded_file):
    """
    Membaca data sumur dari file LAS, CSV, atau Excel.
    Mengembalikan data dalam bentuk pandas DataFrame.
    """
    filename = Path(uploaded_file.name).name.lower()
    file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        raise ValueError("File yang diunggah kosong.")

    if filename.endswith(".las"):
        text_data = file_bytes.decode("utf-8-sig", errors="replace")
        las = lasio.read(io.StringIO(text_data))
        df = las.df().reset_index()

        # Mengubah nama kolom kedalaman menjadi DEPTH
        if len(df.columns) > 0:
            df.rename(columns={df.columns[0]: "DEPTH"}, inplace=True)

    elif filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))

    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes))

    else:
        raise ValueError(
            "Format tidak didukung. Gunakan file LAS, CSV, atau Excel."
        )

    if df.empty:
        raise ValueError("File tidak memiliki data.")

    # Merapikan nama kolom
    df.columns = [str(col).strip().upper() for col in df.columns]

    # Memeriksa apakah ada nama kolom yang kosong
    if any(not col for col in df.columns):
        raise ValueError("File memiliki nama kolom yang kosong.")

    # Menghapus baris yang seluruh nilainya kosong
    df = df.dropna(how="all").reset_index(drop=True)

    if df.empty:
        raise ValueError("Tidak ada baris data yang dapat digunakan.")

    return df

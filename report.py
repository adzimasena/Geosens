"""Penyusun laporan HTML mandiri dari tabel, dict, dan teks."""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

STYLE = """
body{font-family:Arial,Helvetica,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#222}
h1{border-bottom:2px solid #444;padding-bottom:.3rem}h2{margin-top:2rem;color:#234}
table{border-collapse:collapse;margin:.6rem 0;font-size:.9rem}
th,td{border:1px solid #bbb;padding:4px 8px;text-align:right}th{background:#eee}
td:first-child,th:first-child{text-align:left}.note{color:#555;font-size:.85rem}
"""


def _fmt(v):
    if isinstance(v, float):
        return "" if pd.isna(v) else f"{v:.4g}"
    return "" if v is None else str(v)


def table_html(df: pd.DataFrame, max_rows: int = 200) -> str:
    shown = df.head(max_rows)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in shown.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_fmt(v))}</td>" for v in row) + "</tr>"
        for row in shown.itertuples(index=False, name=None)
    )
    note = ""
    if len(df) > max_rows:
        note = f'<p class="note">Ditampilkan {max_rows} dari {len(df)} baris.</p>'
    return f"<table><tr>{head}</tr>{body}</table>{note}"


def section_html(heading: str, content) -> str:
    if isinstance(content, pd.DataFrame):
        inner = table_html(content)
    elif isinstance(content, dict):
        inner = table_html(pd.DataFrame({"Parameter": list(content.keys()),
                                         "Nilai": [str(v) for v in content.values()]}))
    else:
        inner = f"<p>{html.escape(str(content))}</p>"
    return f"<h2>{html.escape(heading)}</h2>{inner}"


def build_html_report(title: str, sections: list, disclaimer: str = "",
                      timestamp: str | None = None) -> str:
    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M")
    body = "".join(section_html(h, c) for h, c in sections)
    disc = f'<p class="note">{html.escape(disclaimer)}</p>' if disclaimer else ""
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title><style>{STYLE}</style></head><body>"
            f"<h1>{html.escape(title)}</h1><p class='note'>Dibuat: {html.escape(ts)}</p>"
            f"{body}{disc}</body></html>")

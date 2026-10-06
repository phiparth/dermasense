"""
Citations: every source the model and the site use, with its link.
The list lives in references.py; this page only renders it.
"""

from __future__ import annotations

import html

import streamlit as st

import references as rf
import ui

ui.eyebrow("citations")
st.title("Citations")

st.markdown(
    """
    <style>
      .cite { width: 100%; border-collapse: collapse; font-size: 0.9rem;
              background: #ffffff; border: 1px solid #e3ded4; border-radius: 10px;
              overflow: hidden; }
      .cite th { text-align: left; font-size: 0.7rem; text-transform: uppercase;
                 letter-spacing: 0.08em; color: #6c757d; font-weight: 600;
                 padding: 0.6rem 0.9rem; border-bottom: 1px solid #e3ded4; }
      .cite td { padding: 0.7rem 0.9rem; border-bottom: 1px solid #efebe3;
                 vertical-align: top; line-height: 1.45; }
      .cite tr:last-child td { border-bottom: 0; }
      .cite .who { font-weight: 600; white-space: nowrap; }
      .cite .where { color: #6c757d; font-size: 0.8rem; }
      .cite a { white-space: nowrap; }
      .cite-wrap { overflow-x: auto; margin-bottom: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def esc(t: str) -> str:
    return html.escape(t, quote=False)


for heading, _, refs in rf.SECTIONS:
    st.subheader(heading)
    rows = []
    for r in refs:
        title = esc(r.title)
        if r.title and r.where:
            title += f"<div class='where'>{esc(r.where)}</div>"
        links = "<br>".join(
            f"<a href='{u}' target='_blank' rel='noopener'>{esc(lab)}</a>"
            for lab, u in r.links
        )
        rows.append(
            f"<tr><td class='who'>{esc(r.cited_as)}</td><td>{title}</td>"
            f"<td>{esc(r.sets)}</td><td>{links}</td></tr>"
        )
    st.markdown(
        "<div class='cite-wrap'><table class='cite'><thead><tr>"
        "<th>Citation</th><th>Title</th><th>Used for</th><th>Link</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>",
        unsafe_allow_html=True,
    )

st.caption("Live air quality: Open-Meteo, which is CAMS underneath.")

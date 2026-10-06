"""
Citations.

Every source the model leans on, with its link and how far verification got.
The list lives in references.py; this page only renders it.
"""

from __future__ import annotations

import streamlit as st

import references as rf
import ui
from ui import INK, MUTED, RUST, SIGNAL

ui.eyebrow("citations")
st.title("Where every number comes from")
st.markdown(
    f"<span style='color:{MUTED}'>Each constant in the model, the source that fixes "
    "it, and a link. Four of them have no source anybody can find, and those are "
    "listed too.</span>",
    unsafe_allow_html=True,
)

c = rf.counts()
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(ui.readout("Read in full", str(c["read"]),
                           "the full text was opened and the numbers read off it"),
                unsafe_allow_html=True)
with m2:
    st.markdown(ui.readout("Located", str(c["matched"]),
                           "details match the claim, full text behind a block"),
                unsafe_allow_html=True)
with m3:
    st.markdown(ui.readout("Not identified", str(c["missing"]),
                           "in the model with no traceable source"),
                unsafe_allow_html=True)

ui.flag(
    "<b>Only one of these was opened and read end to end.</b> For the rest the "
    "citation was located and its reported details line up with what the model "
    "claims, but the full text sits behind a publisher block, so the numbers have "
    "not been checked against the paper itself. That distinction is marked on every "
    "entry rather than left for a reader to assume."
)

TONE = {"read": SIGNAL, "matched": "#8a7b52", "missing": RUST}

for title, blurb, refs in rf.SECTIONS:
    st.divider()
    st.subheader(title)
    st.markdown(f"<span style='color:{MUTED}'>{blurb}</span>", unsafe_allow_html=True)
    st.write("")

    for r in refs:
        tone = TONE[r.checked]
        links = " &nbsp;·&nbsp; ".join(
            f"<a href='{url}' target='_blank' rel='noopener'>{label}</a>"
            for label, url in r.links
        )
        st.markdown(
            f"""
            <div class="card" style="border-left:3px solid {tone};margin-bottom:0.7rem">
              <div style="display:flex;justify-content:space-between;gap:1rem;
                          flex-wrap:wrap;align-items:baseline">
                <div style="font-weight:700;color:{INK}">{r.cited_as}</div>
                <div style="font-size:0.7rem;text-transform:uppercase;
                            letter-spacing:0.08em;color:{tone};font-weight:600">
                  {rf.CHECK_LABEL[r.checked]}</div>
              </div>
              <div style="font-size:0.92rem;color:#3b4441;line-height:1.5;
                          margin-top:0.25rem">{r.title}</div>
              <div style="font-size:0.78rem;color:{MUTED};margin-top:0.2rem">
                {r.where}</div>
              <div style="font-size:0.85rem;color:{INK};margin-top:0.55rem">
                <b>Sets:</b> {r.sets}</div>
              {f'<div style="font-size:0.85rem;color:{MUTED};margin-top:0.4rem">{r.note}</div>' if r.note else ''}
              {f'<div style="font-size:0.85rem;margin-top:0.55rem">{links}</div>' if links else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

ui.note(
    "<b>Two things worth acting on.</b> The four unidentified citations set "
    "<code>k_H</code>, <code>eta_H</code>, <code>A0</code> and <code>alpha_O3</code>; "
    "whoever added them to the specification should supply the DOIs. And the "
    "xylolipid scavenging figure the model uses, 70% at 1 mg/mL, does not match the "
    "60.76% the source's abstract reports. If 60.76% is right, <code>k_X</code> is "
    "overstated, and it is one of the constants the film-side result leans on."
)

st.caption(
    "iGEM IIT Delhi 2026 - DermaSense. Live air quality comes from Open-Meteo, "
    "which is CAMS underneath. Model outputs are predictions from a parameterised "
    "cascade, not clinical guidance."
)

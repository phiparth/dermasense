"""
Lab vs market.

What a commercial moisturiser or serum actually carries, in the units the model
uses, next to what the model recommends. The headline is a units problem: a
cosmetic label says % w/w, the model says mg/mL, and the factor of ten between
them is most of the story.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dermasense_model as dm
import ui
from dermasense_model import CAPS, Dose, Environment, NOMINAL, benchmark_params
from ui import COLOR, INK, MUTED, RUST, SIGNAL

ui.eyebrow("the market")
st.title("What shop shelves carry, and what the model recommends")
st.markdown(
    f"<span style='color:{MUTED}'>A cosmetic label gives % w/w. The model works in "
    "mg/mL. At a density of about 1 g/mL they differ by a factor of ten, and that "
    "factor turns out to be the whole comparison: the market doses an order of "
    "magnitude above where our evidence stops.</span>",
    unsafe_allow_html=True,
)

choice = ui.benchmark_picker("mkt", label="Which surfactant the market product uses")
B = dm.BENCHMARKS[choice]
params = benchmark_params(choice)
env = ui.shared_env()

lab_dose, lab_G = ui.c_optimal(ui.ekey(env), ui.pkey(params), dm.DOSE_ORDER)
lab_dose = Dose(*lab_dose)
lab_ratio = dm.formulation_ratio(lab_dose)
rows = dm.market_scenarios(env, params)
by = {r["level"]: r for r in rows}

# ----------------------------------------------------------------------------
# 1. the unit gap
# ----------------------------------------------------------------------------

st.subheader("First, the units")

u1, u2, u3 = st.columns(3)
with u1:
    st.markdown(
        f"""<div class="card" style="border-top:3px solid {COLOR['H']}">
        <div class="readout-label">A 1 % hyaluronic acid serum</div>
        <div class="readout">10 mg/mL</div>
        <div class="readout-sub">the model's ceiling for it is
        {CAPS['H']:g} mg/mL, so that bottle is <b>10x</b> our cap</div></div>""",
        unsafe_allow_html=True,
    )
with u2:
    st.markdown(
        f"""<div class="card" style="border-top:3px solid {COLOR['S']}">
        <div class="readout-label">A 5 % surfactant leave-on</div>
        <div class="readout">50 mg/mL</div>
        <div class="readout-sub">the benchmark's cap here is
        {params.S_cap:g} mg/mL, so that product is <b>50x</b> it</div></div>""",
        unsafe_allow_html=True,
    )
with u3:
    st.markdown(
        f"""<div class="card" style="border-top:3px solid {SIGNAL}">
        <div class="readout-label">What we recommend, in total</div>
        <div class="readout">{lab_ratio['total']:.2f} mg/mL</div>
        <div class="readout-sub">about
        {lab_ratio['total'] / dm.PCT_TO_MG_PER_ML:.2f} % w/w, all four compounds
        together</div></div>""",
        unsafe_allow_html=True,
    )

ui.flag(
    "<b>This is the single most important sentence on the page.</b> Our entire "
    f"recommended formulation, {lab_ratio['total']:.2f} mg/mL, is about "
    f"{lab_ratio['total'] / dm.PCT_TO_MG_PER_ML:.2f} % by weight. A plain "
    "supermarket moisturiser carries several percent of surfactant alone. We are "
    "not recommending a competitive product; we are recommending a dose at the "
    "bottom edge of what the industry considers a working concentration, because "
    "that is where our evidence stops."
)

st.divider()

# ----------------------------------------------------------------------------
# 2. the published ranges
# ----------------------------------------------------------------------------

st.subheader("What the literature says products actually contain")

m1, m2 = st.columns(2)
for col, key in zip((m1, m2), ("H", "S")):
    mr = dm.MARKET_RANGES[key]
    cap = dm.cap_of(key, params)
    with col:
        st.markdown(
            f"""<div class="card" style="border-top:3px solid {COLOR[key]}">
            <div style="font-weight:700">{mr.label}</div>
            <div style="font-family:monospace;font-size:1.25rem;margin:.45rem 0">
              {mr.low:g} % &nbsp;&ndash;&nbsp; {mr.high:g} %
            </div>
            <div style="font-size:.78rem;color:{MUTED}">
              {mr.low * dm.PCT_TO_MG_PER_ML:g} to
              {mr.high * dm.PCT_TO_MG_PER_ML:g} mg/mL &nbsp;|&nbsp;
              our cap {cap:g} mg/mL</div>
            <div style="font-size:.86rem;color:#3b4441;line-height:1.5;margin-top:.55rem">
              {mr.basis}</div>
            </div>""",
            unsafe_allow_html=True,
        )

st.write("")
rf = go.Figure()
for i, key in enumerate(("H", "S")):
    mr = dm.MARKET_RANGES[key]
    cap = dm.cap_of(key, params)
    rf.add_scatter(
        x=[mr.low * dm.PCT_TO_MG_PER_ML, mr.high * dm.PCT_TO_MG_PER_ML],
        y=[i, i], mode="lines",
        line=dict(color=COLOR[key], width=13), opacity=.35,
        name=f"{key}: market range", showlegend=False,
        hovertemplate=f"{mr.label}<br>market %{{x:.2f}} mg/mL<extra></extra>",
    )
    rf.add_scatter(
        x=[mr.typical * dm.PCT_TO_MG_PER_ML], y=[i], mode="markers",
        marker=dict(color=COLOR[key], size=15, symbol="diamond"),
        showlegend=False,
        hovertemplate=f"typical {mr.typical:.2f} %<extra></extra>",
    )
    rf.add_scatter(
        x=[cap], y=[i], mode="markers",
        marker=dict(color=INK, size=15, symbol="line-ns-open",
                    line=dict(width=3, color=INK)),
        showlegend=False,
        hovertemplate=f"our cap {cap:g} mg/mL<extra></extra>",
    )
rf.add_scatter(x=[lab_dose.H], y=[0], mode="markers",
               marker=dict(color=SIGNAL, size=13, symbol="star"),
               showlegend=False, hovertemplate="we recommend %{x:.3f} mg/mL<extra></extra>")
rf.add_scatter(x=[lab_dose.X + lab_dose.L], y=[1], mode="markers",
               marker=dict(color=SIGNAL, size=13, symbol="star"),
               showlegend=False,
               hovertemplate="our film formers, %{x:.3f} mg/mL total<extra></extra>")
ui.style(rf, 260,
         xaxis=dict(title="concentration, mg/mL (log scale)", type="log"),
         yaxis=dict(tickmode="array", tickvals=[0, 1],
                    ticktext=["Antioxidant /<br>humectant", "Surfactant /<br>film"],
                    range=[-0.6, 1.6]))
st.plotly_chart(rf, width="stretch", key="mkt_ranges", config={"displayModeBar": False})
st.caption(
    "Bars are the published market ranges, diamonds the typical level, the vertical "
    "rule is our evidence cap and the star is what the model recommends. Both stars "
    "sit at or below the bottom of the market range. Note the log scale: the gap is "
    "wider than it looks."
)

ui.note(
    "<b>'Typical' is our word, not a source's.</b> The literature reports how low "
    "and how high products go; where the middle sits is a judgement. It is taken "
    "here as the geometric mean of the two ends, which on a log scale is the "
    "midpoint, and it is marked as a choice rather than quoted as a measurement."
)

st.divider()

# ----------------------------------------------------------------------------
# 3. scoring a market product
# ----------------------------------------------------------------------------

st.subheader("Running a market-level product through the model")
st.markdown(
    f"<span style='color:{MUTED}'>Hyaluronic acid plus {B.name.lower()} at the low, "
    "typical and high ends of their published ranges. Two scores for each: the model "
    "run as written, and the model run with every dose held at its cap. They "
    "disagree, and the disagreement is the point.</span>",
    unsafe_allow_html=True,
)

tbl = pd.DataFrame([{
    "Level": r["level"].title(),
    "Antioxidant": f"{r['pct_H']:.2f} % ({r['H']:.1f} mg/mL)",
    "Surfactant": f"{r['pct_S']:.2f} % ({r['S']:.1f} mg/mL)",
    "Past our caps by": f"{r['factor']:.0f}x",
    "Where that is": r["domain"],
    "G, as written": r["G_raw"] * 100,
    "G, held at caps": r["G_clipped"] * 100,
} for r in rows])
st.dataframe(
    tbl, hide_index=True, width="stretch",
    column_config={
        "G, as written": st.column_config.NumberColumn(format="%.1f%%"),
        "G, held at caps": st.column_config.NumberColumn(format="%.1f%%"),
    },
)

c1, c2 = st.columns([1.3, 1])
with c1:
    bf = go.Figure()
    labels = [r["level"].title() for r in rows] + ["DermaSense<br>recommendation"]
    raw = [r["G_raw"] * 100 for r in rows] + [lab_G * 100]
    clip = [r["G_clipped"] * 100 for r in rows] + [lab_G * 100]
    bf.add_bar(x=labels, y=raw, name="model as written",
               marker_color=[RUST if r["factor"] > 1 else "#c9c2b4" for r in rows] + [SIGNAL])
    bf.add_bar(x=labels, y=clip, name="held at our caps",
               marker_color=[INK] * len(rows) + [SIGNAL], opacity=.85)
    bf.add_hline(y=0, line=dict(color=MUTED, width=1))
    ui.style(bf, 380, barmode="group",
             yaxis=dict(title="protection G, %"),
             legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(bf, width="stretch", key="mkt_bars",
                    config={"displayModeBar": False})

with c2:
    high = by["high"]
    st.markdown(
        f"""
**Read the two bars together.**

At the **low** end, {by['low']['H']:.1f} and {by['low']['S']:.1f} mg/mL, the product
sits inside our validity domain and the two scores agree exactly: G =
**{by['low']['G_raw'] * 100:.1f}%**. Nothing is being extrapolated.

At the **high** end, {high['H']:.0f} and {high['S']:.0f} mg/mL, the model as
written returns **{high['G_raw'] * 100:.0f}%** &mdash; a catastrophe, with the
stripping term at Psi = {high['Psi']:.1f} and water loss several times worse
than doing nothing.

That number is not a prediction. The stripping term is **linear and unbounded**
above the threshold, fitted nowhere near 50 mg/mL, so it runs away. Real 5 %
surfactant products do not destroy skin.
        """
    )

ui.flag(
    "<b>Where the model stops being evidence.</b> Our caps are the edge of the "
    "literature we read, not the edge of what is safe or sold. Past about ten times "
    "a cap the curves are arithmetic rather than knowledge, which is why this page "
    "shows both columns and labels the domain on every row. If a judge asks what "
    "the model predicts for a real serum, the honest answer is: at the low end, "
    f"{by['low']['G_raw'] * 100:.0f}%. Above that, we are extrapolating and say so."
)

st.divider()

# ----------------------------------------------------------------------------
# 4. the ratio
# ----------------------------------------------------------------------------

st.subheader("The ratio: antioxidant against film former")

r1, r2 = st.columns([1, 1.4])

with r1:
    st.markdown(
        f"""
Counting the antioxidants (pulcherrimin, hyaluronic acid) against the film
formers (xylolipid, lyso-ornithine lipid, the benchmark):

| | antioxidant : film |
|---|---|
| **DermaSense** | **{lab_ratio['ratio']:.1f} : 1** |
| Market, low | {by['low']['ratio_antiox_to_surf']:.1f} : 1 |
| Market, typical | {by['typical']['ratio_antiox_to_surf']:.1f} : 1 |
| Market, high | {by['high']['ratio_antiox_to_surf']:.1f} : 1 |

We land **antioxidant-heavy**, and the market runs the other way. The reason is
structural, not a discovery: commercial surfactant is there to emulsify the
product, so its level is set by formulation chemistry, not by skin protection.
Our film formers are there only to protect, and they saturate the interface at
a fraction of a milligram.
        """
    )
    ui.note(
        f"<b>{lab_ratio['antioxidant']:.2f} mg/mL antioxidant to "
        f"{lab_ratio['surfactant']:.2f} mg/mL film former.</b> Almost all of the "
        "antioxidant mass is hyaluronic acid at its cap; pulcherrimin contributes "
        f"{lab_dose.P:.3f} mg/mL, and it is the only coordinate in the whole "
        "formulation with a real interior optimum."
    )

with r2:
    masses = np.logspace(-2, 1.4, 90)
    sf = go.Figure()
    shapes = [
        ("DermaSense proportions", lab_dose, SIGNAL, "solid"),
        ("Market proportions", Dose(H=dm.MARKET_RANGES["H"].typical,
                                    S=dm.MARKET_RANGES["S"].typical), COLOR["S"], "dot"),
    ]
    for name, shape, tone, dash in shapes:
        ys = dm.ratio_sweep(env, params, shape, masses) * 100
        sf.add_scatter(x=masses, y=ys, name=name,
                       line=dict(color=tone, width=2.6, dash=dash),
                       hovertemplate=f"{name}<br>%{{x:.2f}} mg/mL total"
                                     "<br>G %{y:.1f}%<extra></extra>")
    sf.add_vline(x=lab_ratio["total"], line=dict(color=SIGNAL, dash="dash", width=1.5),
                 annotation_text="what we recommend",
                 annotation_font=dict(size=10, color=SIGNAL))
    sf.add_vrect(x0=dm.MARKET_RANGES["S"].low * dm.PCT_TO_MG_PER_ML,
                 x1=dm.MARKET_RANGES["S"].high * dm.PCT_TO_MG_PER_ML,
                 fillcolor=RUST, opacity=.07, line_width=0,
                 annotation_text="market surfactant range",
                 annotation_font=dict(size=10, color=RUST))
    ui.style(sf, 400,
             xaxis=dict(title="total applied mass, mg/mL (log scale)", type="log"),
             yaxis=dict(title="protection G, %", range=[-30, 70]),
             legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(sf, width="stretch", key="mkt_ratio",
                    config={"displayModeBar": False})
    st.caption(
        "The same two mixtures scaled from a trace to a lot, holding each one's "
        "proportions fixed. Both rise, peak and fall. Ours peaks earlier and "
        "higher, because the film saturates at a low dose and the extra mass in "
        "the market mixture is surfactant that has nothing left to cover."
    )

st.divider()

# ----------------------------------------------------------------------------
# 5. the antioxidants the model cannot score
# ----------------------------------------------------------------------------

st.subheader("The antioxidants this model cannot score at all")

a1, a2 = st.columns([1.1, 1])
with a1:
    st.markdown(
        """
The serums people actually buy for pollution are not hyaluronic acid. They are
vitamin C formulations, and they are dosed far above anything in this model:
        """
    )
    st.dataframe(
        pd.DataFrame([{
            "Ingredient": a["name"],
            "Typical range": (f"{a['low']:g} %" if a["low"] == a["high"]
                              else f"{a['low']:g} - {a['high']:g} %"),
            "In mg/mL": (f"{a['low'] * 10:g}" if a["low"] == a["high"]
                         else f"{a['low'] * 10:g} - {a['high'] * 10:g}"),
            "Example": a["example"],
        } for a in dm.UNSCORED_ANTIOXIDANTS]),
        hide_index=True, width="stretch",
    )
with a2:
    ui.flag(
        "<b>We cannot put these on the chart, and should not pretend otherwise.</b> "
        "The model has one scavenging channel, and it needs a rate constant measured "
        "the same way for every compound in it. Ours come from DPPH assays on "
        "hyaluronic acid and xylolipid. Nobody has run that assay, in those "
        "conditions, for ascorbic acid at 15 %. Dropping a literature number in "
        "would produce a bar that looks comparable and is not."
    )
    ui.note(
        "<b>The useful comparison is scale, and it is unflattering.</b> A vitamin C "
        "serum carries 100 to 200 mg/mL of a single antioxidant. Our whole "
        f"formulation is {lab_ratio['total']:.2f} mg/mL. Whatever a living "
        "moisturiser is for, out-dosing a vitamin C serum is not it: the case has "
        "to be that it responds to the day, not that it delivers more."
    )

st.divider()

# ----------------------------------------------------------------------------
# 6. honest summary
# ----------------------------------------------------------------------------

st.subheader("What this comparison does and does not establish")

h1, h2, h3 = st.columns(3)
with h1:
    ui.note(
        "<b>Established: our dose is small.</b> The recommendation is about "
        f"{lab_ratio['total'] / dm.PCT_TO_MG_PER_ML:.2f} % by weight, at or below "
        "the bottom of every published commercial range. Whatever else is "
        "uncertain, we are not proposing to put a lot of material on anybody."
    )
with h2:
    ui.flag(
        "<b>Not established: that we beat a market product.</b> The low-end "
        "comparison is a like-for-like the model can support. The typical and high "
        "ones are extrapolations, and the headline antioxidants of the category "
        "cannot enter the model at all."
    )
with h3:
    ui.flag(
        "<b>Not established: that more would be worse.</b> The high-end collapse "
        "is the stripping term running away outside its fitted range, not a "
        "prediction about real products. Fixing it needs a saturating stripping "
        "term and a measurement to fit it to."
    )

st.caption(
    "iGEM IIT Delhi 2026 - DermaSense. Market concentration ranges are from "
    "published patents, the Cosmetic Ingredient Review and manufacturers' own "
    "stated formulations; every one is cited on this page. Model outputs are "
    "predictions from a parameterised cascade, not clinical guidance."
)

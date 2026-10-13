"""
Landing page: what DermaSense is, what it is made of, and a live taste of the
model before anyone opens the full simulator.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import dermasense_model as dm
import ui
from dermasense_model import Dose, Environment
from ui import COLOR, INK, MUTED, RUST, SIGNAL

# ----------------------------------------------------------------------------
# hero
# ----------------------------------------------------------------------------

st.markdown(
    """
    <style>
      .ds-hero {
        border-radius: 18px; border: 1px solid #e3ded4; padding: 46px 44px 40px;
        background: linear-gradient(135deg, #fbfaf6 0%, #eef3f1 55%, #f6e7dc 100%);
      }
      .ds-kicker { font-size: .74rem; letter-spacing: .16em; text-transform: uppercase;
        color: #0f6f68; font-weight: 700; }
      .ds-title { font-size: 3.2rem; line-height: 1.04; font-weight: 800; color: #1c2321;
        margin: .35rem 0 .7rem; letter-spacing: -.02em; }
      .ds-title b { background: linear-gradient(90deg,#0f6f68,#3d5a80 55%,#b23a2a);
        -webkit-background-clip: text; background-clip: text; color: transparent;
        font-weight: 800; }
      .ds-lede { font-size: 1.07rem; color: #3b4441; line-height: 1.6; max-width: 640px; }
      .ds-badge { display:inline-block; background:rgba(255,255,255,.8);
        border:1px solid #e3ded4; border-radius:999px; padding:.3rem .8rem;
        font-size:.8rem; margin:.9rem .4rem 0 0; color:#1c2321; }
      @media (max-width: 760px) { .ds-title { font-size: 2.3rem; } .ds-hero { padding: 26px 20px; } }
    </style>
    <div class="ds-hero">
      <div class="ds-kicker">iGEM IIT Delhi 2026 &middot; modelling</div>
      <div class="ds-title">Derma<b>Sense</b></div>
      <div class="ds-lede">
        A living moisturiser that reads the air. An engineered skin microbe senses the
        day's oxidative stress and answers with a red antioxidant pigment and two
        film-forming biosurfactants. This site is the model that decides how much of
        each, and says plainly how sure it is of every number.
      </div>
      <div>
        <span class="ds-badge">Pulcherrimin &middot; antioxidant</span>
        <span class="ds-badge">Xylolipid &middot; biosurfactant</span>
        <span class="ds-badge">Lyso-ornithine lipid &middot; biosurfactant</span>
        <span class="ds-badge">+ 2 literature controls</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ----------------------------------------------------------------------------
# headline numbers, computed rather than typed
# ----------------------------------------------------------------------------

REF = Environment()
_ref_dose_k, G_ref = ui.c_optimal(ui.ekey(REF), ui.pkey(dm.NOMINAL))
ref_res = ui.c_evaluate(ui.ekey(REF), _ref_dose_k, ui.pkey(dm.NOMINAL))

h1, h2, h3, h4 = st.columns(4)
with h1:
    st.markdown(ui.readout("Protection on a reference day", f"{G_ref * 100:.1f}%",
                           "share of untreated damage removed"), unsafe_allow_html=True)
with h2:
    st.markdown(ui.readout("Damage, RSD",
                           f"{ref_res['RSD0']:.2f} → {ref_res['RSD']:.2f}",
                           "reference day is 1.00 by construction"),
                unsafe_allow_html=True)
with h3:
    st.markdown(ui.readout("Water loss, TEWL",
                           f"{ref_res['TEWL_improvement'] * 100:.1f}%",
                           f"{ref_res['T']:.1f} → {ref_res['T_prime']:.1f} g/m2/h"),
                unsafe_allow_html=True)
with h4:
    n_c = sum(1 for v in dm.EVIDENCE.values() if v == "C")
    st.markdown(ui.readout("Constants nobody measured", f"{n_c} of {len(dm.EVIDENCE)}",
                           "class C: we chose the value, and say so"),
                unsafe_allow_html=True)

st.caption(
    "Every number on this page is computed live by the model in this repository, "
    "not typed in. Reference day: PM2.5 80 ug/m3, ozone 60 ug/m3, UV index 8, "
    "phototype I, healthy barrier."
)

st.divider()

# ----------------------------------------------------------------------------
# the three compounds and their controls
# ----------------------------------------------------------------------------

st.subheader("Three compounds from the lab, two controls from the literature")
st.markdown(
    f"<span style='color:{MUTED}'>A protective effect is only worth reporting next to "
    "something already known to work. Pulcherrimin is benchmarked against hyaluronic "
    "acid; the two biosurfactants are benchmarked against a glycolipid that is already "
    "sold for skin. All five run through identical equations.</span>",
    unsafe_allow_html=True,
)
st.write("")
ui.legend()
st.write("")

CARDS = [
    ("P", "Pulcherrimin", "lab", "antioxidant",
     "An iron-binding red pigment. Screens UV, and locks up the iron that particulates "
     "carry so it cannot drive Fenton chemistry. The only compound with an interior "
     "optimum: past about 0.4 mg/mL it starts harming the cells it protects."),
    ("X", "Xylolipid", "lab", "biosurfactant",
     "Spreads at the skin-air interface and intercepts particles before they land. "
     "Also scavenges radicals directly. Its evidence base is a fruit-juice "
     "preservation study, so treat its claim with care."),
    ("L", "Lyso-ornithine lipid", "lab", "biosurfactant",
     "The second film former, effective at a far lower concentration than the first. "
     "Its ceiling here is what the strain can secrete, 10 mg/L, not a measured "
     "efficacy limit."),
    ("H", "Hyaluronic acid", "control", "antioxidant + humectant",
     "The comparator everyone already knows. Holds water in the stratum corneum and "
     "mops up radicals. It is in the formulation and in the control column at once: "
     "it is what a buyer would otherwise be putting on."),
    ("S", ui.benchmark().name, "control", "biosurfactant",
     "The control we added for the film side, run through the same shielding, "
     "stripping and self-toxicity terms as ours."),
]

cols = st.columns(len(CARDS))
for col, (k, name, kind, mech, body) in zip(cols, CARDS):
    tone = COLOR[k]
    tag_bg = "#eef5f3" if kind == "lab" else "#f5f0f8"
    edge = (f"border-top:3px solid {tone}" if kind == "lab"
            else f"border:1.5px dashed {tone};border-top:3px dashed {tone}")
    with col:
        st.markdown(
            f"""
            <div class="card" style="{edge}">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:.4rem">
                <div style="font-weight:700;font-size:1.02rem;color:{INK}">{name}</div>
                {ui.chip(k)}
              </div>
              <div style="margin:.35rem 0 .55rem">
                <span style="background:{tag_bg};border-radius:999px;padding:.12rem .55rem;
                  font-size:.7rem;font-weight:600;color:{tone}">{kind}</span>
                <span style="font-size:.72rem;color:{MUTED};margin-left:.3rem">{mech}</span>
              </div>
              <div style="font-size:.86rem;color:#3b4441;line-height:1.5">{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ----------------------------------------------------------------------------
# live mini simulator
# ----------------------------------------------------------------------------

st.subheader("Try it on a day")
st.markdown(
    f"<span style='color:{MUTED}'>Three numbers describe the air. Move them and watch "
    "the dose and the protection move with them. The full version, with live air "
    "quality for your city and a skin questionnaire, is on the "
    "<b>Dose simulator</b> page.</span>",
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.5])

with left:
    preset = st.selectbox("Start from a day", list(ui.DAY_PRESETS), index=0,
                          key="home_preset")
    base = ui.DAY_PRESETS[preset]
    pm = st.slider("PM2.5, ug/m3", 0, 400, int(base["C_PM"]), 5, key=f"home_pm_{preset}")
    o3 = st.slider("Ozone, ug/m3", 0, 300, int(base["C_O3"]), 5, key=f"home_o3_{preset}")
    uv = st.slider("UV dose, normalised", 0.0, 2.5, float(base["I_UV"]), 0.05,
                   key=f"home_uv_{preset}",
                   help="1.0 is UV index 8, the bottom of the WHO 'very high' band.")
    pt = st.select_slider("Fitzpatrick phototype", options=[1, 2, 3, 4, 5, 6], value=1,
                          format_func=lambda i: "I II III IV V VI".split()[i - 1],
                          key="home_pt")
    barrier = st.radio("Barrier", ["Healthy", "Compromised"], horizontal=True,
                       key="home_barrier")
    bench = ui.benchmark_picker(
        "home", label="Control surfactant to compare against",
        help="Hyaluronic acid is the control for the antioxidant side. This is "
             "the control for the film side. Both are run through the same "
             "equations as our own compounds.",
    )

e = Environment(C_PM=float(pm), C_O3=float(o3), I_UV=float(uv), PT=int(pt),
                A0=dm.A0_HEALTHY if barrier == "Healthy" else dm.A0_COMPROMISED)
params = dm.benchmark_params(bench)
B = dm.BENCHMARKS[bench]

# Optimised over all five, so both controls appear in the dose table: the
# antioxidant control (hyaluronic acid) and the film control side by side with
# our own three.
dose_k, G = ui.c_optimal(ui.ekey(e), ui.pkey(params), dm.ALL_KEYS)
dose = Dose(*dose_k)
res = ui.c_evaluate(ui.ekey(e), dose_k, ui.pkey(params))

# The three arms, each optimised on its own, so every bar is the best that arm
# can do rather than an arbitrary dose.
ARMS = [
    ("Our compounds<br>P + X + L", dm.LAB_KEYS, ui.LAB_GROUP),
    (f"Both controls<br>H + S ({B.name})", dm.CONTROL_KEYS, ui.CONTROL_GROUP),
    ("All five<br>together", dm.ALL_KEYS, ui.ALL_GROUP),
]
arm_G = {}
for label, keys, _ in ARMS:
    arm_G[label] = ui.c_optimal(ui.ekey(e), ui.pkey(params), keys)[1]

with right:
    g1, g2 = st.columns([1, 1.3])
    with g1:
        st.plotly_chart(ui.gauge_figure(G), width="stretch",
                        key="home_gauge", config={"displayModeBar": False})
    with g2:
        bar = go.Figure()
        keys = dm.ALL_KEYS
        bar.add_bar(
            x=[getattr(dose, k) / dm.cap_of(k, params) * 100 for k in keys],
            y=[ui.compound_label(k) for k in keys],
            orientation="h",
            # the shared rule: lab solid, controls hatched
            marker=ui.bar_marker(keys),
            text=[f"{getattr(dose, k):.3f} mg/mL" for k in keys],
            textposition="auto",
            hovertemplate="%{y}<br>%{x:.0f}% of its cap<extra></extra>",
        )
        ui.style(bar, 300, showlegend=False,
                 xaxis=dict(title="% of the evidence cap", range=[0, 118]),
                 margin=dict(l=0, r=0, t=26, b=0))
        st.plotly_chart(bar, width="stretch", key="home_dose",
                        config={"displayModeBar": False})

    a1, a2 = st.columns([1, 1.25])
    with a1:
        af = go.Figure()
        af.add_bar(
            x=[arm_G[l] * 100 for l, _, _ in ARMS],
            y=[l for l, _, _ in ARMS],
            orientation="h",
            marker=dict(color=[c for _, _, c in ARMS],
                        pattern=dict(shape=["", "/", ""], fgcolor="white", size=5)),
            text=[f"{arm_G[l] * 100:.1f}%" for l, _, _ in ARMS],
            textposition="auto",
            hovertemplate="%{y}<br>G = %{x:.1f}%<extra></extra>",
        )
        ui.style(af, 300, showlegend=False,
                 xaxis=dict(title="protection G, %", range=[0, 72]),
                 yaxis=dict(autorange="reversed"),
                 margin=dict(l=0, r=0, t=26, b=0))
        st.plotly_chart(af, width="stretch", key="home_arms",
                        config={"displayModeBar": False})
    with a2:
        st.plotly_chart(ui.waterfall_figure(e, dose, params, 300), width="stretch",
                        key="home_waterfall", config={"displayModeBar": False})

    st.caption(
        f"Left: each arm optimised on its own. Our three compounds reach "
        f"{arm_G[ARMS[0][0]] * 100:.1f}%; the two established comparators together, "
        f"hyaluronic acid and {B.name.lower()}, reach "
        f"{arm_G[ARMS[1][0]] * 100:.1f}%. Right: how the recommended formulation "
        f"takes damage from {res['RSD0']:.2f} down to {res['RSD']:.2f} on this day."
    )

ui.note(
    f"<b>Read the control surfactant's dose loosely.</b> Our two film formers "
    "already cover about 96% of the interface, so a third changes protection by "
    "less than a hundredth of a point. The arms chart on the left is the "
    "comparison that carries weight."
)

ui.flag(
    f"<b>Protection is a ratio, so read it with the absolute damage.</b> "
    f"This day scores G = {G * 100:.1f}% against RSD {res['RSD0']:.2f} "
    "untreated. Compromised skin scores a <i>higher</i> G and higher damage at "
    "once, so G alone would suggest less product for the skin that needs more."
)

st.divider()

# ----------------------------------------------------------------------------
# what the model is, in one picture
# ----------------------------------------------------------------------------

st.subheader("What the model does")

c1, c2 = st.columns([1.35, 1])

with c1:
    st.plotly_chart(ui.sankey_figure(res, params, 440, d=dose), width="stretch",
                    key="home_sankey", config={"displayModeBar": False})
    st.caption(
        "The oxidative load on this day, in R_gen units, and where each part of it "
        "ends up. Each stage is coloured by the compound doing most of it. Ozone is "
        "the channel that mostly survives: the film barely slows a gas, and nothing "
        "in the formulation targets ozone specifically."
    )

with c2:
    st.markdown(
        "Three pollution channels, **particulates, ozone and UV**, drive radical "
        "production. The table reads across from the right-hand side of the flow: "
        "the same stages, in the same words."
    )
    st.markdown(ui.stage_table_html(res, dose, params), unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:.85rem;margin-top:.6rem'>Because they act at different "
        "points, the stages <b>multiply</b>. Set every dose to zero and the whole "
        "thing collapses back to the untreated day exactly; that is asserted as a "
        "test.</div>",
        unsafe_allow_html=True,
    )
    st.page_link("views/model.py", label="Read the full derivation", icon=":material/functions:")

st.divider()

# ----------------------------------------------------------------------------
# honesty section + navigation
# ----------------------------------------------------------------------------

ui.flag(
    "<b>Read the doses with care.</b> Only pulcherrimin has an interior optimum. "
    "The other three sit on their caps because nothing in the tested range "
    "penalises them, so they mark the edge of the evidence, not a recommendation."
)

st.write("")
n1, n2, n3, n4 = st.columns(4)
with n1:
    st.page_link("views/simulator.py", label="Open the dose simulator",
                 icon=":material/science:")
with n2:
    st.page_link("views/controls.py", label="Lab compounds vs the controls",
                 icon=":material/compare_arrows:")
with n3:
    st.page_link("views/market.py", label="Lab vs the market",
                 icon=":material/storefront:")
with n4:
    st.page_link("views/model.py", label="How the model works",
                 icon=":material/functions:")

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
        each &mdash; and says plainly how sure it is of every number.
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

CARDS = [
    ("P", "Pulcherrimin", "lab", "antioxidant",
     "An iron-binding red pigment. Screens UV, and locks up the iron that particulates "
     "carry so it cannot drive Fenton chemistry. The only compound with an interior "
     "optimum: past about 0.4 mg/mL it starts harming the cells it protects."),
    ("X", "Xylolipid", "lab", "biosurfactant",
     "Spreads at the skin-air interface and intercepts particles before they land. "
     "Also scavenges radicals directly. Its evidence base is a fruit-juice "
     "preservation study, which is the honest limit of what we can claim."),
    ("L", "Lyso-ornithine lipid", "lab", "biosurfactant",
     "The second film former, effective at a far lower concentration than the first. "
     "Its ceiling here is what the strain can secrete, 10 mg/L, not a measured "
     "efficacy limit."),
    ("H", "Hyaluronic acid", "control", "antioxidant + humectant",
     "The comparator everyone already knows. Holds water in the stratum corneum and "
     "mops up radicals. It is in the formulation and in the control column at once: "
     "it is what a buyer would otherwise be putting on."),
    ("S", "Benchmark biosurfactant", "control", "biosurfactant",
     "The control we added for the film side: acidic sophorolipid, a cosmetic "
     "glycolipid with real human skin data, or SDS as a negative control. Same "
     "shielding, stripping and self-toxicity terms as ours."),
]

cols = st.columns(len(CARDS))
for col, (k, name, kind, mech, body) in zip(cols, CARDS):
    tone = COLOR[k]
    tag_bg = "#eef5f3" if kind == "lab" else "#f5f0f8"
    with col:
        st.markdown(
            f"""
            <div class="card" style="border-top:3px solid {tone}">
              <div style="font-weight:700;font-size:1.02rem;color:{INK}">{name}</div>
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

dose_k, G = ui.c_optimal(ui.ekey(e), ui.pkey(params), dm.DOSE_ORDER)
dose = Dose(*dose_k)
res = ui.c_evaluate(ui.ekey(e), dose_k, ui.pkey(params))

# The three arms, each optimised on its own, so every bar is the best that arm
# can do rather than an arbitrary dose.
ARMS = [
    ("Our compounds<br>P + X + L", dm.LAB_KEYS, "#2f6f4f"),
    (f"Both controls<br>HA + {B.name}", dm.CONTROL_KEYS, COLOR["S"]),
    ("All five<br>together", dm.ALL_KEYS, INK),
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
        keys = dm.DOSE_ORDER
        bar.add_bar(
            x=[getattr(dose, k) / dm.cap_of(k, params) * 100 for k in keys],
            y=[ui.compound_label(k) for k in keys],
            orientation="h",
            marker_color=[COLOR[k] for k in keys],
            text=[f"{getattr(dose, k):.3f} mg/mL" for k in keys],
            textposition="auto",
            hovertemplate="%{y}<br>%{x:.0f}% of its cap<extra></extra>",
        )
        ui.style(bar, 250, showlegend=False,
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
    f"<b>Why the control surfactant is not in the dose table above.</b> The "
    f"recommendation is our own formulation, and adding {B.name.lower()} on top of "
    "it buys almost nothing: xylolipid and lyso-ornithine lipid have already "
    "covered about 96% of the interface, so a third film former has nothing left "
    "to cover. The comparison that means something is the one on the left, where "
    "each arm is built from scratch."
)

ui.flag(
    f"<b>Protection is a ratio, so read it with the absolute number.</b> This day "
    f"scores G = {G * 100:.1f}% against RSD {res['RSD0']:.2f} untreated. Compromised "
    "skin scores a <i>higher</i> G and higher absolute damage at once, because a "
    "smaller reserve of its own leaves more room for the applied compounds to win. "
    "Quoted alone, G would recommend less product for the skin that needs more."
)

st.divider()

# ----------------------------------------------------------------------------
# what the model is, in one picture
# ----------------------------------------------------------------------------

st.subheader("What the model does")

c1, c2 = st.columns([1.35, 1])

with c1:
    st.plotly_chart(ui.sankey_figure(res, params, 400), width="stretch",
                    key="home_sankey", config={"displayModeBar": False})
    st.caption(
        "The oxidative load on this day, in R_gen units, and where each part of it "
        "ends up. Ozone is the channel that mostly survives: the film barely slows a "
        "gas, and nothing in the formulation targets ozone specifically."
    )

with c2:
    st.markdown(
        """
Three pollution channels — **particulates, ozone and UV** — drive radical
production and barrier damage. Six stages then say what the compounds do about
it, each acting at a different point in the chain:

| stage | what happens | who |
|---|---|---|
| S1a | particles stopped at the film | X, L, S |
| S1b | UV screened | P |
| S2 | iron chelated, Fenton suppressed | P |
| S3 | radicals scavenged | X, H, skin |
| S4 | water held in, or stripped out | H, L, X, S |
| S5 | the delay between sensing and secreting | all |
| S6 | the dose harming the skin itself | P, X, L, S |

Because they act at different points, the stages **multiply**. Set every dose to
zero and the whole thing collapses back to the untreated day exactly — that is
asserted as a test, not hoped for.
        """
    )
    st.page_link("views/model.py", label="Read the full derivation", icon=":material/functions:")

st.divider()

# ----------------------------------------------------------------------------
# honesty section + navigation
# ----------------------------------------------------------------------------

st.subheader("What this model cannot tell you")

_ref_total = dm.formulation_ratio(Dose(*_ref_dose_k))["total"]

k1, k2, k3 = st.columns(3)
with k1:
    ui.flag(
        "<b>Three of four doses are not recommendations.</b> Only pulcherrimin has an "
        "interior optimum. The rest sit on their caps because nothing inside the "
        "validity domain penalises them — that is the model saying the evidence ran "
        "out, not that this is the right amount."
    )
with k2:
    ui.flag(
        "<b>The best-evidenced compound contributes the least.</b> Pulcherrimin has "
        "human keratinocyte data. The two biosurfactants, which carry most of the "
        "headline number, have never touched a keratinocyte in the published record."
    )
with k3:
    ui.flag(
        f"<b>Our dose is tiny next to the market.</b> The whole formulation is "
        f"about {_ref_total / dm.PCT_TO_MG_PER_ML:.2f} % by weight, at or below the "
        "bottom of every published commercial range, and a vitamin C serum carries "
        "a hundred times the antioxidant mass. The case has to be that this "
        "responds to the day, not that it delivers more."
    )

st.write("")
with st.container():
    ui.flag(
        "<b>One unmeasured constant dominates.</b> theta_max, the fraction of "
        "particles a saturated film can stop, moves the answer by more than twenty "
        "points across its plausible range. Nobody has measured it. That experiment "
        "is a membrane, some reference dust and a particle counter."
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

st.caption(
    "iGEM IIT Delhi 2026 - DermaSense. Model outputs are predictions from a "
    "parameterised cascade. They are not clinical guidance and not a formulation "
    "instruction."
)

"""
How the model works: the derivation, stage by stage, with every curve drawn
from the same code the simulator runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dermasense_model as dm
import ui
from dermasense_model import Dose, Environment, NOMINAL
from ui import CHANNEL_COLOR, COLOR, INK, MUTED, RUST, SIGNAL

ui.eyebrow("derivation")
st.title("How the model works")
st.markdown(
    f"<span style='color:{MUTED}'>Every equation below is the one that runs. The curves "
    "are drawn by calling the model, so nothing on this page can drift away from the "
    "code.</span>",
    unsafe_allow_html=True,
)

env = ui.shared_env()
params = dm.benchmark_params(ui.benchmark_key())

st.divider()

# ----------------------------------------------------------------------------
# 1. the one function everything is built from
# ----------------------------------------------------------------------------

st.subheader("1. One function, used seven times")

c1, c2 = st.columns([1, 1.2])
with c1:
    st.latex(r"H(c; K, n) = \frac{c^{n}}{K^{n} + c^{n}}")
    st.markdown(
        """
A saturating response. Doubling the dose stops doubling the effect once the
sites are full, which is how every biological binding, adsorption and
antioxidant response actually behaves.

`K` is the concentration at half effect — the only number that matters for
comparing two compounds doing the same job. `n` is cooperativity, and is 1
everywhere except the self-toxicity term, where a threshold that turns on
sharply is the point.

The same shape carries **interfacial coverage** (S1a), **iron chelation** (S2),
**humectancy** (S4), **the skin's own reserve depleting** (Part 1) and
**self-toxicity** (S6). Different constants each time; one curve.
        """
    )
with c2:
    hf = go.Figure()
    c = np.linspace(0, 4, 300)
    for K, n, lab, col in [(0.5, 1, "K = 0.5, n = 1", SIGNAL),
                           (1.0, 1, "K = 1.0, n = 1", "#3d5a80"),
                           (1.0, 2, "K = 1.0, n = 2 (threshold)", RUST)]:
        hf.add_scatter(x=c, y=[dm.hill(float(v), K, n) for v in c], name=lab,
                       line=dict(color=col, width=2.4))
    hf.add_hline(y=0.5, line=dict(color=MUTED, dash="dash", width=1))
    ui.style(hf, 320, xaxis=dict(title="concentration c"),
             yaxis=dict(title="H(c)", range=[0, 1.02]),
             legend=dict(orientation="h", y=-0.24))
    st.plotly_chart(hf, width="stretch", key="m_hill",
                    config={"displayModeBar": False})

st.divider()

# ----------------------------------------------------------------------------
# 2. part one: exposure to damage
# ----------------------------------------------------------------------------

st.subheader("2. From the air to a damage score, before any product")

st.markdown(
    """
Three channels, each normalised against a reference day so the whole model is
dimensionless and a score of 1.00 always means *the reference day*:
    """
)
st.latex(r"p = \frac{C_{PM}}{80},\qquad z = \frac{C_{O_3}}{60},"
         r"\qquad u = \frac{I_{UV}\, f_{PT}}{1.0}")
st.latex(r"R_{gen} = w_{PM}\,p + w_{O_3}\,z + w_{UV}\,u"
         r"\qquad (0.40,\ 0.35,\ 0.25)")

d1, d2 = st.columns([1.1, 1])
with d1:
    st.markdown(
        """
`f_PT` is the phototype modifier, and it acts on the **UV channel only** —
melanin does not protect against ozone.

Weights: particulates carry the most mass; the ozone weight is anchored to He
et al. (2006), who measured ozone-driven radical generation; UV is smallest
because most of its damage is already inside the photodamage term.

The skin then fights back. Its own antioxidant reserve `kappa_endo` depletes as
the load rises — the worse the day, the less reserve is left, which is why
damage grows faster than exposure does:
        """
    )
    st.latex(r"\kappa_{endo} = A_0\,\kappa_0\left(1 - d_{max}H(R_{gen}; K_d)\right)")
    st.latex(r"\Phi_{scav} = \frac{1}{1 + \kappa_{endo}},"
             r"\qquad R = \frac{R_{gen}\,\Phi_{scav}}{N_R}")
    ui.note(
        "<b>N_R is deliberately frozen</b> at the reference day with healthy skin "
        "and no product. If it were recomputed per person, everybody would score "
        "R = 1 on their own reference and the model would lose the ability to say "
        "that some people are simply worse off than others."
    )

with d2:
    pm_grid = np.linspace(0, 400, 120)
    rf = go.Figure()
    for A0, lab, dash in [(dm.A0_HEALTHY, "healthy barrier", "solid"),
                          (dm.A0_COMPROMISED, "compromised barrier", "dot")]:
        ys = [dm.evaluate(Environment(C_PM=float(v), C_O3=env.C_O3, I_UV=env.I_UV,
                                      PT=env.PT, A0=A0), dm.ZERO_DOSE, params)["RSD0"]
              for v in pm_grid]
        rf.add_scatter(x=pm_grid, y=ys, name=lab,
                       line=dict(color=INK if A0 == 1.0 else RUST, width=2.4, dash=dash))
    rf.add_vline(x=80, line=dict(color=SIGNAL, dash="dash"),
                 annotation_text="reference day", annotation_font=dict(size=11, color=SIGNAL))
    ui.style(rf, 300, xaxis=dict(title="PM2.5, ug/m3"),
             yaxis=dict(title="untreated damage RSD"),
             legend=dict(orientation="h", y=-0.26))
    st.plotly_chart(rf, width="stretch", key="m_rsd0",
                    config={"displayModeBar": False})
    st.caption(
        "Untreated damage against particulate load, at the ozone and UV currently "
        "loaded. The compromised curve sits above the healthy one everywhere and "
        "the gap widens: a smaller reserve depletes sooner."
    )

st.divider()

# ----------------------------------------------------------------------------
# 3. the six stages
# ----------------------------------------------------------------------------

st.subheader("3. Six stages, acting at six different points")

st.markdown(
    f"<span style='color:{MUTED}'>This is the part that matters: the stages act at "
    "different points in one causal chain, so they <b>multiply</b> rather than add. "
    "A film that stops a particle before it lands means there is no radical for an "
    "antioxidant to scavenge later.</span>",
    unsafe_allow_html=True,
)

s_tabs = st.tabs(["S1a film", "S1b UV screen", "S2 iron", "S3 scavenging",
                  "S4 barrier", "S5 timing", "S6 self-harm"])

with s_tabs[0]:
    a, b = st.columns([1, 1.2])
    with a:
        st.markdown("**Interfacial shielding** — xylolipid, lyso-ornithine lipid, "
                    "and the control surfactant, all competing for one surface.")
        st.latex(r"\theta = \frac{L/CMC_L + X/K_X + S/CMC_S}"
                 r"{1 + L/CMC_L + X/K_X + S/CMC_S}")
        st.latex(r"\tilde{C}_{PM} = (1 - \theta_{max}\theta)\,C_{PM},\qquad"
                 r"\tilde{C}_{O_3} = (1 - \theta_{max}\phi_{gas}\theta)\,C_{O_3}")
        st.markdown(
            "One shared denominator, so the three surfactants **compete**: adding "
            "the third to a saturated film buys almost nothing. `phi_gas = 0.30` "
            "says a monolayer stops a gas far less well than a particle, which is "
            "why ozone is the channel that survives."
        )
        ui.flag(
            "<b>theta_max is the single biggest unknown in the model.</b> It is the "
            "ceiling on what a fully saturated film can intercept. Nobody has "
            "measured it for any biosurfactant. Across its plausible range it moves "
            "the final answer by more than twenty points."
        )
    with b:
        tm = go.Figure()
        th = np.linspace(0, 1, 100)
        for tmax, col in [(0.20, "#c9c2b4"), (0.45, SIGNAL), (0.65, INK)]:
            tm.add_scatter(x=th * 100, y=(1 - tmax * th) * 100,
                           name=f"theta_max = {tmax}",
                           line=dict(color=col, width=2.4))
        ui.style(tm, 300, xaxis=dict(title="interface coverage, %"),
                 yaxis=dict(title="particulate reaching skin, %"),
                 legend=dict(orientation="h", y=-0.26))
        st.plotly_chart(tm, width="stretch", key="m_theta",
                        config={"displayModeBar": False})

with s_tabs[1]:
    a, b = st.columns([1, 1.2])
    with a:
        st.markdown("**UV screening** — pulcherrimin absorbs, so less UV arrives.")
        st.latex(r"SPF(P) = 1 + \epsilon_{film}\,a_{SPF}\,P,"
                 r"\qquad \tilde{I}_{UV} = \frac{I_{UV}}{SPF(P)}")
        st.markdown(
            "`a_SPF = 95` per mg/mL comes from Kregiel 2024 (SPF 20 at 0.2 mg/mL, "
            "1 at zero). `eps_film = 0.25` is a haircut: a cuvette of dissolved "
            "pigment is not a film on skin, and assuming otherwise would be the "
            "single most flattering error available to us."
        )
    with b:
        sf = go.Figure()
        P = np.linspace(0, dm.CAPS["P"], 120)
        for eps, col, lab in [(1.0, "#c9c2b4", "cuvette, eps = 1.0"),
                              (0.25, COLOR["P"], "film haircut, eps = 0.25")]:
            sf.add_scatter(x=P, y=1 + eps * params.spf_slope * P, name=lab,
                           line=dict(color=col, width=2.4))
        ui.style(sf, 300, xaxis=dict(title="pulcherrimin, mg/mL"),
                 yaxis=dict(title="SPF"), legend=dict(orientation="h", y=-0.26))
        st.plotly_chart(sf, width="stretch", key="m_spf",
                        config={"displayModeBar": False})

with s_tabs[2]:
    st.markdown("**Fenton suppression** — pulcherrimin chelates the iron that "
                "particulates carry, upstream of any radical being made.")
    st.latex(r"\chi = E_{chel}\,H(P; K_{chel}),\qquad"
             r"\tilde{R}_{gen} = w_{PM}\tilde{p}\left(1 - \phi_{Fe}\chi\right)"
             r" + w_{O_3}\tilde{z} + w_{UV}\tilde{u}")
    a, b = st.columns([1, 1.2])
    with a:
        st.markdown(
            "This is why pulcherrimin's scavenging rate constant `k_P` is set to "
            "**zero** in S3. Kregiel's measurement is of the chelation effect, and "
            "counting it again as direct scavenging would be the same evidence used "
            "twice. `phi_Fe = 0.50`: half the particulate radical burden is "
            "iron-driven, and that half is ours to suppress."
        )
    with b:
        chf = go.Figure()
        P = np.linspace(0, dm.CAPS["P"], 120)
        chf.add_scatter(x=P, y=[params.E_chel * dm.hill(float(v), params.K_chel) * 100
                                for v in P],
                        line=dict(color=COLOR["P"], width=2.4), name="chelated fraction")
        ui.style(chf, 300, showlegend=False, xaxis=dict(title="pulcherrimin, mg/mL"),
                 yaxis=dict(title="iron chelated, %", range=[0, 100]))
        st.plotly_chart(chf, width="stretch", key="m_chel",
                        config={"displayModeBar": False})

with s_tabs[3]:
    st.markdown("**Competitive radical scavenging** — everything that scavenges "
                "shares one denominator, including the skin itself.")
    st.latex(r"\kappa_{app} = \eta\,\psi\left(k_X X + k_H H\right),\qquad"
             r"\Phi_{scav} = \frac{1}{1 + \kappa_{endo} + \kappa_{app}}")
    a, b = st.columns([1, 1.2])
    with a:
        ui.note(
            "<b>This shared denominator is the sub-additivity.</b> The first "
            "antioxidant removes a large share of the radical pool; the fourth is "
            "competing with the first three for what is left. It is the one-line "
            "answer to why a four-compound formulation is not four times better "
            "than a one-compound one."
        )
        st.markdown(
            "`psi = 0.35` is a transfer factor from a cell-free DPPH assay to whole "
            "cells. A tube of radicals and a living keratinocyte are not the same "
            "environment, and the constant says so out loud rather than pretending "
            "the assay transfers at face value."
        )
    with b:
        per_c = dm.per_compound_optima(env, params)
        _, g_joint = dm.optimal_dose(env, params)
        sub = go.Figure()
        sub.add_bar(x=[f"{k} alone" for k in dm.DOSE_ORDER],
                    y=[per_c[k] * 100 for k in dm.DOSE_ORDER],
                    marker_color=[COLOR[k] for k in dm.DOSE_ORDER],
                    text=[f"{per_c[k] * 100:.0f}%" for k in dm.DOSE_ORDER],
                    textposition="auto", name="alone")
        sub.add_bar(x=["naive sum", "actual, together"],
                    y=[sum(per_c.values()) * 100, g_joint * 100],
                    marker_color=[RUST, INK],
                    text=[f"{sum(per_c.values()) * 100:.0f}%", f"{g_joint * 100:.0f}%"],
                    textposition="auto", showlegend=False)
        ui.style(sub, 320, showlegend=False, yaxis=dict(title="protection G, %"))
        st.plotly_chart(sub, width="stretch", key="m_subadd",
                        config={"displayModeBar": False})

with s_tabs[4]:
    st.markdown("**Barrier modulation** — hyaluronic acid holds water in; a "
                "surfactant past its threshold strips lipids out.")
    st.latex(r"\Gamma_H = 1 - \eta_H H(H; K_H),\qquad"
             r"\Psi = 1 + s\sum_i \max(0,\, c_i - C_{crit,i})")
    st.latex(r"T' = (T_0 + \Delta\tilde{T})\,\Gamma_H\,\Psi")
    a, b = st.columns([1, 1.2])
    with a:
        st.markdown(
            "`Gamma_H` below 1 is an improvement, `Psi` above 1 is net harm. The "
            "two fight each other, which is exactly the trade a real moisturiser "
            "makes: the same molecules that spread the film can also dissolve the "
            "lipids holding water in."
        )
        ui.flag(
            "<b>Inside the validity domain, Psi never fires for our compounds.</b> "
            "C_crit sits above the xylolipid cap, so the stripping penalty is "
            "switched off over the whole recommended range. That is not a finding, "
            "it is an artefact of where the caps are, and it is why three of four "
            "doses pin to their ceilings."
        )
    with b:
        gf = go.Figure()
        H = np.linspace(0, dm.CAPS["H"], 120)
        gf.add_scatter(x=H, y=[(1 - params.eta_H * dm.hill(float(v), params.K_H)) * 100
                               for v in H],
                       line=dict(color=COLOR["H"], width=2.4), name="humectant gain")
        ui.style(gf, 300, showlegend=False,
                 xaxis=dict(title="hyaluronic acid, mg/mL"),
                 yaxis=dict(title="water loss, % of untreated"))
        st.plotly_chart(gf, width="stretch", key="m_gamma",
                        config={"displayModeBar": False})

with s_tabs[5]:
    st.markdown("**Delivery timing** — the price of sensing before secreting.")
    st.latex(r"\eta(t) = \eta_{post} + (1 - \eta_{post})e^{-t/\tau},"
             r"\qquad D_\eta[\phi] = 1 - \eta(1 - \phi)")
    a, b = st.columns([1, 1.2])
    with a:
        st.markdown(
            "The OxyR circuit fires *after* stress begins, so what the system "
            "delivers is a post-treatment, not a pre-treatment. The discount scales "
            "the protective part — the distance from 1 — rather than the value "
            "itself. Multiplying a protection factor directly would manufacture "
            "protection out of nothing whenever the factor is already 1."
        )
    with b:
        lf = go.Figure()
        lag = np.linspace(0, 12, 40)
        gs = [dm.optimal_dose(Environment(C_PM=env.C_PM, C_O3=env.C_O3, I_UV=env.I_UV,
                                          PT=env.PT, A0=env.A0, lag_h=float(t)),
                              params)[1] * 100 for t in lag]
        lf.add_scatter(x=lag, y=gs, line=dict(color=SIGNAL, width=2.4))
        ui.style(lf, 300, showlegend=False, xaxis=dict(title="lag, hours"),
                 yaxis=dict(title="protection G, %"))
        st.plotly_chart(lf, width="stretch", key="m_lag",
                        config={"displayModeBar": False})
        st.caption("Worth about three points of G across the whole range — real, "
                   "but not the thing that decides whether this works.")

with s_tabs[6]:
    st.markdown("**Self-toxicity** — the reason a recommended dose exists at all.")
    st.latex(r"\Omega = \rho_P H(P; C_{horm}, 2) + s\,H(X; C_{crit}, 2)"
             r" + s\,H(L; C_{crit}, 2) + s_S H(S; C_{crit,S}, 2)")
    st.latex(r"RSD = w_{ox}R' + (1 - w_{ox})B' + \Omega D_s")
    a, b = st.columns([1, 1.2])
    with a:
        st.markdown(
            "`n = 2` here, not 1: harm has a threshold and then rises quickly, "
            "rather than creeping in from the first microgram. Without this term "
            "every dose curve would rise forever and the model would recommend "
            "drowning the face in product."
        )
        ui.note(
            "<b>Only pulcherrimin's threshold falls inside its cap</b>, at "
            "0.40 mg/mL from Kregiel's efficacy reversal. So pulcherrimin is the "
            "one compound with a genuine interior optimum, and the one number on "
            "the dose table that is actually a recommendation."
        )
    with b:
        of = go.Figure()
        P = np.linspace(0, dm.CAPS["P"], 150)
        of.add_scatter(x=P, y=[params.rho_P * dm.hill(float(v), params.C_horm,
                                                      params.n_horm) for v in P],
                       line=dict(color=COLOR["P"], width=2.4), name="omega from P")
        gP = []
        base = dm.optimal_dose(env, params)[0]
        for v in P:
            gP.append(dm.protection(env, Dose(**{**base.__dict__, "P": float(v)}),
                                    params) * 100)
        of.add_scatter(x=P, y=np.array(gP) / max(gP) * params.rho_P, name="G, rescaled",
                       line=dict(color=INK, width=2, dash="dot"), yaxis="y")
        best = float(P[int(np.argmax(gP))])
        of.add_vline(x=best, line=dict(color=SIGNAL, dash="dash"),
                     annotation_text=f"optimum {best:.3f}",
                     annotation_font=dict(size=11, color=SIGNAL))
        ui.style(of, 300, xaxis=dict(title="pulcherrimin, mg/mL"),
                 yaxis=dict(title="self-harm omega (G rescaled to match)"),
                 legend=dict(orientation="h", y=-0.26))
        st.plotly_chart(of, width="stretch", key="m_omega",
                        config={"displayModeBar": False})

st.divider()

# ----------------------------------------------------------------------------
# 4. assembly
# ----------------------------------------------------------------------------

st.subheader("4. Putting it together")

a, b = st.columns([1, 1])
with a:
    st.latex(r"RSD(e,d) = w_{ox}R'(e,d) + (1 - w_{ox})B'(e,d) + \Omega D_s")
    st.latex(r"G(e,d) = 1 - \frac{RSD(e,d)}{RSD(e,0)}")
    st.markdown(
        """
`w_ox = 0.5` splits damage evenly between the oxidative and barrier axes. `G`
is the share of *that day's* untreated damage the product removes.

Set every dose to zero and this collapses to Part 1 exactly, to 1e-12, at any
exposure and not only at the reference day. `test_model.py` asserts it, so any
future edit that breaks the correspondence fails a test instead of quietly
becoming a second model sharing notation with the first.
        """
    )
with b:
    dose_k, G = ui.c_optimal(ui.ekey(env), ui.pkey(params))
    st.plotly_chart(ui.waterfall_figure(env, Dose(*dose_k), params, 340),
                    width="stretch", key="m_waterfall",
                    config={"displayModeBar": False})

st.divider()

# ----------------------------------------------------------------------------
# 5. evidence classes
# ----------------------------------------------------------------------------

st.subheader("5. Every constant, and how much we trust it")

counts = pd.Series(list(dm.EVIDENCE.values())).value_counts()
cls_cols = st.columns(4)
for col, (cls, label, colour) in zip(cls_cols, [
    ("A", "measured directly", "#2f6f4f"),
    ("B", "fitted to one study", "#b08a3e"),
    ("C", "nobody measured it", RUST),
    ("-", "structurally zero", MUTED),
]):
    with col:
        st.markdown(
            f"""<div class="card" style="border-top:3px solid {colour}">
            <div class="readout-label">class {cls}</div>
            <div class="readout" style="font-size:1.8rem">{int(counts.get(cls, 0))}</div>
            <div class="readout-sub">{label}</div></div>""",
            unsafe_allow_html=True,
        )

st.write("")
rows = []
for name, cls in dm.EVIDENCE.items():
    lo, hi = dm.BANDS.get(name, (None, None))
    rows.append({"Constant": name, "Value": getattr(params, name, None),
                 "Class": cls, "Low": lo, "High": hi})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=380)

ui.flag(
    "<b>Four class C constants carry most of the spread:</b> theta_max, "
    "assay_xfer, C_crit and kappa_0. Three of the four are directly measurable "
    "with experiments this team could run inside a year, which is the useful "
    "output of an uncertainty analysis — not an error bar, a to-do list. The "
    "<b>Uncertainty</b> tab of the simulator recomputes that ranking for whatever "
    "day is loaded, so the priorities are derived rather than quoted."
)

st.divider()

st.subheader("6. What this model is not")

l1, l2 = st.columns(2)
with l1:
    st.markdown(
        """
- Both lab biosurfactants are extrapolated from a **fruit-juice preservation**
  study and a **crude-oil emulsification** study. Neither has met a keratinocyte
  in the published record, yet together they carry most of the headline number.
- The Milani agreement (18.7% predicted against 19% measured) validates the
  **scale of the barrier channel**, nothing about the shielding mechanism —
  Milani's serum contains no film-forming surfactant at all.
- Channels **add** rather than multiply, so UV and ozone synergy is ignored.
  This under-predicts damage under simultaneous exposure.
        """
    )
with l2:
    st.markdown(
        """
- Concentration in a film is assumed to be the right variable, rather than a
  surface density in mg/cm2, and film thickness is assumed identical across
  people. No source examines either assumption.
- `alpha_PM` is fitted over roughly 30 to 100 ug/m3. At Delhi winter levels the
  barrier channel is **extrapolating**.
- The live feed is a CAMS grid cell, not the air this person is breathing.
  Indoors, in traffic, or twenty floors up, it is wrong in a direction the app
  cannot see.
        """
    )

st.caption(
    "iGEM IIT Delhi 2026 - DermaSense. Model outputs are predictions from a "
    "parameterised cascade. They are not clinical guidance and not a formulation "
    "instruction."
)

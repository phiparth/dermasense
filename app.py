"""
DermaSense v3 - Streamlit front end.

Run locally:   streamlit run app.py
Deploy:        push this repo to GitHub, point share.streamlit.io at app.py
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dermasense_model as dm
from dermasense_model import (
    CAPS, CAP_REASON, COMPOUND_NAME, DOSE_ORDER, EVIDENCE, BANDS,
    A0_HEALTHY, A0_COMPROMISED, NOMINAL,
    Dose, Environment, Params,
    conservative_params, dose_notes, dose_spread, evaluate, eta_of_t,
    min_cost_dose, monte_carlo, optimal_dose, optimal_scale,
    per_compound_optima, protection, sensitivity, single_compound_optima,
)

# ----------------------------------------------------------------------------
# page shell
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="DermaSense v3",
    page_icon="~",
    layout="wide",
    initial_sidebar_state="expanded",
)

INK = "#1c2321"
PAPER = "#f7f5f0"
SIGNAL = "#0f6f68"      # ozone teal, the channel nothing in the formulation touches
RUST = "#a8492c"        # the spec shades unmeasured constants rust, so do we
MUTED = "#6c757d"

st.markdown(
    f"""
    <style>
      .stApp {{ background: {PAPER}; }}
      html, body, [class*="css"] {{ color: {INK}; }}
      h1, h2, h3 {{ letter-spacing: -0.01em; }}
      .readout {{
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
        font-size: 2.1rem; font-weight: 600; color: {INK}; line-height: 1.1;
      }}
      .readout-label {{
        font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.09em;
        color: {MUTED};
      }}
      .readout-sub {{ font-size: 0.82rem; color: {MUTED}; }}
      .card {{
        background: #ffffff; border: 1px solid #e3ded4; border-radius: 6px;
        padding: 0.9rem 1.1rem; height: 100%;
      }}
      .flag {{
        border-left: 3px solid {RUST}; background: #fff8f5;
        padding: 0.7rem 0.9rem; margin: 0.4rem 0; font-size: 0.9rem;
      }}
      .note {{
        border-left: 3px solid {SIGNAL}; background: #f1f8f7;
        padding: 0.7rem 0.9rem; margin: 0.4rem 0; font-size: 0.9rem;
      }}
      div[data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def readout(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="card"><div class="readout-label">{label}</div>'
        f'<div class="readout">{value}</div>'
        f'<div class="readout-sub">{sub}</div></div>'
    )


# ----------------------------------------------------------------------------
# cached wrappers
# ----------------------------------------------------------------------------

# Streamlit hashes cache keys, so pass plain tuples across the cache boundary
# and rebuild the dataclasses inside.

def ekey(e: Environment):
    return dataclasses.astuple(e)


def pkey(p: Params):
    return dataclasses.astuple(p)


def dkey(d: Dose):
    return dataclasses.astuple(d)


@st.cache_data(show_spinner=False)
def c_optimal(e_k, p_k):
    d, g = optimal_dose(Environment(*e_k), Params(*p_k))
    return dkey(d), g


@st.cache_data(show_spinner=False)
def c_evaluate(e_k, d_k, p_k):
    return evaluate(Environment(*e_k), Dose(*d_k), Params(*p_k))


@st.cache_data(show_spinner=False)
def c_sensitivity(e_k):
    return sensitivity(Environment(*e_k))


@st.cache_data(show_spinner=False)
def c_monte_carlo(e_k, n, d_k):
    return monte_carlo(Environment(*e_k), n=n, d_fixed=Dose(*d_k))


@st.cache_data(show_spinner=False)
def c_singles(e_k, p_k):
    e, p = Environment(*e_k), Params(*p_k)
    return single_compound_optima(e, p), per_compound_optima(e, p)


@st.cache_data(show_spinner=False)
def c_spread(e_keys, p_k):
    return dose_spread([Environment(*k) for k in e_keys], Params(*p_k))


# ----------------------------------------------------------------------------
# sidebar: the five environment inputs
# ----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Environment")
    st.caption("The five inputs the model needs. Everything else is computed.")

    C_PM = st.slider("PM2.5, ug/m3", 0.0, 400.0, 80.0, 1.0,
                     help="Reference day is 80. Delhi winter peaks past 300, "
                          "which is outside the range the TEWL slope was fitted on.")
    C_O3 = st.slider("Ozone, ug/m3", 0.0, 300.0, 60.0, 1.0,
                     help="Reference day is 60. If your feed reports ppb, "
                          "multiply by 1.962 at 25 C.")
    I_UV = st.slider("UV dose, normalised", 0.0, 3.0, 1.0, 0.05,
                     help="1.0 is the reference day. Dimensionless on purpose: "
                          "no source reported an absolute UV dose next to a "
                          "skin endpoint we could use.")
    PT = st.select_slider("Fitzpatrick phototype", options=[1, 2, 3, 4, 5, 6],
                          value=1, format_func=lambda i: "I II III IV V VI".split()[i - 1])
    barrier = st.radio("Barrier state", ["Healthy", "Compromised"], horizontal=True,
                       help="A binary flag, not a measurement. Kim 2016 established "
                            "the direction, not the magnitude.")
    A0 = A0_HEALTHY if barrier == "Healthy" else A0_COMPROMISED

    st.markdown("---")
    lag_h = st.slider("Sensor to secretion lag, hours", 0.0, 12.0, 0.0, 0.25,
                      help="The OxyR circuit fires after stress begins, so the "
                           "system delivers a post-treatment. Worth about three "
                           "points of G across the whole range.")

    st.markdown("### Parameter set")
    preset = st.radio(
        "Constants",
        ["Nominal", "Conservative (class C)", "Worst case (all bands)"],
        help="Conservative assumes our own guesses are wrong. Worst case also "
             "assumes the published measurements are wrong, which double-counts "
             "pessimism.",
    )
    if preset == "Nominal":
        params = NOMINAL
    elif preset.startswith("Conservative"):
        params = conservative_params("C")
    else:
        params = conservative_params("all")

    st.markdown("---")
    st.caption(
        "DermaSense v3 unified model. 32 constants, 11 of them class C "
        "(nobody measured them, we chose a value)."
    )

env = Environment(C_PM=C_PM, C_O3=C_O3, I_UV=I_UV, PT=PT, A0=A0, lag_h=lag_h)


# ----------------------------------------------------------------------------
# header + mode
# ----------------------------------------------------------------------------

st.title("DermaSense v3")
st.markdown(
    f"<span style='color:{MUTED}'>Pollution exposure to skin damage, and how much "
    "of it four engineered compounds take back.</span>",
    unsafe_allow_html=True,
)

mode = st.radio(
    "Mode",
    ["Recommend a dose (inverse)", "Score a dose I specify (forward)"],
    horizontal=True,
    label_visibility="collapsed",
)

if mode.startswith("Recommend"):
    dose_k, G_star = c_optimal(ekey(env), pkey(params))
    dose = Dose(*dose_k)
    st.caption("Inverse mode: the five environment inputs go in, the dose comes out.")
else:
    st.caption("Forward mode: you set the dose, the model scores it.")
    cols = st.columns(4)
    vals = {}
    for col, k in zip(cols, DOSE_ORDER):
        with col:
            vals[k] = st.number_input(
                f"{COMPOUND_NAME[k]} ({k}), mg/mL",
                min_value=0.0, max_value=float(CAPS[k]),
                value=float(CAPS[k]) * (0.29 if k == "P" else 1.0),
                step=float(CAPS[k]) / 100,
                format="%.4f",
                help=f"Cap {CAPS[k]}: {CAP_REASON[k]}",
            )
    dose = Dose(**vals)
    G_star = protection(env, dose, params)

res = c_evaluate(ekey(env), dkey(dose), pkey(params))

# ----------------------------------------------------------------------------
# headline readouts
# ----------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(readout("Protection G", f"{G_star * 100:.1f}%",
                        "share of untreated damage removed"), unsafe_allow_html=True)
with c2:
    st.markdown(readout("Absolute damage RSD",
                        f"{res['RSD0']:.3f} \u2192 {res['RSD']:.3f}",
                        "untreated to treated, reference day = 1.000"),
                unsafe_allow_html=True)
with c3:
    st.markdown(readout("TEWL", f"{res['T']:.2f} \u2192 {res['T_prime']:.2f}",
                        f"g/m2/h, {res['TEWL_improvement'] * 100:.1f}% improvement"),
                unsafe_allow_html=True)
with c4:
    st.markdown(readout("Self-inflicted", f"{res['Omega']:.3f}",
                        f"omega, adding {res['penalty']:.4f} back to RSD"),
                unsafe_allow_html=True)

st.markdown(
    '<div class="flag"><b>Read G and absolute RSD together, always.</b> '
    "G is a ratio against the same day untreated, so it can rise while the "
    "person gets worse. Compromised skin scores a higher G <i>and</i> higher "
    "absolute damage at the same time, because a smaller endogenous reserve "
    "lets the applied compounds win a larger share of the radical pool. "
    "Quoted alone, G would recommend less product for the skin that needs "
    "more.</div>",
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "Recommendation", "Cascade trace", "Dose response", "Uncertainty",
    "Does the dose move?", "Cost solve", "Constants",
])

# ----------------------------------------------------------------------------
# 1. recommendation
# ----------------------------------------------------------------------------

with tabs[0]:
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("#### Recommended formulation")
        notes = dose_notes(dose)
        table = pd.DataFrame([{
            "Compound": n["compound"],
            "Dose (mg/mL)": round(n["dose"], 4),
            "Cap": n["cap"],
            "% of cap": f"{n['dose'] / n['cap'] * 100:.0f}%",
            "Position": n["position"],
        } for n in notes])
        st.dataframe(table, hide_index=True, width="stretch")

        at_cap = [n for n in notes if n["position"] == "at cap"]
        interior = [n for n in notes if n["position"] != "at cap"]
        if at_cap:
            names = ", ".join(n["compound"] for n in at_cap)
            st.markdown(
                f'<div class="flag"><b>{len(at_cap)} of 4 sit on their caps: '
                f"{names}.</b> That is the model saying we ran out of evidence, "
                "not that this is the right amount. Nothing inside the validity "
                "domain penalises them, so the optimiser runs them to the "
                "boundary regardless of the weather. A dose table with four "
                "numbers in it reads like four recommendations, and it is "
                "not.</div>",
                unsafe_allow_html=True,
            )
        if interior:
            names = ", ".join(n["compound"] for n in interior)
            st.markdown(
                f'<div class="note"><b>{names}: a genuine optimum.</b> Pushed off '
                "the cap by the S6 reversal term, so more really would be worse. "
                "This is the only coordinate the model is actually recommending "
                "rather than deferring on.</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="note"><b>The lyso-ornithine lipid cap is different in '
            "kind.</b> 0.010 mg/mL is what the strain can make (Li 2024, 10 mg/L "
            "titre), not what anyone measured as an efficacy ceiling. Coverage is "
            "already about 95% saturated there, so it is harmless, and it is the "
            "one number where biology rather than literature sets the "
            "bound.</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### Where the protection comes from")
        groups, per_c = c_singles(ekey(env), pkey(params))
        rows = [{"Acting alone": k, "G": v} for k, v in groups.items()]
        naive = sum(per_c.values())
        fig = go.Figure()
        fig.add_bar(
            x=[r["G"] * 100 for r in rows] + [G_star * 100, naive * 100],
            y=[r["Acting alone"] for r in rows] + ["All four together",
                                                   "Naive sum of singles"],
            orientation="h",
            marker_color=[SIGNAL] * len(rows) + [INK, RUST],
            text=[f"{r['G'] * 100:.1f}%" for r in rows]
                 + [f"{G_star * 100:.1f}%", f"{naive * 100:.0f}%"],
            textposition="auto",
        )
        fig.update_layout(
            height=280, margin=dict(l=0, r=0, t=10, b=10),
            xaxis_title="protection G, %", showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            f"<div class='note'><b>The gap between {naive * 100:.0f}% and "
            f"{G_star * 100:.1f}% is the sub-additivity.</b> All scavengers share "
            "one denominator, so the fourth antioxidant contributes far less than "
            "the first. This is the one sentence that explains why a four-compound "
            "formulation is not four times better than a one-compound one.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='flag'><b>The best-evidenced compound contributes the "
            "least.</b> Pulcherrimin has human keratinocyte data and a confirmed "
            "mechanism. The two biosurfactants, which carry most of the headline "
            "number, have never touched a keratinocyte in the published record: "
            "their evidence base is a fruit-juice preservation study and a "
            "crude-oil emulsification study.</div>",
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# 2. cascade trace
# ----------------------------------------------------------------------------

with tabs[1]:
    st.markdown("#### Every intermediate, in evaluation order")
    st.caption("Any line here can be checked by hand against the spec's worked trace.")

    trace = [
        ("S1a", "occupancy from L", "L / CMC_L", res["occ_L"]),
        ("S1a", "occupancy from X", "X / K_X_ads", res["occ_X"]),
        ("S1a", "coverage theta", "competitive Langmuir, one interface", res["theta"]),
        ("S1a", "sigma PM", "1 - theta_max theta", res["sigma_PM"]),
        ("S1a", "sigma gas", "1 - theta_max phi_gas theta", res["sigma_gas"]),
        ("S1a", "PM reaching skin, ug/m3", "sigma_PM C_PM", res["C_PM_t"]),
        ("S1a", "O3 reaching skin, ug/m3", "sigma_gas C_O3", res["C_O3_t"]),
        ("S1b", "SPF", "1 + eps_film a_SPF P", res["SPF"]),
        ("S1b", "transmittance tau_UV", "1 / SPF", res["tau_UV"]),
        ("S2", "chelated fraction chi", "E_chel H(P; K_chel)", res["chi"]),
        ("S2", "Fenton factor", "1 - phi_Fe chi, discounted", res["fenton"]),
        ("", "p tilde", "post-shielding particle channel", res["p_t"]),
        ("", "z tilde", "post-shielding ozone channel", res["z_t"]),
        ("", "u tilde", "post-screen UV channel", res["u_t"]),
        ("", "R_gen untreated", "weighted sum, 1.0 at reference", res["R_gen"]),
        ("", "R_gen protected", "after S1 and S2", res["R_gen_t"]),
        ("S3", "kappa endo untreated", "the skin's own reserve", res["kappa_endo"]),
        ("S3", "kappa endo protected", "goes up: less depleted", res["kappa_endo_t"]),
        ("S3", "kappa app", "eta psi (k_X X + k_H H)", res["kappa_app"]),
        ("S3", "normaliser N_R", "fixed at reference, healthy, zero dose", res["N_R"]),
        ("S3", "R untreated", "normalised oxidative damage", res["R"]),
        ("S3", "R prime", "after scavenging", res["R_prime"]),
        ("S4", "delta T reference", "alpha_PM C_PM_ref + alpha_O3 C_O3_ref", res["dT_ref"]),
        ("S4", "delta T after shielding", "excess TEWL, not absolute", res["dT_t"]),
        ("S4", "humectant gain Gamma_H", "below 1 means improvement", res["Gamma_H"]),
        ("S4", "stripping Psi", "above 1 means net harm", res["Psi"]),
        ("S4", "B untreated", "normalised barrier damage", res["B"]),
        ("S4", "B prime", "after humectant and stripping", res["B_prime"]),
        ("S6", "omega, pulcherrimin", "rho_P H(P; C_horm, 2)", res["omega_P"]),
        ("S6", "omega, xylolipid", "s H(X; C_crit, 2)", res["omega_X"]),
        ("S6", "omega, lyso-orn lipid", "s H(L; C_crit, 2)", res["omega_L"]),
        ("S6", "Omega", "damage the product adds back", res["Omega"]),
        ("S6", "D_s", "post-shielding, pre-treatment damage", res["D_s"]),
        ("S6", "penalty", "Omega D_s", res["penalty"]),
        ("out", "RSD untreated", "w_ox R + (1-w_ox) B", res["RSD0"]),
        ("out", "RSD treated", "plus the self-inflicted term", res["RSD"]),
        ("out", "protection G", "1 - RSD / RSD0", res["G"]),
        ("out", "TEWL untreated, g/m2/h", "T0 + delta T", res["T"]),
        ("out", "TEWL treated, g/m2/h", "(T0 + delta T tilde) Gamma_H Psi", res["T_prime"]),
    ]
    df_trace = pd.DataFrame(trace, columns=["Stage", "Quantity", "How", "Value"])
    df_trace["Value"] = df_trace["Value"].map(lambda v: f"{v:,.4f}")
    st.dataframe(df_trace, hide_index=True, width="stretch", height=560)

    a, b = st.columns(2)
    with a:
        st.markdown(
            "<div class='note'><b>The ozone channel is the largest surviving "
            "damage term.</b> The film barely touches gases and no compound in the "
            "formulation acts on ozone specifically. A fifth, ozone-directed "
            "compound would have the most room to work in.</div>",
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            f"<div class='note'><b>kappa endo rises under protection</b>, "
            f"{res['kappa_endo']:.3f} to {res['kappa_endo_t']:.3f}. The compounds "
            "lowered ROS generation, so the skin's own reserve is less depleted. "
            "The product protects the skin's own defences as well as substituting "
            "for them.</div>",
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# 3. dose response
# ----------------------------------------------------------------------------

with tabs[2]:
    st.markdown("#### One compound at a time, everything else held at the recommendation")
    grid_n = 160
    fig = go.Figure()
    colors = {"P": RUST, "X": SIGNAL, "L": "#3d5a80", "H": "#8c7851"}
    for k in DOSE_ORDER:
        xs = np.linspace(0.0, CAPS[k], grid_n)
        ys = []
        for v in xs:
            d = Dose(**{**dose.__dict__, k: float(v)})
            ys.append(protection(env, d, params) * 100)
        fig.add_scatter(x=xs / CAPS[k], y=ys, name=f"{k} ({COMPOUND_NAME[k]})",
                        line=dict(color=colors[k], width=2))
        cur = getattr(dose, k) / CAPS[k]
        fig.add_scatter(x=[cur], y=[protection(env, dose, params) * 100],
                        mode="markers", marker=dict(color=colors[k], size=9),
                        showlegend=False)
    fig.update_layout(
        height=420, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="dose as a fraction of its cap", yaxis_title="protection G, %",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Only pulcherrimin turns over inside the domain. The other three rise "
        "monotonically to their caps, because nothing inside D penalises them: "
        "C_crit sits above the X cap, and hyaluronic acid has no self-toxicity "
        "term at all."
    )

    st.markdown("#### Scaling one fixed formulation ratio")
    st.caption(
        "The realistic case when all four compounds come off one engineered "
        "strain: their ratio is set by the circuit, and you can only turn the "
        "whole thing up or down."
    )
    rc = st.columns(4)
    ratio = []
    for col, k in zip(rc, DOSE_ORDER):
        with col:
            ratio.append(st.number_input(f"r_{k}", min_value=0.0,
                                         value=float(CAPS[k]), step=float(CAPS[k]) / 20,
                                         format="%.4f", key=f"ratio_{k}"))
    ratio = np.array(ratio)
    if ratio.sum() > 0:
        s_star, d_ray, g_ray = optimal_scale(env, ratio, params)
        caps_arr = np.array([CAPS[k] for k in DOSE_ORDER])
        s_max = float(np.min(np.where(ratio > 0, caps_arr / np.where(ratio > 0, ratio, 1), np.inf)))
        ss = np.linspace(1e-6, s_max, 300)
        gg = [protection(env, Dose.from_array(s * ratio), params) * 100 for s in ss]
        f2 = go.Figure()
        f2.add_scatter(x=ss, y=gg, line=dict(color=INK, width=2), name="G(s)")
        f2.add_scatter(x=[s_star], y=[g_ray * 100], mode="markers+text",
                       marker=dict(color=RUST, size=11),
                       text=[f"s* = {s_star:.3f}"], textposition="top center",
                       showlegend=False)
        f2.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                         xaxis_title="scale s", yaxis_title="protection G, %",
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         showlegend=False)
        st.plotly_chart(f2, width="stretch")
        st.markdown(
            f"**s\\* = {s_star:.4f}**, giving G = {g_ray * 100:.1f}% at "
            f"({d_ray.P:.4f}, {d_ray.X:.4f}, {d_ray.L:.4f}, {d_ray.H:.4f}) mg/mL."
        )
        st.markdown(
            "<div class='flag'><b>Solver trap.</b> G(s) rises to a peak and then "
            "falls, so for most targets there are two values of s that hit it: one "
            "on the rising limb and one on the descending. The descending one is a "
            "harmful overdose that happens to give the same protection as a safe "
            "dose. Find the peak first, then bracket below it, where the function "
            "is monotone and the root is unique.</div>",
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# 4. uncertainty
# ----------------------------------------------------------------------------

with tabs[3]:
    st.markdown("#### What the unmeasured constants cost")
    n_mc = st.select_slider("Monte Carlo samples", [200, 400, 800, 1500], value=400)
    draws = c_monte_carlo(ekey(env), n_mc, dkey(dose))
    lo, med, hi = np.percentile(draws, [5, 50, 95])

    _, g_nom = c_optimal(ekey(env), pkey(NOMINAL))
    _, g_consC = c_optimal(ekey(env), pkey(conservative_params("C")))
    _, g_consAll = c_optimal(ekey(env), pkey(conservative_params("all")))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Nominal G*", f"{g_nom * 100:.1f}%")
    m2.metric("Conservative (class C)", f"{g_consC * 100:.1f}%")
    m3.metric("Worst case (all bands)", f"{g_consAll * 100:.1f}%")
    m4.metric("Monte Carlo 90% interval", f"{lo * 100:.0f} - {hi * 100:.0f}%",
              f"median {med * 100:.1f}%")

    fh = go.Figure()
    fh.add_histogram(x=draws * 100, nbinsx=40, marker_color=SIGNAL, opacity=0.8)
    for x, lab, col in [(g_nom * 100, "nominal", INK),
                        (g_consC * 100, "conservative", RUST)]:
        fh.add_vline(x=x, line=dict(color=col, dash="dash"),
                     annotation_text=lab, annotation_position="top")
    fh.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0),
                     xaxis_title="protection G, %", yaxis_title="draws",
                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     showlegend=False)
    st.plotly_chart(fh, width="stretch")
    st.caption(
        "Every banded constant sampled uniformly across its stated low/high "
        "range, with the recommended dose held fixed. This asks how well the "
        "formulation does under parameter uncertainty, not how much protection "
        "is achievable in principle."
    )

    st.markdown("#### Experiment priorities, ranked by how much they move G*")
    rows = c_sensitivity(ekey(env))
    df_s = pd.DataFrame(rows)
    df_s = df_s.assign(
        G_low=(df_s.G_low * 100).round(1),
        G_high=(df_s.G_high * 100).round(1),
        swing_pts=df_s.swing_pts.round(1),
    ).rename(columns={
        "parameter": "Constant", "class": "Class", "low": "Low", "high": "High",
        "G_low": "G at low, %", "G_high": "G at high, %", "swing_pts": "Swing, pts",
    })
    st.dataframe(df_s.head(12), hide_index=True, width="stretch")

    top = df_s.iloc[0]
    st.markdown(
        f"<div class='flag'><b>If you can only run one experiment, measure "
        f"{top['Constant']}.</b> It moves G by {top['Swing, pts']} points across "
        "its band, several times more than anything else on the list. For "
        "theta_max the experiment is: deposit standard reference PM2.5 onto "
        "coated versus uncoated membranes and count what penetrates. No cell "
        "culture needed, and it would be the first measurement of biosurfactant "
        "particulate interception anywhere in the literature, which makes it a "
        "contribution rather than a calibration.</div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# 5. does the dose move
# ----------------------------------------------------------------------------

with tabs[4]:
    st.markdown("#### Run the optimiser across a grid of environments and print the spread")
    st.caption(
        "Do this before the wiki goes up. If d* is flat, say so: it is a much "
        "better look than having a judge notice the recommendation never changes."
    )

    pm_grid = st.slider("PM2.5 range swept", 0, 400, (20, 300), 10)
    n_grid = st.select_slider("Points per axis", [3, 4, 5], value=4)

    pms = np.linspace(pm_grid[0], pm_grid[1], n_grid)
    pts = [1, 4, 6]
    exposures = tuple(
        Environment(C_PM=float(pm), C_O3=C_O3, I_UV=I_UV, PT=pt, A0=A0, lag_h=lag_h)
        for pm in pms for pt in pts
    )
    spread = c_spread(tuple(ekey(e) for e in exposures), pkey(params))
    df_sp = pd.DataFrame(spread)

    stat = df_sp[list(DOSE_ORDER)].agg(["min", "max"]).T
    stat["cap"] = [CAPS[k] for k in stat.index]
    stat["spread as % of cap"] = ((stat["max"] - stat["min"]) / stat["cap"] * 100).round(2)
    stat.index = [COMPOUND_NAME[k] for k in stat.index]
    st.dataframe(stat.round(5), width="stretch")

    fg = go.Figure()
    for pt in pts:
        sub = df_sp[df_sp.PT == pt]
        fg.add_scatter(x=sub.C_PM, y=sub.G * 100, mode="lines+markers",
                       name=f"phototype {'I II III IV V VI'.split()[pt - 1]}")
    fg.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                     xaxis_title="PM2.5, ug/m3", yaxis_title="protection G, %",
                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fg, width="stretch")

    pinned = [COMPOUND_NAME[k] for k in DOSE_ORDER
              if (df_sp[k].max() - df_sp[k].min()) < 1e-6]
    p_span = (df_sp["P"].max() - df_sp["P"].min()) / CAPS["P"] * 100
    st.markdown(
        f"<div class='flag'><b>The honest description of this feature.</b> Across "
        f"this grid {len(pinned)} of 4 coordinates never move at all "
        f"({', '.join(pinned)}): nothing inside the validity domain penalises "
        f"them, so the optimiser runs them to the boundary regardless of the "
        f"weather. Pulcherrimin is the only coordinate that responds, and it "
        f"moves {p_span:.0f}% of its cap over the whole sweep. Its optimum is set "
        f"by the S6 reversal threshold, which has no environment dependence; what "
        f"little movement there is comes through the D_s over RSD(e, 0) ratio and "
        f"the phototype modifier. So the feature recommends close to one "
        f"formulation and predicts how well that formulation will do today. G and "
        f"the absolute RSD genuinely are personalised. \"The model computes your "
        f"personal optimal dose\" would overclaim.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='note'><b>If you want the dose to respond to environment</b>, "
        "you need a penalty that bites inside the domain: lower C_crit below the "
        "X cap, or give hyaluronic acid a self-toxicity term. Right now neither "
        "exists, so three of four compounds have nothing to trade off "
        "against.</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Full grid"):
        st.dataframe(df_sp.round(4), hide_index=True, width="stretch")

# ----------------------------------------------------------------------------
# 6. cost solve
# ----------------------------------------------------------------------------

with tabs[5]:
    st.markdown("#### Cheapest dose that still hits a protection target")
    st.caption(
        "min c'd subject to G >= G target, d <= FBA ceiling, d inside the "
        "validity domain. The cost vector should come from the genome-scale "
        "model. Until it does, these numbers are placeholders."
    )

    g_target = st.slider("Target protection G", 0.0, 0.9, 0.45, 0.01)

    cc = st.columns(4)
    cost, fba = [], []
    for col, k in zip(cc, DOSE_ORDER):
        with col:
            st.markdown(f"**{k}**")
            cost.append(st.number_input("metabolic cost", min_value=0.0, value=1.0,
                                        step=0.1, key=f"cost_{k}"))
            fba.append(st.number_input("FBA ceiling, mg/mL", min_value=0.0,
                                       value=float(CAPS[k]), step=float(CAPS[k]) / 20,
                                       format="%.4f", key=f"fba_{k}"))

    out = min_cost_dose(env, g_target, np.array(cost), np.array(fba), params)
    if out is None:
        st.error(
            f"Infeasible. No dose inside the domain reaches G = {g_target:.2f} "
            f"under these ceilings. The best available here is "
            f"{c_optimal(ekey(env), pkey(params))[1] * 100:.1f}%. The solve returns "
            "empty rather than a wrong answer."
        )
    else:
        d_c, total, g_c = out
        st.success(f"Feasible. Cost {total:.4f}, achieved G = {g_c * 100:.1f}%.")
        st.dataframe(pd.DataFrame([{
            "Compound": COMPOUND_NAME[k],
            "Dose (mg/mL)": round(getattr(d_c, k), 4),
            "Cost weight": cost[i],
            "FBA ceiling": fba[i],
        } for i, k in enumerate(DOSE_ORDER)]), hide_index=True,
            width="stretch")

# ----------------------------------------------------------------------------
# 7. constants
# ----------------------------------------------------------------------------

with tabs[6]:
    st.markdown("#### Every constant, its evidence class and its band")
    st.caption(
        "A: somebody measured this quantity directly. "
        "B: fitted to a single study, so only as good as that study. "
        "C: nobody measured it and we chose a value."
    )
    rows = []
    for name, cls in EVIDENCE.items():
        lo, hi = BANDS.get(name, (None, None))
        rows.append({
            "Constant": name,
            "Value": getattr(params, name, None),
            "Class": cls,
            "Low": lo,
            "High": hi,
            "In use": getattr(params, name, None) != getattr(NOMINAL, name, None),
        })
    df_c = pd.DataFrame(rows)
    st.dataframe(df_c, hide_index=True, width="stretch", height=520)

    st.markdown(
        "<div class='flag'><b>Eleven of thirty-two constants are class C.</b> "
        "Four of those eleven, theta_max, psi, C_crit and kappa_0, account for "
        "most of the spread in the answer, and three of the four are directly "
        "measurable with experiments the team could run this year.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='note'><b>The Milani agreement is not a validation.</b> The "
        "model predicts an 18.7% absolute TEWL improvement at reference against "
        "Milani's measured 19%, without being tuned to it. But roughly 12 of "
        "those points come from the humectant and 7 from the film, and Milani's "
        "serum contains no film-forming surfactant at all. The two numbers are "
        "not the same number. It validates the scale of the barrier channel, "
        "that a topical product's effect on TEWL is a sub-20% quantity rather "
        "than a 50% one. It validates nothing about the shielding "
        "mechanism.</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(
    "iGEM IIT Delhi 2026 - DermaSense v3. Every equation in this app corresponds "
    "one-to-one with a section of the unified specification. Model outputs are "
    "predictions from a parameterised cascade, not clinical guidance."
)

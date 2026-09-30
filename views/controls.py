"""
Lab vs controls.

The model already carried one control: hyaluronic acid, the known comparator on
the antioxidant and humectant side. This page adds the matching control on the
biosurfactant side and runs every arm head to head through identical equations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dermasense_model as dm
import ui
from dermasense_model import Dose, Environment, NOMINAL, benchmark_params
from ui import COLOR, INK, MUTED, RUST, SIGNAL

ui.eyebrow("controls")
st.title("The lab's compounds against known comparators")
st.markdown(
    f"<span style='color:{MUTED}'>An effect is only worth reporting next to something "
    "already known to work. The specification benchmarks pulcherrimin against "
    "hyaluronic acid. Nothing benchmarked the two biosurfactants, so this page adds "
    "that control and puts every arm through the same S1a, S4 and S6 terms.</span>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# the pairing, stated before any chart
# ----------------------------------------------------------------------------

p1, p2 = st.columns(2)
with p1:
    st.markdown(
        f"""
        <div class="card" style="border-top:3px solid {COLOR['H']}">
          <div style="font-weight:700">Antioxidant arm &mdash; already controlled</div>
          <div style="font-size:.88rem;color:#3b4441;line-height:1.55;margin-top:.4rem">
          <b>Pulcherrimin</b> against <b>hyaluronic acid</b>. Both are scored through
          S3 competitive scavenging with one shared denominator, and hyaluronic acid
          additionally holds water through S4. Its scavenging rate constant is class A,
          measured; pulcherrimin's contribution enters upstream at S2 instead, as iron
          chelation.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with p2:
    st.markdown(
        f"""
        <div class="card" style="border-top:3px solid {COLOR['S']}">
          <div style="font-weight:700">Biosurfactant arm &mdash; the control added here</div>
          <div style="font-size:.88rem;color:#3b4441;line-height:1.55;margin-top:.4rem">
          <b>Xylolipid + lyso-ornithine lipid</b> against a <b>benchmark surfactant</b>
          occupying the same interface. It enters the same competitive Langmuir term
          for coverage, the same stripping term in S4 and the same self-toxicity term
          in S6. Only its three constants differ.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ----------------------------------------------------------------------------
# benchmark picker
# ----------------------------------------------------------------------------

bench_keys = list(dm.BENCHMARKS)
choice = st.radio(
    "Benchmark in the S slot",
    bench_keys,
    index=bench_keys.index(ui.benchmark_key()),
    horizontal=True,
    format_func=lambda k: f"{dm.BENCHMARKS[k].name} ({dm.BENCHMARKS[k].role})",
    key="benchmark",
)
B = dm.BENCHMARKS[choice]
params = benchmark_params(choice)

bc1, bc2 = st.columns([1.6, 1])
with bc1:
    ui.note(f"<b>{B.name}.</b> {B.note}")
with bc2:
    st.markdown(
        f"""
        <div class="card">
          <div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;
            color:{MUTED}">its three constants</div>
          <div style="font-family:monospace;font-size:.84rem;line-height:1.7;margin-top:.4rem">
          CMC_S &nbsp;&nbsp;&nbsp;{B.CMC_S:>6.2f} mg/mL<br>
          C_crit_S {B.C_crit_S:>6.2f} mg/mL<br>
          strip_S &nbsp;{B.strip_S:>6.2f}<br>
          S_cap &nbsp;&nbsp;&nbsp;{B.S_cap:>6.2f} mg/mL</div>
          <div style="font-size:.74rem;color:{MUTED};margin-top:.5rem">{B.evidence}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.expander("Where these constants come from"):
    for s in B.sources:
        st.markdown(f"- {s}")
    st.caption(
        "The stripping threshold and slope are deliberately set equal to ours "
        "wherever nobody measured them separately. A control tuned to lose is not a "
        "control, so where the evidence is silent the two arms get the same number."
    )

# ----------------------------------------------------------------------------
# the day
# ----------------------------------------------------------------------------

env = ui.shared_env()
with st.expander(
    f"Day being compared: PM {env.C_PM:.0f}, ozone {env.C_O3:.0f}, "
    f"I_UV {env.I_UV:.2f}, phototype {'I II III IV V VI'.split()[env.PT - 1]}",
    expanded=False,
):
    st.caption(
        "This is whatever the dose simulator last resolved, so a live city carries "
        "across pages. Pick a different day here to compare on it instead."
    )
    preset = st.selectbox("Use a preset day instead", ["Keep the current day"]
                          + list(ui.DAY_PRESETS), key="ctl_preset")
    if preset != "Keep the current day":
        b = ui.DAY_PRESETS[preset]
        env = Environment(C_PM=b["C_PM"], C_O3=b["C_O3"], I_UV=b["I_UV"],
                          PT=env.PT, A0=env.A0, lag_h=env.lag_h)

rows = ui.c_controls(ui.ekey(env), ui.pkey(params))
by_arm = {r["arm"]: r for r in rows}

st.divider()

# ----------------------------------------------------------------------------
# 1. head to head
# ----------------------------------------------------------------------------

st.subheader("Every arm, each optimised on its own")
st.markdown(
    f"<span style='color:{MUTED}'>Each bar is the best that arm can do inside its own "
    "evidence caps, not an arbitrary dose. Lab compounds in solid colour, controls "
    "hatched.</span>",
    unsafe_allow_html=True,
)

ARM_TONE = {
    "Lab antioxidant (P)": COLOR["P"],
    "Control antioxidant (H)": COLOR["H"],
    "Lab biosurfactants (X + L)": COLOR["X"],
    "Control biosurfactant (S)": COLOR["S"],
    "Lab formulation (P + X + L)": "#2f6f4f",
    "Controls only (H + S)": "#8e7cab",
    "Spec formulation (P + X + L + H)": INK,
    "Everything (P + X + L + H + S)": "#4a5b57",
}

fig = go.Figure()
labels = [r["arm"] for r in rows]
fig.add_bar(
    x=[r["G"] * 100 for r in rows],
    y=labels,
    orientation="h",
    marker=dict(
        color=[ARM_TONE[r["arm"]] for r in rows],
        pattern=dict(shape=["/" if "Control" in r["arm"] else "" for r in rows],
                     fgcolor="white", size=5),
    ),
    text=[f"{r['G'] * 100:.1f}%" for r in rows],
    textposition="auto",
    customdata=[[r["mass_mg_per_mL"], r["theta"], r["TEWL_improvement"] * 100] for r in rows],
    hovertemplate=("%{y}<br>G = %{x:.1f}%<br>total mass %{customdata[0]:.3f} mg/mL"
                   "<br>coverage %{customdata[1]:.2f}"
                   "<br>TEWL improvement %{customdata[2]:.1f}%<extra></extra>"),
)
ui.style(fig, 420, showlegend=False,
         xaxis=dict(title="protection G, %"),
         yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, width="stretch", key="ctl_bars", config={"displayModeBar": False})

lab_bs = by_arm["Lab biosurfactants (X + L)"]
ctl_bs = by_arm["Control biosurfactant (S)"]
lab_ao = by_arm["Lab antioxidant (P)"]
ctl_ao = by_arm["Control antioxidant (H)"]

m1, m2, m3 = st.columns(3)
m1.metric("Biosurfactants: ours vs the control",
          f"{lab_bs['G'] * 100:.1f}% vs {ctl_bs['G'] * 100:.1f}%",
          f"{(lab_bs['G'] - ctl_bs['G']) * 100:+.1f} points")
m2.metric("Antioxidant: ours vs the control",
          f"{lab_ao['G'] * 100:.1f}% vs {ctl_ao['G'] * 100:.1f}%",
          f"{(lab_ao['G'] - ctl_ao['G']) * 100:+.1f} points")
m3.metric("Whole lab formulation vs controls only",
          f"{by_arm['Lab formulation (P + X + L)']['G'] * 100:.1f}% vs "
          f"{by_arm['Controls only (H + S)']['G'] * 100:.1f}%",
          f"{(by_arm['Lab formulation (P + X + L)']['G'] - by_arm['Controls only (H + S)']['G']) * 100:+.1f} points")

if lab_bs["G"] > ctl_bs["G"]:
    ui.note(
        f"<b>On this day the lab pair beats the {B.name.lower()} by "
        f"{(lab_bs['G'] - ctl_bs['G']) * 100:.1f} points of G.</b> Read the next "
        "section before quoting that: almost all of the gap is mass efficiency, "
        "which is a property of the Langmuir constants, and two of the three "
        "constants involved are ours rather than measured."
    )
else:
    ui.flag(
        f"<b>On this day the {B.name.lower()} matches or beats the lab pair.</b> "
        "That is the control doing its job. It is a result worth reporting either "
        "way, and it is the reason the arm exists."
    )

st.divider()

# ----------------------------------------------------------------------------
# 2. mass efficiency, which is where the difference actually lives
# ----------------------------------------------------------------------------

st.subheader("Where the difference comes from: mass, not mechanism")
st.markdown(
    f"<span style='color:{MUTED}'>All three film formers do the same thing at the same "
    "interface. They differ in how much you need. The half-coverage constant is the "
    "whole story, and it spans four orders of magnitude.</span>",
    unsafe_allow_html=True,
)

halves = dm.half_coverage_dose(params)
cc1, cc2 = st.columns([1.4, 1])

with cc1:
    cf = go.Figure()
    grid = np.logspace(-5, 1.2, 400)
    for k in ("L", "X", "S"):
        cf.add_scatter(
            x=grid, y=dm.coverage_alone(k, grid, params) * 100,
            name=ui.compound_label(k), mode="lines",
            line=dict(color=COLOR[k], width=2.4,
                      dash="dot" if k == "S" else "solid"),
            hovertemplate="%{fullData.name}<br>%{x:.4f} mg/mL<br>"
                          "%{y:.0f}% coverage<extra></extra>",
        )
        cf.add_scatter(x=[halves[k]], y=[50], mode="markers",
                       marker=dict(color=COLOR[k], size=9, symbol="diamond"),
                       showlegend=False, hoverinfo="skip")
    cf.add_hline(y=50, line=dict(color=MUTED, dash="dash", width=1),
                 annotation_text="half the interface covered",
                 annotation_position="top left",
                 annotation_font=dict(size=11, color=MUTED))
    ui.style(cf, 380,
             xaxis=dict(title="concentration, mg/mL (log scale)", type="log"),
             yaxis=dict(title="interface coverage, %", range=[0, 102]),
             legend=dict(orientation="h", y=-0.22))
    st.plotly_chart(cf, width="stretch", key="ctl_coverage",
                    config={"displayModeBar": False})

with cc2:
    eq = dm.control_equivalent_dose(params)
    tbl = pd.DataFrame([
        {"Film former": ui.compound_label(k),
         "Half-coverage, mg/mL": halves[k],
         "vs lyso-orn lipid": f"{halves[k] / halves['L']:.0f}x"}
        for k in ("L", "X", "S")
    ])
    st.dataframe(tbl, hide_index=True, width="stretch")
    st.markdown(
        f"""
**Matched coverage.** At their caps the lab pair covers
**{eq['theta_lab'] * 100:.1f}%** of the interface using **{eq['lab_mass']:.3f} mg/mL**
of material. Reproducing that coverage with {B.name.lower()} alone takes
**{eq['S_equivalent']:.2f} mg/mL**, about **{eq['mass_ratio']:.0f}x the mass**
{'&mdash; which is above its own safe-use cap, so it cannot get there at all.'
 if not eq['within_S_cap'] else '&mdash; still inside its safe-use cap.'}
        """,
        unsafe_allow_html=True,
    )

ui.flag(
    "<b>This comparison is only as good as K_X_ads and CMC_S.</b> The "
    "lyso-ornithine lipid constant is class A, measured. The xylolipid one is "
    "inferred, class C. If the real xylolipid constant is ten times larger, the "
    "curve above slides right and the mass advantage shrinks with it. Measuring "
    "the two lab surfactants' CMCs is a single afternoon of tensiometry, and it "
    "would turn this chart from an argument into a result."
)

st.divider()

# ----------------------------------------------------------------------------
# 3. what each arm costs the skin
# ----------------------------------------------------------------------------

st.subheader("What each arm costs the skin")
st.markdown(
    f"<span style='color:{MUTED}'>Protection is not the only axis. S4 says a surfactant "
    "above its stripping threshold makes water loss worse, and S6 says a dose can "
    "harm the tissue it was applied to. A good control makes those columns "
    "visible.</span>",
    unsafe_allow_html=True,
)

cost_rows = []
for r in rows:
    cost_rows.append({
        "Arm": r["arm"],
        "G, %": r["G"] * 100,
        "Damage RSD": r["RSD"],
        "TEWL improvement, %": r["TEWL_improvement"] * 100,
        "Self-inflicted, Omega": r["Omega"],
        "Total mass, mg/mL": r["mass_mg_per_mL"],
        "G per mg/mL": r["G"] / r["mass_mg_per_mL"] if r["mass_mg_per_mL"] > 0 else np.nan,
    })
df = pd.DataFrame(cost_rows)
st.dataframe(
    df, hide_index=True, width="stretch",
    column_config={
        "G, %": st.column_config.ProgressColumn(
            "G, %", format="%.1f%%", min_value=0.0,
            max_value=float(max(df["G, %"].max(), 1.0))),
        "Damage RSD": st.column_config.NumberColumn(format="%.3f"),
        "TEWL improvement, %": st.column_config.NumberColumn(format="%.1f"),
        "Self-inflicted, Omega": st.column_config.NumberColumn(format="%.4f"),
        "Total mass, mg/mL": st.column_config.NumberColumn(format="%.3f"),
        "G per mg/mL": st.column_config.NumberColumn(
            "G per mg/mL", format="%.2f",
            help="Protection bought per unit of material applied."),
    },
)

sc = go.Figure()
# Two arms land on almost the same point (the control adds little once the film
# is saturated), so labels alternate above and below rather than overlapping.
for i, r in enumerate(sorted(rows, key=lambda r: r["mass_mg_per_mL"])):
    sc.add_scatter(
        x=[r["mass_mg_per_mL"]], y=[r["G"] * 100],
        mode="markers+text",
        marker=dict(size=13 + 220 * r["Omega"], color=ARM_TONE[r["arm"]],
                    line=dict(color="white", width=1.5)),
        text=[r["arm"].replace(" (", "<br>(")],
        textposition="top center" if i % 2 == 0 else "bottom center",
        textfont=dict(size=10, color=MUTED),
        name=r["arm"], showlegend=False,
        hovertemplate=f"{r['arm']}<br>mass %{{x:.3f}} mg/mL<br>G %{{y:.1f}}%"
                      f"<br>Omega {r['Omega']:.4f}<extra></extra>",
    )
ui.style(sc, 440,
         xaxis=dict(title="total applied mass, mg/mL", range=[-0.25, 1.95]),
         yaxis=dict(title="protection G, %", range=[-4, 72]))
st.plotly_chart(sc, width="stretch", key="ctl_scatter",
                config={"displayModeBar": False})
st.caption(
    "Up and to the left is better: more protection for less material. Marker size "
    "is the self-inflicted damage term, so a large marker high on the chart is still "
    "paying for its protection."
)

st.divider()

# ----------------------------------------------------------------------------
# 4. is the control's margin robust
# ----------------------------------------------------------------------------

st.subheader("Does the result survive being wrong about the constants?")

_kx_lo, _kx_hi = dm.BANDS["K_X_ads"]
sweep = st.slider(
    "Sweep the xylolipid half-coverage constant K_X_ads across its band",
    float(_kx_lo), float(_kx_hi), float(NOMINAL.K_X_ads), 0.005,
    format="%.3f", key="ctl_kx",
    help="Class C: nobody measured it. The mass-efficiency argument on this page "
         "rests on it, so the honest thing is to let a reader move it.",
)
sw_params = benchmark_params(choice, NOMINAL.with_values(K_X_ads=float(sweep)))
sw_lab = dm.optimal_dose(env, sw_params, ("X", "L"))[1]
sw_ctl = dm.optimal_dose(env, sw_params, ("S",))[1]

s1, s2, s3 = st.columns(3)
s1.metric("Lab pair, G", f"{sw_lab * 100:.1f}%",
          f"{(sw_lab - lab_bs['G']) * 100:+.1f} vs nominal")
s2.metric(f"{B.name}, G", f"{sw_ctl * 100:.1f}%")
s3.metric("Margin", f"{(sw_lab - sw_ctl) * 100:+.1f} points",
          "negative means the control wins" if sw_lab < sw_ctl else "lab pair ahead")

band = np.linspace(*dm.BANDS["K_X_ads"], 24)
margin = []
for v in band:
    pv = benchmark_params(choice, NOMINAL.with_values(K_X_ads=float(v)))
    margin.append((dm.optimal_dose(env, pv, ("X", "L"))[1]
                   - dm.optimal_dose(env, pv, ("S",))[1]) * 100)
mf = go.Figure()
mf.add_scatter(x=band, y=margin, mode="lines", line=dict(color=INK, width=2.4),
               name="margin")
mf.add_hline(y=0, line=dict(color=RUST, dash="dash"),
             annotation_text="control wins below this line",
             annotation_font=dict(size=11, color=RUST))
mf.add_vline(x=NOMINAL.K_X_ads, line=dict(color=SIGNAL, dash="dot"),
             annotation_text="value we chose", annotation_font=dict(size=11, color=SIGNAL))
ui.style(mf, 320, showlegend=False,
         xaxis=dict(title="K_X_ads, mg/mL"),
         yaxis=dict(title="lab pair minus control, points of G"))
st.plotly_chart(mf, width="stretch", key="ctl_margin",
                config={"displayModeBar": False})

crosses = any(m <= 0 for m in margin)
if crosses:
    ui.flag(
        "<b>The margin crosses zero inside the band.</b> There is a value of "
        "K_X_ads, well inside what the evidence allows, at which the control wins. "
        "Until that constant is measured, the honest claim is that the lab pair is "
        "competitive with an established cosmetic biosurfactant, not that it beats it."
    )
else:
    ui.note(
        "<b>The margin never reaches zero across the band.</b> The lab pair stays "
        "ahead of this control for every value of K_X_ads the evidence allows, which "
        "is the strongest form this claim can take while the constant is unmeasured."
    )

st.divider()

# ----------------------------------------------------------------------------
# 5. what to actually measure
# ----------------------------------------------------------------------------

st.subheader("The experiments that would settle this")

e1, e2, e3 = st.columns(3)
with e1:
    st.markdown(
        f"""
        <div class="card" style="border-top:3px solid {SIGNAL}">
        <b>1. Tensiometry, both lab surfactants</b>
        <div style="font-size:.87rem;color:#3b4441;line-height:1.5;margin-top:.4rem">
        A du Nouy ring and a dilution series gives CMC directly. It replaces the one
        class C constant that the entire mass-efficiency argument on this page rests
        on. Half a day.</div></div>
        """, unsafe_allow_html=True)
with e2:
    st.markdown(
        f"""
        <div class="card" style="border-top:3px solid {SIGNAL}">
        <b>2. Particle interception, side by side</b>
        <div style="font-size:.87rem;color:#3b4441;line-height:1.5;margin-top:.4rem">
        Reference PM2.5 onto coated and uncoated membranes, count what penetrates.
        Measures theta_max, the biggest lever in the model, and does it for our
        surfactants and the control in the same run.</div></div>
        """, unsafe_allow_html=True)
with e3:
    st.markdown(
        f"""
        <div class="card" style="border-top:3px solid {SIGNAL}">
        <b>3. Keratinocyte viability, matched concentrations</b>
        <div style="font-size:.87rem;color:#3b4441;line-height:1.5;margin-top:.4rem">
        The control has this data and our compounds do not, which is the real
        asymmetry between the arms. An MTT series against the same cells at the same
        concentrations closes it.</div></div>
        """, unsafe_allow_html=True)

st.caption(
    "iGEM IIT Delhi 2026 - DermaSense. The control arm runs through identical "
    "equations; only its four constants differ, and each is labelled with its "
    "evidence class."
)

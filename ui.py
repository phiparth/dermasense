"""
Shared look, cached model calls and small widgets for every page.

No page logic lives here. Anything a page computes that another page wants to
reuse (the resolved environment, the chosen benchmark) goes through
st.session_state, set by the page that owns it.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import dermasense_model as dm
from dermasense_model import Dose, Environment, Params

# ----------------------------------------------------------------------------
# palette
# ----------------------------------------------------------------------------

INK = "#1c2321"
PAPER = "#f7f5f0"
SIGNAL = "#0f6f68"      # ozone teal, the channel nothing in the formulation touches
RUST = "#a8492c"        # the spec shades unmeasured constants rust, so do we
MUTED = "#6c757d"
LINE = "#e3ded4"

# pulcherrimin is a brick-red pigment, so it gets the red
COLOR = {
    "P": "#b23a2a",
    "X": "#0f6f68",
    "L": "#3d5a80",
    "H": "#b08a3e",
    "S": "#7a5c99",
}
CHANNEL_COLOR = {"PM": "#5b5f66", "O3": SIGNAL, "UV": "#7b4fc9"}

CSS = f"""
<style>
  .stApp {{ background: {PAPER}; }}
  html, body, [class*="css"] {{ color: {INK}; }}
  h1, h2, h3 {{ letter-spacing: -0.01em; }}
  .block-container {{ padding-top: 2.2rem; }}
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
    background: #ffffff; border: 1px solid {LINE}; border-radius: 10px;
    padding: 0.9rem 1.1rem; height: 100%;
  }}
  .flag {{
    border-left: 3px solid {RUST}; background: #fff8f5;
    padding: 0.7rem 0.9rem; margin: 0.4rem 0; font-size: 0.9rem;
    border-radius: 0 6px 6px 0;
  }}
  .note {{
    border-left: 3px solid {SIGNAL}; background: #f1f8f7;
    padding: 0.7rem 0.9rem; margin: 0.4rem 0; font-size: 0.9rem;
    border-radius: 0 6px 6px 0;
  }}
  .eyebrow {{
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.14em;
    color: {SIGNAL}; font-weight: 600; margin-bottom: 0.2rem;
  }}
  .chip {{
    display: inline-block; padding: 0.12rem 0.55rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
    border: 1px solid currentColor; margin-right: 0.3rem;
  }}
  div[data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
  /* The readouts are monospace and wide. At phone width they would otherwise
     push the whole page sideways, so they shrink and wrap instead. */
  @media (max-width: 640px) {{
    .readout {{ font-size: 1.5rem; word-break: break-word; }}
    .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
  }}
  .stApp {{ overflow-x: hidden; }}
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def readout(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="card"><div class="readout-label">{label}</div>'
        f'<div class="readout">{value}</div>'
        f'<div class="readout-sub">{sub}</div></div>'
    )


def flag(html: str) -> None:
    st.markdown(f'<div class="flag">{html}</div>', unsafe_allow_html=True)


def note(html: str) -> None:
    st.markdown(f'<div class="note">{html}</div>', unsafe_allow_html=True)


def eyebrow(text: str) -> None:
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 360, **kw) -> go.Figure:
    """One plotly look for the whole site."""
    fig.update_layout(
        height=height,
        margin=kw.pop("margin", dict(l=0, r=0, t=30, b=0)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=13),
        hoverlabel=dict(bgcolor="white", font_size=12),
        **kw,
    )
    fig.update_xaxes(gridcolor="#ebe6dc", zerolinecolor="#d9d3c7")
    fig.update_yaxes(gridcolor="#ebe6dc", zerolinecolor="#d9d3c7")
    return fig


def compound_label(k: str, p: Params | None = None) -> str:
    """Display name, with the benchmark's real name in the S slot."""
    if k == "S":
        return f"{benchmark().name} (control)"
    if k == "H":
        return "Hyaluronic acid (control)"
    return dm.COMPOUND_NAME[k]


# ----------------------------------------------------------------------------
# cross-page state
# ----------------------------------------------------------------------------

def benchmark_key() -> str:
    return st.session_state.get("benchmark", dm.DEFAULT_BENCHMARK)


def benchmark() -> dm.Benchmark:
    return dm.BENCHMARKS[benchmark_key()]


def shared_env() -> Environment:
    """The day and skin the simulator last resolved, or the reference day."""
    k = st.session_state.get("env_tuple")
    return Environment(*k) if k else Environment()


def remember_env(e: Environment) -> None:
    st.session_state["env_tuple"] = dataclasses.astuple(e)


# ----------------------------------------------------------------------------
# cached wrappers
# ----------------------------------------------------------------------------
# Streamlit hashes cache keys, so plain tuples cross the cache boundary and the
# dataclasses are rebuilt inside.

def ekey(e: Environment):
    return dataclasses.astuple(e)


def pkey(p: Params):
    return dataclasses.astuple(p)


def dkey(d: Dose):
    return dataclasses.astuple(d)


@st.cache_data(show_spinner=False)
def c_optimal(e_k, p_k, keys=dm.DOSE_ORDER):
    d, g = dm.optimal_dose(Environment(*e_k), Params(*p_k), tuple(keys))
    return dkey(d), g


@st.cache_data(show_spinner=False)
def c_evaluate(e_k, d_k, p_k):
    return dm.evaluate(Environment(*e_k), Dose(*d_k), Params(*p_k))


@st.cache_data(show_spinner=False)
def c_sensitivity(e_k):
    return dm.sensitivity(Environment(*e_k))


@st.cache_data(show_spinner=False)
def c_monte_carlo(e_k, n, d_k):
    return dm.monte_carlo(Environment(*e_k), n=n, d_fixed=Dose(*d_k))


@st.cache_data(show_spinner=False)
def c_singles(e_k, p_k):
    e, p = Environment(*e_k), Params(*p_k)
    return dm.single_compound_optima(e, p), dm.per_compound_optima(e, p)


@st.cache_data(show_spinner=False)
def c_spread(e_keys, p_k):
    return dm.dose_spread([Environment(*k) for k in e_keys], Params(*p_k))


@st.cache_data(show_spinner=False)
def c_controls(e_k, p_k):
    return dm.control_comparison(Environment(*e_k), Params(*p_k))


def waterfall_steps(e: Environment, d: Dose, p: Params):
    """RSD as compounds are added one mechanism group at a time: film formers
    first (they act furthest upstream), then pulcherrimin, then the two
    controls. Order matters for the bar heights, never for the end point."""
    groups = [
        ("Film: xylolipid + LOL", ("X", "L")),
        ("Pulcherrimin", ("P",)),
        ("Hyaluronic acid", ("H",)),
        ("Control biosurfactant", ("S",)),
    ]
    on: dict = {}
    prev = dm.evaluate(e, dm.ZERO_DOSE, p)["RSD"]
    rsd0 = prev
    steps = []
    for label, keys in groups:
        if all(getattr(d, k) == 0 for k in keys):
            continue
        for k in keys:
            on[k] = getattr(d, k)
        cur = dm.evaluate(e, Dose(**on), p)["RSD"]
        steps.append((label, cur - prev))
        prev = cur
    return rsd0, steps, prev


def waterfall_figure(e: Environment, d: Dose, p: Params, height: int = 330) -> go.Figure:
    rsd0, steps, final = waterfall_steps(e, d, p)
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"] + ["relative"] * len(steps) + ["total"],
        x=["Untreated"] + [s[0] for s in steps] + ["Treated"],
        y=[rsd0] + [s[1] for s in steps] + [final],
        text=[f"{rsd0:.2f}"] + [f"{s[1]:+.2f}" for s in steps] + [f"{final:.2f}"],
        textposition="outside",
        connector=dict(line=dict(color="#c9c2b4", dash="dot")),
        decreasing=dict(marker=dict(color=SIGNAL)),
        increasing=dict(marker=dict(color=RUST)),
        totals=dict(marker=dict(color=INK)),
    ))
    top = max(rsd0, final) * 1.25 + 0.05
    style(fig, height, showlegend=False, yaxis=dict(title="residual skin damage, RSD", range=[0, top]))
    return fig


def gauge_figure(G: float, height: int = 250) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=G * 100,
        number=dict(suffix="%", font=dict(size=44, color=INK)),
        title=dict(text="protection G", font=dict(size=14, color=MUTED)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=MUTED),
            bar=dict(color=SIGNAL, thickness=0.28),
            bgcolor="white", borderwidth=0,
            steps=[dict(range=[0, 25], color="#f3e7e2"),
                   dict(range=[25, 50], color="#f1ede4"),
                   dict(range=[50, 100], color="#e4f0ee")],
        ),
    ))
    style(fig, height, margin=dict(l=20, r=20, t=40, b=0))
    return fig


def sankey_figure(res: dict, p: Params, height: int = 420) -> go.Figure:
    """Where the oxidative load goes, in R_gen units (reference day = 1)."""
    w = {"PM": p.w_PM, "O3": p.w_O3, "UV": p.w_UV}
    raw = {"PM": w["PM"] * res["p"], "O3": w["O3"] * res["z"], "UV": w["UV"] * res["u"]}
    film_pm = w["PM"] * (res["p"] - res["p_t"])
    film_o3 = w["O3"] * (res["z"] - res["z_t"])
    screen_uv = w["UV"] * (res["u"] - res["u_t"])
    chel = w["PM"] * res["p_t"] * (1.0 - res["fenton"])
    reach = {
        "PM": w["PM"] * res["p_t"] * res["fenton"],
        "O3": w["O3"] * res["z_t"],
        "UV": w["UV"] * res["u_t"],
    }
    total_reach = sum(reach.values())
    surv = res["Phi_scav_t"]
    k_e, k_a = res["kappa_endo_t"], res["kappa_app"]
    removed = total_reach * (1.0 - surv)
    endo = removed * k_e / (k_e + k_a) if (k_e + k_a) > 0 else 0.0
    appl = removed - endo
    resid = total_reach * surv

    nodes = ["Particulates", "Ozone", "UV",
             "Stopped by the film", "Screened by pulcherrimin", "Iron chelated",
             "Radicals formed in skin",
             "Skin's own antioxidants", "Applied scavengers", "Residual oxidative damage"]
    ncol = [CHANNEL_COLOR["PM"], CHANNEL_COLOR["O3"], CHANNEL_COLOR["UV"],
            COLOR["X"], COLOR["P"], "#8a3324", "#c98a3c",
            "#6c8c5a", COLOR["H"], RUST]
    links = [
        (0, 3, film_pm), (1, 3, film_o3), (2, 4, screen_uv), (0, 5, chel),
        (0, 6, reach["PM"]), (1, 6, reach["O3"]), (2, 6, reach["UV"]),
        (6, 7, endo), (6, 8, appl), (6, 9, resid),
    ]
    links = [l for l in links if l[2] > 1e-6]

    def rgba(hex_, a=0.35):
        h = hex_.lstrip("#")
        return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=nodes, color=ncol, pad=18, thickness=16,
                  line=dict(color="white", width=0.5)),
        link=dict(source=[l[0] for l in links], target=[l[1] for l in links],
                  value=[l[2] for l in links],
                  color=[rgba(ncol[l[0]]) for l in links],
                  hovertemplate="%{value:.3f} R_gen units<extra></extra>"),
    ))
    style(fig, height, margin=dict(l=0, r=0, t=10, b=10))
    return fig


# ----------------------------------------------------------------------------
# presets for the quick sliders
# ----------------------------------------------------------------------------

DAY_PRESETS = {
    "Reference day": dict(C_PM=80.0, C_O3=60.0, I_UV=1.0),
    "Delhi, winter smog": dict(C_PM=260.0, C_O3=35.0, I_UV=0.45),
    "Delhi, pre-monsoon": dict(C_PM=95.0, C_O3=120.0, I_UV=1.35),
    "Coastal city, summer": dict(C_PM=25.0, C_O3=90.0, I_UV=1.2),
    "Clean hill air": dict(C_PM=8.0, C_O3=45.0, I_UV=0.9),
}


def illustrative_day(hours=None):
    """A smooth, made-up urban day for when the live feed is unreachable.
    Two traffic peaks for PM, an afternoon photochemical hump for ozone and a
    noon bell for UV. Shapes only; labelled as such wherever it is drawn."""
    h = np.arange(24) if hours is None else np.asarray(hours)
    pm = 70 + 55 * np.exp(-((h - 8.5) / 2.2) ** 2) + 70 * np.exp(-((h - 21) / 2.8) ** 2)
    o3 = 25 + 95 * np.exp(-((h - 15) / 3.0) ** 2)
    uv = np.clip(9.0 * np.exp(-((h - 12.5) / 2.6) ** 2), 0, None)
    return h, pm, o3, uv

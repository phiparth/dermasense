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
# Pollution channels are drawn in neutrals on purpose. Ozone used to share the
# xylolipid teal and UV sat next to the control surfactant's violet, so a flow
# chart read as if a pollutant were a compound.
CHANNEL_COLOR = {"PM": "#4b4f55", "O3": "#7f8c94", "UV": "#a89f8c"}

# One rule for every chart, card and diagram on the site:
#   lab compounds (P, X, L)   solid fill in their own colour
#   controls (H, S)           hatched in bars, outlined in chips and cards
# and mixtures use the group colours below, never a compound's own colour.
LAB_GROUP = "#2f6f4f"
CONTROL_GROUP = "#8e7cab"
ALL_GROUP = INK

# Where the letters come from. Each is the initial the model's equations use.
ABBREV = {
    "P": ("Pulcherrimin", "lab"),
    "X": ("Xylolipid", "lab"),
    "L": ("Lyso-ornithine Lipid", "lab"),
    "H": ("Hyaluronic acid", "control"),
    "S": ("benchmark Surfactant", "control"),
}


def is_control(k: str) -> bool:
    return k in dm.CONTROL_KEYS


def bar_marker(keys, colors=None) -> dict:
    """Plotly bar marker: compound colours, controls hatched."""
    return dict(
        color=colors or [COLOR[k] for k in keys],
        pattern=dict(shape=["/" if is_control(k) else "" for k in keys],
                     fgcolor="white", size=5),
    )


def chip(k: str, size: str = "0.78rem") -> str:
    """A compound's letter as a chip: filled for lab, outlined for a control."""
    c = COLOR[k]
    if is_control(k):
        style = (f"color:{c};background:transparent;border:1.5px dashed {c};")
    else:
        style = f"color:#ffffff;background:{c};border:1.5px solid {c};"
    return (f"<span title='{ABBREV[k][0]}' style='{style}display:inline-block;"
            f"min-width:1.55em;text-align:center;border-radius:6px;font-weight:700;"
            f"font-size:{size};line-height:1.45;padding:0 .3em;margin-right:.25em;"
            f"font-family:ui-monospace,Consolas,monospace'>{k}</span>")


def chips(keys) -> str:
    return "".join(chip(k) for k in keys)


def legend_html(compact: bool = False) -> str:
    """Defines the letters once, in the same visual language as every chart."""
    items = []
    for k, (name, kind) in ABBREV.items():
        # bold the letter the abbreviation is taken from
        if k == "L":
            label = "Lyso-ornithine <b><u>L</u></b>ipid"
        elif k == "S":
            label = "benchmark <b><u>S</u></b>urfactant"
        else:
            label = f"<b><u>{name[0]}</u></b>{name[1:]}"
        items.append(f"<span style='white-space:nowrap;margin-right:1.1rem'>"
                     f"{chip(k)}{label}</span>")
    rule = ("" if compact else
            f"<div style='font-size:.78rem;color:{MUTED};margin-top:.45rem'>"
            "Each letter is the initial the model's equations use for that compound. "
            "Filled = our lab's compounds. Dashed outline = controls from the "
            "literature. Hatched bars in charts mean the same thing.</div>")
    return (f"<div class='card' style='padding:.75rem 1rem'>"
            f"<div style='line-height:2'>{''.join(items)}</div>{rule}</div>")


def legend() -> None:
    st.markdown(legend_html(), unsafe_allow_html=True)

CSS = f"""
<style>
  /* The cover's ground, shared by every page: two soft washes over a warm
     vertical gradient, pinned so a long page scrolls against a still
     background instead of stretching it. */
  .stApp {{
    background:
      radial-gradient(1100px 560px at 78% 8%, rgba(244, 205, 150, .55), rgba(244, 205, 150, 0) 60%),
      radial-gradient(900px 500px at 8% 30%, rgba(190, 214, 190, .45), rgba(190, 214, 190, 0) 62%),
      linear-gradient(180deg, #fbf7ef 0%, #f3ecdd 58%, #e8e0cb 100%);
    background-attachment: fixed;
  }}
  /* the chrome sits on the same ground rather than on a flat panel */
  [data-testid="stHeader"] {{ background: transparent; }}
  section[data-testid="stSidebar"] > div {{ background: rgba(255, 253, 247, .72); }}
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

# The chosen benchmark lives under its own plain key. A widget's value is
# dropped by Streamlit the moment its page stops rendering it, so keeping the
# choice in a widget key would reset it every time someone changed page.
_BENCH = "_benchmark_choice"


def benchmark_key() -> str:
    return st.session_state.get(_BENCH, dm.DEFAULT_BENCHMARK)


def benchmark_picker(where: str, *, kind: str = "select", container=None,
                     label: str = "Benchmark biosurfactant (S)", **kw) -> str:
    """One picker for the S slot, usable on any page. `where` makes the widget
    key unique per page; the choice itself is shared."""
    c = container or st
    keys = list(dm.BENCHMARKS)
    wkey = f"bench_{where}"

    def _sync():
        st.session_state[_BENCH] = st.session_state[wkey]

    widget = c.radio if kind == "radio" else c.selectbox
    choice = widget(
        label, keys, index=keys.index(benchmark_key()), key=wkey, on_change=_sync,
        format_func=lambda k: f"{dm.BENCHMARKS[k].name}: {dm.BENCHMARKS[k].role}",
        **kw,
    )
    st.session_state[_BENCH] = choice
    return choice


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


# The right-hand labels of the flow are the same words as the rows of the stage
# table next to it, so the two can be read across.
FLOW_STAGES = [
    ("S1a", "Particles stopped by the film"),
    ("S1b", "UV screened"),
    ("S2", "Iron chelated"),
    ("S3", "Radicals scavenged by applied compounds"),
    ("S3", "Radicals scavenged by the skin's own antioxidants"),
    ("", "Residual oxidative damage"),
]
FLOW_MIDDLE = "Radicals formed in skin"


def _dominant(shares: dict) -> list:
    """Compounds ordered by their share of a stage, largest first, zeros dropped."""
    return [k for k, v in sorted(shares.items(), key=lambda kv: -kv[1]) if v > 1e-12]


def flow_actors(res: dict, d: Dose, p: Params) -> dict:
    """Which compounds do each stage on this day, largest contributor first.
    The flow colours each stage by its largest contributor, so a stage carried
    mostly by lyso-ornithine lipid is blue, not teal."""
    film = _dominant({"L": res["occ_L"], "X": res["occ_X"], "S": res["occ_S"]})
    scav = _dominant({"X": p.k_X * d.X, "H": p.k_H * d.H})
    return {
        "Particles stopped by the film": film,
        "UV screened": ["P"] if d.P > 0 else [],
        "Iron chelated": ["P"] if d.P > 0 else [],
        "Radicals scavenged by applied compounds": scav,
    }


def sankey_figure(res: dict, p: Params, height: int = 420, d: Dose | None = None
                  ) -> go.Figure:
    """Where the oxidative load goes, in R_gen units (reference day = 1)."""
    if d is None:
        d = Dose()
    w = {"PM": p.w_PM, "O3": p.w_O3, "UV": p.w_UV}
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

    actors = flow_actors(res, d, p)

    def stage_label(code, text):
        who = actors.get(text)
        tail = f"  ({', '.join(who)})" if who else ""
        return f"{code} · {text}{tail}" if code else text

    def stage_color(text):
        if text in actors:
            who = actors[text]
            return COLOR[who[0]] if who else "#c9c2b4"
        if "skin's own" in text:
            return "#6c8c5a"
        return "#3d3a36"    # residual damage: dark, never a compound's colour

    nodes = (["Particulates", "Ozone", "UV"]
             + [stage_label(c, t) for c, t in FLOW_STAGES[:3]]
             + [FLOW_MIDDLE]
             + [stage_label(c, t) for c, t in FLOW_STAGES[3:]])
    ncol = ([CHANNEL_COLOR["PM"], CHANNEL_COLOR["O3"], CHANNEL_COLOR["UV"]]
            + [stage_color(t) for _, t in FLOW_STAGES[:3]]
            + ["#c9b48c"]
            + [stage_color(t) for _, t in FLOW_STAGES[3:]])
    # 0 PM, 1 O3, 2 UV, 3 film, 4 UV screen, 5 iron, 6 middle,
    # 7 applied, 8 skin's own, 9 residual
    links = [
        (0, 3, film_pm), (1, 3, film_o3), (2, 4, screen_uv), (0, 5, chel),
        (0, 6, reach["PM"]), (1, 6, reach["O3"]), (2, 6, reach["UV"]),
        (6, 7, appl), (6, 8, endo), (6, 9, resid),
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
                  # a link takes the colour of where it ends, so each compound's
                  # stage shows in its own colour all the way back to the source
                  color=[rgba(ncol[l[1]]) for l in links],
                  hovertemplate="%{value:.3f} R_gen units<extra></extra>"),
    ))
    style(fig, height, margin=dict(l=0, r=0, t=10, b=10))
    return fig


def stage_table_html(res: dict, d: Dose, p: Params) -> str:
    """The stage table that sits beside the flow. Its first rows are the flow's
    right-hand labels, word for word; the stages that act on the barrier and on
    dose rather than on radicals follow under a divider."""
    actors = flow_actors(res, d, p)
    rows = []
    for code, text in FLOW_STAGES:
        who = actors.get(text)
        if who is not None:
            who_html = chips(who) if who else f"<span style='color:{MUTED}'>none today</span>"
        elif "skin's own" in text:
            who_html = f"<span style='color:{MUTED}'>the skin itself</span>"
        else:
            who_html = f"<span style='color:{MUTED}'>what is left</span>"
        rows.append(f"<tr><td class='st-code'>{code}</td><td>{text}</td><td>{who_html}</td></tr>")
    rows.append("<tr><td colspan='3' class='st-div'>Not in the flow: these act on the "
                "barrier and on the dose, not on radicals</td></tr>")
    rest = [
        ("S4", "Water held in, or stripped out", ["H", "X", "L", "S"]),
        ("S5", "Delay between sensing and secreting", ["P", "X", "H"]),
        ("S6", "Dose harming the skin itself", ["P", "X", "L", "S"]),
    ]
    for code, text, keys in rest:
        rows.append(f"<tr><td class='st-code'>{code}</td><td>{text}</td><td>{chips(keys)}</td></tr>")
    return (
        "<style>.stg{width:100%;border-collapse:collapse;font-size:.88rem;"
        "background:#fff;border:1px solid #e3ded4;border-radius:10px;overflow:hidden}"
        ".stg th{text-align:left;font-size:.68rem;text-transform:uppercase;letter-spacing:.08em;"
        f"color:{MUTED};font-weight:600;padding:.5rem .7rem;border-bottom:1px solid #e3ded4}}"
        ".stg td{padding:.5rem .7rem;border-bottom:1px solid #efebe3;vertical-align:middle}"
        ".stg tr:last-child td{border-bottom:0}"
        ".stg .st-code{font-family:ui-monospace,Consolas,monospace;font-weight:600;"
        f"color:{MUTED};white-space:nowrap}}"
        f".stg .st-div{{font-size:.72rem;color:{MUTED};background:#faf8f3;"
        "text-transform:uppercase;letter-spacing:.06em}</style>"
        "<table class='stg'><thead><tr><th>Stage</th><th>What happens</th>"
        "<th>Compounds</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


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

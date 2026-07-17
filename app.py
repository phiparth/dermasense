"""
DermaSense — Adaptive Skincare Console (live-data Streamlit build)
==================================================================
iGEM modelling subteam · living pollution-responsive moisturizer

Same recommendation core as the HTML console, wired to LIVE environmental data,
with every coefficient anchored to real measured / peer-reviewed values:

  Decision 0 — environmental driver = Canadian Air Quality Health Index (AQHI)
                Stieb et al. 2008, J Air Waste Manag Assoc 58(3):435-50.
                excess-mortality relative-risk coefficients (NO2, O3, PM2.5).
  Decision 1 — pulcherrimin active dose (mg/mL), anchored to MEASURED data:
                Kregiel et al. 2024, Molecules 29(20):4873 (HaCaT keratinocytes)
                  0.20 mg/mL -> SPF 20            (effective floor)
                  0.40 mg/mL -> 26% ROS scavenging (dose-response)
                  <=1.58 mg/mL -> non-cytotoxic    (safety ceiling)
                  3.2 mg/mL   -> viability drops    (cytotoxic onset)
  Decision 2 — carrier blend (xylolipid : lyso-ornithine), HLB / required-HLB
                framework; direction is real surfactant science, gains to be
                fitted by tensiometry + TEWL on the team's own compounds.

Live data: Open-Meteo Air-Quality + Geocoding APIs (free, no API key).

Run:  pip install -r requirements.txt  &&  streamlit run app.py

Research prototype — no medical claims.
"""
from __future__ import annotations
import math
import numpy as np
import requests
import streamlit as st
from PIL import Image

# --------------------------------------------------------------------------- #
#  MODEL — real-sourced coefficients (mirror of the HTML console)              #
# --------------------------------------------------------------------------- #
# AQHI relative-risk coefficients (Stieb et al. 2008). Gases in ppb, PM2.5 ug/m3.
AQHI_B   = {"o3": 0.000537, "no2": 0.000871, "pm25": 0.000487}
AQHI_SCALE = 1000.0 / 10.4
UG_TO_PPB = {"no2": 0.531, "o3": 0.509}     # ug/m3 -> ppb at 25 C (24.45 / MW)

W_AIR, W_UV, UV_MAX = 0.60, 0.40, 11.0      # oxidative load = blend(AQHI, UV index)

# pulcherrimin dose model (mg/mL) — Kregiel et al. 2024, HaCaT
DOSE_FLOOR = 0.20      # SPF-20 effective concentration (measured)
FORM_MAX   = 1.00      # formulation ceiling: ~1.6x safety factor below 1.58 measured
CYTO_LIMIT = 1.58      # highest measured NON-cytotoxic concentration on HaCaT
SENS_CUT   = 0.40      # sensitive skin lowers ceiling: FORM_MAX - SENS_CUT*(S/100)

# carrier blend (HLB framework; gains to be fitted)
BLEND_BASE, K_OIL, K_SENS = 0.50, 0.25, 0.25
X_LO, X_HI = 0.15, 0.85


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def compute_aqhi(env: dict) -> tuple[float, dict]:
    """env: pm25, no2, o3 in ug/m3. Returns AQHI (0-10+) and its three terms."""
    o3ppb, no2ppb = env["o3"] * UG_TO_PPB["o3"], env["no2"] * UG_TO_PPB["no2"]
    term = {
        "o3":   math.exp(AQHI_B["o3"]   * o3ppb)     - 1,
        "no2":  math.exp(AQHI_B["no2"]  * no2ppb)    - 1,
        "pm25": math.exp(AQHI_B["pm25"] * env["pm25"]) - 1,
    }
    aqhi = max(0.0, AQHI_SCALE * sum(term.values()))
    return aqhi, term


def oxidative_load(env: dict) -> tuple[float, float, dict]:
    """Combine validated AQHI (pollution) and UV index into a 0-100 load."""
    aqhi, term = compute_aqhi(env)
    uv_n = clamp(env["uv"] / UV_MAX, 0, 1)
    air_n = clamp(aqhi / 10, 0, 1)
    load = 100 * (W_AIR * air_n + W_UV * uv_n)
    tsum = sum(term.values()) or 1.0
    air_c = W_AIR * air_n
    contrib = {
        "PM2.5": air_c * term["pm25"] / tsum,
        "Ozone": air_c * term["o3"] / tsum,
        "NO₂": air_c * term["no2"] / tsum,
        "UV":    W_UV * uv_n,
    }
    return load, aqhi, contrib


def active_dose(load: float, sensitivity: float) -> tuple[float, float, float]:
    """Decision 1: pulcherrimin mg/mL, floored at SPF-20 conc, capped for safety."""
    form_max = FORM_MAX - SENS_CUT * (sensitivity / 100.0)
    dose = clamp(DOSE_FLOOR + (form_max - DOSE_FLOOR) * (load / 100.0), DOSE_FLOOR, form_max)
    margin = CYTO_LIMIT / dose            # safety factor vs measured cytotoxic threshold
    return dose, form_max, margin


def carrier_blend(oiliness: float, sensitivity: float) -> float:
    x = BLEND_BASE + K_OIL * (oiliness - 50) / 50 - K_SENS * (sensitivity - 50) / 50
    return clamp(x, X_LO, X_HI)


def band_for(load: float) -> tuple[str, str]:
    if load < 25: return "Low load", "#2f9d72"
    if load < 50: return "Moderate load", "#c98a2b"
    if load < 75: return "High load", "#b23162"
    return "Severe load", "#c14b3f"


# --------------------------------------------------------------------------- #
#  IMAGE ANALYSIS — client-side logic mirrored (shine, CIE-Lab a*, ITA)        #
# --------------------------------------------------------------------------- #
def _rgb2lab(r, g, b):
    """Scalar sRGB (0-255) -> CIE-Lab (D65)."""
    def lin(c):
        c /= 255.0
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92
    r, g, b = lin(r), lin(g), lin(b)
    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722)
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883
    f = lambda t: t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)
    fx, fy, fz = f(x), f(y), f(z)
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def analyze_skin_image(pil_img: "Image.Image") -> dict:
    """Derive oiliness/sensitivity from a skin photo — same heuristics as the HTML tool."""
    img = pil_img.convert("RGB")
    img.thumbnail((440, 440))
    arr = np.asarray(img).astype(float)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    sat = np.where(mx == 0, 0.0, (mx - mn) / np.where(mx == 0, 1, mx))
    skin = (r > 50) & (r >= g * 0.9) & (g >= b * 0.85) & ((r - b) > 8)
    if skin.sum() < 40:                        # fallback: use all pixels
        skin = np.ones_like(skin, dtype=bool)
    n = int(skin.sum())
    # tone-relative shine: highlights brighter than the skin's OWN median (robust across skin tones)
    lum_skin, sat_skin = lum[skin], sat[skin]
    med = float(np.median(lum_skin))
    shine_frac = float(((lum_skin > med + 22) & (lum_skin > 70) & (sat_skin < 0.32)).sum()) / n
    mR, mG, mB = r[skin].mean(), g[skin].mean(), b[skin].mean()
    Ls, a_star, b_star = _rgb2lab(mR, mG, mB)
    ita = math.degrees(math.atan2(Ls - 50, b_star))
    oiliness = clamp(25 + shine_frac * 600, 8, 96)
    tone_adj = 8 if ita > 41 else (-4 if ita < 10 else 0)
    sensitivity = clamp((a_star - 8) * 7 + tone_adj + 22, 8, 96)
    tone = ("very light" if ita > 55 else "light" if ita > 41 else "intermediate"
            if ita > 28 else "tan" if ita > 10 else "brown" if ita > -30 else "dark")
    return {"O": oiliness, "S": sensitivity, "shine": shine_frac * 100,
            "a_star": a_star, "ita": ita, "tone": tone}


# --------------------------------------------------------------------------- #
#  LIVE DATA — Open-Meteo (free, no key)                                        #
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=1800, show_spinner=False)
def geocode(place: str):
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": place, "count": 1, "language": "en", "format": "json"}, timeout=15)
    r.raise_for_status()
    hits = r.json().get("results")
    if not hits:
        return None
    h = hits[0]
    return {"lat": h["latitude"], "lon": h["longitude"],
            "name": ", ".join(filter(None, [h.get("name"), h.get("admin1"), h.get("country")]))}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_env(lat: float, lon: float) -> dict:
    r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality",
                     params={"latitude": lat, "longitude": lon,
                             "current": "pm2_5,pm10,nitrogen_dioxide,ozone,uv_index",
                             "timezone": "auto"}, timeout=15)
    r.raise_for_status()
    c = r.json()["current"]
    return {"pm25": c.get("pm2_5") or 0.0, "pm10": c.get("pm10") or 0.0,
            "no2": c.get("nitrogen_dioxide") or 0.0, "o3": c.get("ozone") or 0.0,
            "uv": c.get("uv_index") or 0.0}


# --------------------------------------------------------------------------- #
#  UI                                                                           #
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="DermaSense Console", page_icon="\U0001f9eb", layout="wide")
st.markdown("""<style>
  .cap-ok{color:#2f9d72;} .cap-warn{color:#c98a2b;} .cap-cap{color:#c14b3f;}
  div[data-testid="stMetricValue"]{font-family:ui-monospace,monospace;}
</style>""", unsafe_allow_html=True)

st.title("\U0001f9eb DermaSense — Adaptive Skincare Console")
st.caption("Living moisturizer that senses the environment and doses its response. "
           "Environment driver = validated **AQHI**; active dose anchored to **measured** pulcherrimin "
           "photoprotection + keratinocyte-cytotoxicity data. Research prototype — no medical claims.")

left, right = st.columns([1, 1.15], gap="large")

O_MAP = {"Tight/flaky": 10, "Comfortable": 40, "Shiny T-zone": 65, "Shiny all over": 90,
         "Barely visible": 15, "Visible on nose": 45, "Large & visible": 78,
         "Rarely": 15, "Occasionally": 50, "Frequently": 85}
S_MAP = {"Never": 10, "Sometimes": 45, "Often": 78, "Almost always": 95,
         "No": 15, "A little": 55, "Yes, easily": 90,
         "Tan, rarely burn": 20, "Sometimes burn": 50, "Burn easily": 85}

with left:
    st.subheader("Signals in")
    place = st.text_input("Location", value="Delhi", help="City name -> live air-quality + UV")

    skin_mode = st.radio("Skin profile", ["Questionnaire", "Photo analysis"], horizontal=True)
    skin_src = "quiz"

    if skin_mode == "Questionnaire":
        q1 = st.select_slider("A few hours after washing, your face is…",
                              ["Tight/flaky", "Comfortable", "Shiny T-zone", "Shiny all over"], value="Comfortable")
        q2 = st.select_slider("Your pores are…", ["Barely visible", "Visible on nose", "Large & visible"], value="Visible on nose")
        q3 = st.select_slider("You get breakouts…", ["Rarely", "Occasionally", "Frequently"], value="Occasionally")
        q4 = st.select_slider("New products cause redness/stinging…", ["Never", "Sometimes", "Often", "Almost always"], value="Sometimes")
        q5 = st.select_slider("Your skin flushes/reddens easily…", ["No", "A little", "Yes, easily"], value="A little")
        q6 = st.select_slider("In strong sun you tend to…", ["Tan, rarely burn", "Sometimes burn", "Burn easily"], value="Sometimes burn")
        oiliness = (O_MAP[q1] + O_MAP[q2] + O_MAP[q3]) / 3
        sensitivity = (S_MAP[q4] + S_MAP[q5] + S_MAP[q6]) / 3
    else:
        st.caption("Upload a well-lit close-up of cheek/forehead skin. Analysed locally — no ML, no upload to any server.")
        photo = st.file_uploader("Skin photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        if photo is not None:
            pic = Image.open(photo)
            st.image(pic, caption="Analysed region: whole image", use_container_width=True)
            res = analyze_skin_image(pic)
            oiliness, sensitivity, skin_src = res["O"], res["S"], "photo"
            c1, c2, c3 = st.columns(3)
            c1.metric("Shine → oiliness", f"{res['shine']:.1f}%")
            c2.metric("Erythema a*", f"{res['a_star']:.1f}")
            c3.metric("Skin tone ITA°", f"{res['ita']:.0f}°", res["tone"])
            st.caption("⚠ Optical proxy, not a diagnosis. Redness is harder to read on deeper skin tones — cross-check with the questionnaire.")
        else:
            oiliness, sensitivity = 45.0, 50.0
            st.info("Upload a photo, or switch to the questionnaire. Showing neutral defaults meanwhile.")

barrier = clamp(100 - 0.45 * (100 - oiliness) - 0.45 * sensitivity, 0, 100)

with right:
    st.subheader("Readout")
    try:
        loc = geocode(place.strip()) if place.strip() else None
    except Exception as e:
        st.error(f"Geocoding failed: {e}"); st.stop()
    if not loc:
        st.warning("Type a city to resolve live environmental data."); st.stop()
    try:
        env = fetch_env(loc["lat"], loc["lon"])
    except Exception as e:
        st.error(f"Air-quality lookup failed: {e}"); st.stop()

    load, aqhi, contrib = oxidative_load(env)
    dose, form_max, margin = active_dose(load, sensitivity)
    xfrac = carrier_blend(oiliness, sensitivity)
    xpct, lpct = round(xfrac * 100), 100 - round(xfrac * 100)
    band_lbl, _ = band_for(load)

    st.markdown(f"**{loc['name']}** · live")
    m1, m2, m3 = st.columns(3)
    m1.metric("AQHI", f"{aqhi:.1f}/10", band_lbl)
    m2.metric("Pulcherrimin", f"{dose:.2f} mg/mL", f"{margin:.1f}x safety")
    m3.metric("Carrier xylo:lyso", f"{xpct}:{lpct}")

    st.markdown("**Live environmental readout** (µg/m³, UV index)")
    st.dataframe({"PM2.5": [round(env["pm25"], 1)], "PM10": [round(env["pm10"], 1)],
                  "NO₂": [round(env["no2"], 1)], "O₃": [round(env["o3"], 1)], "UV": [round(env["uv"], 1)]},
                 hide_index=True, use_container_width=True)

    st.markdown("**What's driving the oxidative load** (share of index)")
    for k, v in sorted(contrib.items(), key=lambda kv: kv[1], reverse=True):
        share = v / (load / 100) if load > 0 else 0
        st.progress(min(share, 1.0), text=f"{k}  {share*100:4.0f}%")

    if dose >= form_max - 1e-6 and sensitivity >= 66:
        st.markdown(f"<div class='cap-cap'>⚠ Held at the sensitive-skin ceiling <b>{form_max:.2f} mg/mL</b> — "
                    f"{margin:.1f}× below the measured cytotoxic threshold. Safety wins.</div>", unsafe_allow_html=True)
    elif sensitivity >= 66:
        st.markdown(f"<div class='cap-warn'>◑ Sensitive-skin ceiling lowered to {form_max:.2f} mg/mL; dose sits below it.</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='cap-ok'>✓ Inside the tested-safe window — {margin:.1f}× below the concentration "
                    f"where HaCaT viability drops (1.58 mg/mL).</div>", unsafe_allow_html=True)

    load_word = "clean-air" if load < 25 else "moderately polluted" if load < 50 else "heavily polluted" if load < 75 else "extreme"
    top = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)[0][0]
    carrier_word = ("weighted to xylolipid for spreading" if xpct > 60
                    else "weighted to lyso-ornithine to stay gentle" if xpct < 40 else "kept near balanced")
    st.markdown("**Why this formulation**")
    st.write(
        f"Live air quality scores **AQHI {aqhi:.1f}/10** ({load_word}, led by {top}); with UV that is an oxidative "
        f"load of **{load:.0f}/100**. The microbe is dialled to **{dose:.2f} mg/mL pulcherrimin** — above the measured "
        f"SPF-20 floor (0.20 mg/mL) and **{margin:.1f}× below** the concentration that reduced keratinocyte viability. "
        f"Carrier **{xpct}:{lpct}**, {carrier_word}, sliding the blend's HLB toward what this skin type tolerates."
    )
    with st.expander("Skin parameters"):
        st.write(f"Oiliness **{oiliness:.0f}** · Sensitivity **{sensitivity:.0f}** · "
                 f"Barrier **{barrier:.0f}** · source **{skin_src}**")

with st.expander("Recommendation core — every coefficient sourced (mechanistic, not ML)"):
    st.markdown(
        """
No published dataset maps *(pollution, skin type) → optimal dose*, so this is a **mechanistic model with
real-measured anchors**, not a black box. Tags: ✅ measured/validated · 🔧 to-fit (Phase-4 calibration).

```text
1 · AQHI (✅ Stieb 2008)  gases ug/m3 -> ppb (x0.531 NO2, x0.509 O3)
    AQHI = (1000/10.4)·[ (e^0.000537·O3 −1) + (e^0.000871·NO2 −1) + (e^0.000487·PM2.5 −1) ]
    load = 100·( 0.60·clamp(AQHI/10,0,1) + 0.40·clamp(UVI/11,0,1) )      # 0.60/0.40 = 🔧

2 · dose (✅ anchors, Kregiel 2024, HaCaT)
    floor 0.20 mg/mL (SPF 20) · non-cytotoxic <=1.58 mg/mL · cytotoxic 3.2 mg/mL
    form_max = 1.00 − 0.40·(S/100)        # 1.00 = 1.6x safety factor 🔧 ; 0.40 cut 🔧
    dose = clamp(0.20 + (form_max−0.20)·(load/100), 0.20, form_max)      # mg/mL

3 · carrier (✅ HLB direction · 🔧 gains)
    xylo_frac = clamp(0.50 + 0.25·(O−50)/50 − 0.25·(S−50)/50, 0.15, 0.85)
    blend = xylo_frac : (1 − xylo_frac)   # glycolipid (↑HLB, oily) : amino-lipid (↓HLB, sensitive)
```

**Calibration targets (Phase 4):** dose curve ← DPPH/ABTS vs concentration on your strain's pulcherrimin;
in-vitro → formulation translation ← permeation + patch test; blend gains ← tensiometry (measured HLB) + TEWL.

**Sources —** according to PubMed:
AQHI: Stieb et al. 2008, [10.3155/1047-3289.58.3.435](https://doi.org/10.3155/1047-3289.58.3.435).
Pulcherrimin photoprotection/cytotoxicity: Kregiel et al. 2024, [10.3390/molecules29204873](https://doi.org/10.3390/molecules29204873).
Glycolipid biosurfactants in skincare: [PMC10254413](https://pmc.ncbi.nlm.nih.gov/articles/PMC10254413/).
"""
    )

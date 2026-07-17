# DermaSense — Adaptive Skincare Console

A pollution-responsive *living moisturizer* demo for iGEM. An engineered skin microbe
**senses the environment and doses its response**. This repo turns two real-world signals —
**where you are** and **what your skin is like** — into a defensible formulation.

## The scientific spine: two coupled decisions

The recommendation is **not** "pick a percentage of each of three compounds." Per the biology,
pulcherrimin is the **active**; xylolipid and lyso-ornithine are **carriers** (biosurfactants).
So there are two decisions, not three numbers:

| Decision | Output | Driven by |
|----------|--------|-----------|
| **1 — how much active** | pulcherrimin % w/w | oxidative load (pollution + UV) |
| **2 — which carrier blend** | xylolipid : lyso-ornithine | skin type (oiliness + sensitivity) |

This split is what makes the model **mechanistic and defensible** instead of a black box
hallucinating three numbers with no training data behind it.

## The model — grounded in real data, not invented coefficients

There is **no** published dataset mapping *(pollution, skin type) → optimal dose*, so this is a
**mechanistic model with real-measured anchors**, never a black box trained on data that doesn't exist.
Each coefficient is tagged ✅ **measured/validated** or 🔧 **to-fit** (Phase-4 wet-lab).

```text
1 · Environment = Air Quality Health Index    ✅ Stieb et al. 2008 (Health Canada)
    gases ug/m3 -> ppb (x0.531 NO2, x0.509 O3)
    AQHI = (1000/10.4)·[ (e^0.000537·O3 −1) + (e^0.000871·NO2 −1) + (e^0.000487·PM2.5 −1) ]
    load = 100·( 0.60·clamp(AQHI/10,0,1) + 0.40·clamp(UVI/11,0,1) )        # 0.60/0.40 = 🔧

2 · Active dose (mg/mL)                        ✅ Kregiel et al. 2024, HaCaT keratinocytes
    floor 0.20 mg/mL = SPF 20 (measured) · non-cytotoxic <=1.58 mg/mL · cytotoxic 3.2 mg/mL
    form_max = 1.00 − 0.40·(S/100)             # 1.00 = 1.6x safety factor below 1.58 🔧
    dose = clamp( 0.20 + (form_max−0.20)·(load/100), 0.20, form_max )

3 · Carrier blend                              ✅ HLB direction · 🔧 gains
    xylo_frac = clamp(0.50 + 0.25·(O−50)/50 − 0.25·(S−50)/50, 0.15, 0.85)
    blend = xylo_frac : (1 − xylo_frac)        # glycolipid (↑HLB, oily) : amino-lipid (↓HLB, sensitive)
```

### Real measured anchors behind the two decisions

| Quantity | Value | Source |
|----------|-------|--------|
| AQHI risk coefficients (NO₂/O₃/PM2.5) | 0.000871 / 0.000537 / 0.000487 | Stieb 2008, [DOI](https://doi.org/10.3155/1047-3289.58.3.435) |
| Pulcherrimin dose floor (SPF 20) | 0.20 mg/mL | Kregiel 2024, [DOI](https://doi.org/10.3390/molecules29204873) |
| ROS scavenging (dose-response) | 11% @1mM, 26% @5mM H₂O₂ (0.4 mg/mL) | Kregiel 2024 |
| Highest non-cytotoxic conc. (HaCaT) | 1.58 mg/mL | Kregiel 2024 |
| Cytotoxic onset | 3.2 mg/mL (viability →84%) | Kregiel 2024 |

**Still 🔧 to-fit in Phase 4:** the air-vs-UV weight, the safety factor, and the carrier HLB gains —
each maps to one bench assay (DPPH/ABTS · patch test · tensiometry + TEWL).

## Two builds

### 1. `dermasense.html` — shareable console (no install, no key)
Self-contained interactive page. City presets + manual override, full questionnaire, live gauge,
two-decision outputs, and the model-transparency panel. Open in any browser, or use the published
Artifact link. (A browser sandbox can't call external APIs, so this uses curated city presets.)

### 2. `app.py` — live-data Streamlit build
Same core, wired to **live** air-quality + UV via the free **Open-Meteo** API (no API key), plus the
same **photo skin-analysis** (shine → oiliness, CIE-Lab a* → sensitivity, ITA° skin tone):

```bash
pip install -r requirements.txt
streamlit run app.py
```

Type a city → real-time PM2.5 / NO₂ / O₃ / UV → AQHI → formulation, or upload a skin close-up.
Verified against live Delhi data (AQHI ~11 on a high-pollution reading).

> Swap Open-Meteo for **CPCB** (India) or **WAQI/OpenAQ** if you want a specific national source;
> the `fetch_env()` function is the only thing that changes.

## Deploy to Streamlit Community Cloud (free)

The repo is deploy-ready. After pushing to GitHub (see below):

1. Go to **https://share.streamlit.io** → sign in with GitHub.
2. **New app** → pick this repo, branch `main`, main file `app.py`.
3. **Deploy.** Streamlit installs `requirements.txt` and serves `app.py`. No secrets or API keys needed.

Files that make it deployable: `app.py`, `requirements.txt`, `.streamlit/config.toml` (theme), `.gitignore`.
`dermasense.html` is the standalone no-install console (open directly or host on any static site / GitHub Pages).

## Grounding literature
- Particulate matter → cutaneous oxidative stress & barrier disruption — PMID **38718731** (+23 related)
- Ozone depletes stratum-corneum antioxidants (vit. E/C) — Thiele & Packer, PMID **10648834**
- Pulcherrimin: iron-sequestering pigment, antioxidant activity — PMID **34579848**

## Roadmap (maps to the project phases)
- **Phase 1 (done)** environmental pipeline + composite OSI → `app.py` live API
- **Phase 2 (done, v1)** Baumann-style questionnaire → skin-parameter vector
- **Phase 3 (done)** two-decision recommendation core, safety-capped
- **Phase 4 (next)** wet-lab calibration loop — replace each placeholder coefficient with a fitted value
- **Phase 5 (done)** Streamlit + HTML front-ends
- **Phase 6** sensitivity analysis, edge cases, wiki writeup

---
**Research prototype — no medical claims.** No published clinical dose-response literature exists for
these topical ratios; the model is a *testable hypothesis* to calibrate against bench data, not a
validated clinical tool.

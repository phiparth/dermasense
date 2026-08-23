# DermaSense v3

Streamlit front end for the DermaSense v3 unified model: pollution exposure to
skin damage, and how much of it four engineered compounds take back.

iGEM IIT Delhi 2026, modelling.

## What it does

Two modes, both driven by the same cascade.

**Inverse (default).** Five environment inputs go in, a dose comes out.

| in | | ref | where it can come from |
|---|---|---|---|
| `C_PM` | PM2.5, ug/m3 | 80 | live feed for a place, or a slider |
| `C_O3` | ozone, ug/m3 | 60 | live feed for a place, or a slider |
| `I_UV` | UV dose, dimensionless 0 to 1 | 1.0 | live UV index / `UVI_REF`, or a slider |
| `PT` | Fitzpatrick phototype, 1 to 6 | I | questionnaire, photo, or a slider |
| `A0` | barrier state, binary | 1.00 healthy | questionnaire, photo, or a radio |

Out: `d* = (P, X, L, H)` in mg/mL, protection `G`, absolute `RSD` before and
after, absolute TEWL before and after.

**Forward.** You supply the dose as well, and the model scores it.

At the reference day the app reproduces the specification exactly:
`d* = (0.086, 0.300, 0.010, 1.000)`, `G = 55.8%`, `RSD 1.000 -> 0.442`,
`TEWL 20.54 -> 16.69 g/m2/h` (18.7%).

## Where the inputs come from

The sidebar picks a route for the exposure and a route for the skin, the panel
under the title does the capture, and the five resolved numbers are printed back
in the sidebar so the interface and the model can never quietly disagree.

### Exposure, from a location

Type a city or a `lat, lon` pair. `env_api.py` geocodes it and pulls the current
hour from Open-Meteo's air-quality API: PM2.5 and ozone in ug/m3, which are the
model's units already, plus the UV index.

Open-Meteo is CAMS underneath, so a city reading is a model reanalysis at the
nearest grid cell, not a kerbside monitor. It needs no key, which is why it
survives a Community Cloud deploy.

The one conversion is UV. The model's `I_UV` is a ratio and the spec never said
against what, so `UVI_REF = 8.0` is ours, evidence class C: the bottom of the
WHO "very high" band, the same kind of day `C_PM = 80` and `C_O3 = 60` describe.
It is editable in the app. **Today's peak UV is the default, not the current
hour** - a moisturiser applied in the morning is on the face all day, and "right
now" reads 0.0 after dark, which would recommend no UV screen at all.

`domain_warnings()` fires when a live day lands outside the range the constants
were fitted on, which Delhi in winter does routinely.

```bash
python env_api.py Delhi
python env_api.py "28.65, 77.23" --uv now
```

### Skin, from a questionnaire

Two published instruments, scored exactly as published:

- **Phototype**: Fitzpatrick's ten self-report items, 0-4 each, banded 0-6 = I
  through 35-40 = VI (Fitzpatrick 1988).
- **Barrier**: POEM, seven items about the last seven days, 0-4 each (Charman
  2004). Used as a barrier-symptom score, not as an eczema diagnosis: its items
  ask about dryness, cracking and flaking, which is what `A0` stands in for.

The mapping POEM to `A0` is ours, class C: linear from `A0 = 1.00` at POEM 0 to
`A0 = 0.65` at POEM 16, the top of the moderate band. The spec's `A0` is binary,
and `skin_inputs.snap_binary()` collapses it back if you want the two literal
values. A checkbox forces the compromised value for the insults POEM does not
ask about: an active retinoid or acid, a fresh peel, a shave rash, a week of
cold wind.

### Skin, from a photo

Upload a well-lit close-up of a cheek or forehead with some background in frame.
Nothing is uploaded anywhere: the arithmetic runs in the same process as the
rest of the app.

1. Skin mask, Kovac's RGB rule unioned with the Chai/Ngan YCbCr box, then the
   top 3% and bottom 5% of luminance dropped, because blown highlights and deep
   shadow are illumination rather than pigment.
2. **Phototype**: median L\*a\*b\* over what is left, ITA = atan2(L\* - 50, b\*),
   classed by Del Bino 2013's published edges (very light above 55 deg through
   dark below -30 deg) and read across to types I to VI.
3. **Barrier**, both proxies class C and uncalibrated: erythema spread, the
   p90 - p50 gap of a smoothed a\* map, and surface texture, a band-passed Weber
   contrast with the sensor-noise floor subtracted.

The noise subtraction and the Weber normalisation exist for one reason: a photo
of deep skin carries more noise per unit signal and a harder edge against the
background, and without them both read as flaking. `test_inputs.py` asserts that
calm skin scores `A0 > 0.93` at every one of the six tones, and that flaking and
redness lower it at every one of the six.

**Why ITA and not a CNN.** ITA is the instrument dermatology already uses for
constitutive pigmentation, its class edges are published, it is about forty
lines of arithmetic that can be shown on the wiki, it has no training set to
inherit bias from, and it runs in milliseconds on a free dyno. A MobileNetV3-S
or EfficientNet-B0 fine-tuned on Fitzpatrick17k and exported to int8 ONNX would
be about 6 MB and 30 ms on CPU, which does fit the free tier, but
Fitzpatrick17k's labels are annotator-assigned rather than measured and are
noisy above type IV, its images are clinical dermatology photos where ours are
phone selfies, and a black box we cannot defend on the wiki is worse than
arithmetic we can. The hook is still there: drop an ONNX file at
`models/phototype.onnx`, or point `DERMASENSE_PHOTOTYPE_ONNX` at one, and the
app shows its answer and its confidence next to the ITA answer. ITA stays the
one that feeds the model.

ITA's real weakness is the illuminant. The app warns on colour cast, warns when
there is no background to judge the cast against, and offers a grey-world
correction for frames that do have a neutral surface in them.

```bash
python skin_inputs.py --photo cheek.jpg
```

### When both skin routes run

- **Phototype: the darker of the two.** `f_PT` lowers the UV channel as the
  phototype rises, so the darker answer recommends less pulcherrimin. A
  disagreement then never inflates the dose on the strength of a self-report the
  literature says is optimistic on darker skin.
- **Barrier: the worse of the two.** Under-treating a disrupted barrier is the
  failure that matters, and the model's own S6 self-toxicity term caps the dose
  from above, so there is no runaway on that side.

The app says which rule it applied and what the two answers were.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Command line, for numbers you want to paste into the wiki:

```bash
python dermasense_unified.py                  # reference day
python dermasense_unified.py --pm 180 --o3 70 --uv 0.8 --pt 4 --compromised
python dermasense_unified.py --sensitivity    # experiment ranking
python dermasense_unified.py --grid           # does the recommended dose move
```

Tests:

```bash
python test_model.py          # or: python -m pytest test_model.py -q
python test_inputs.py         # the input layer: questionnaires, photos, the feed
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repo.
2. Go to share.streamlit.io, sign in with GitHub, pick **New app**.
3. Repository: your repo. Branch: `main`. Main file path: `app.py`.
4. Deploy. Dependencies come from `requirements.txt`; theme from
   `.streamlit/config.toml`.

Nothing here needs secrets, a database, or a paid tier. Cold start is a few
seconds; every optimisation runs in well under a second and results are cached.

## Files

| file | what it is |
|---|---|
| `dermasense_model.py` | the model. Every equation maps one to one onto a section of the spec. No Streamlit imports, so it is usable from a notebook |
| `app.py` | the Streamlit interface |
| `dermasense_unified.py` | command line entry point |
| `env_api.py` | place name to live PM2.5, ozone and UV index, converted to model units. No Streamlit import |
| `skin_inputs.py` | the questionnaires, and the photo route: skin mask, ITA, the two barrier proxies. No Streamlit import |
| `test_model.py` | the spec's claims, asserted rather than described |
| `test_inputs.py` | the input layer's claims, including the tone-fairness ones |
| `requirements.txt`, `.streamlit/config.toml` | deploy |

## What the app is careful about

These are the three things most likely to be misread, so the interface states
them rather than leaving them in a footnote.

**G is never shown alone.** It is a ratio against the same day untreated, so it
can rise while the person gets worse. Compromised skin scores a higher `G` and
higher absolute damage at the same time, because a smaller endogenous reserve
lets the applied compounds win a larger share of the radical pool. Quoted
alone, `G` would recommend less product for the skin that needs more. The app
prints absolute `RSD` next to it everywhere.

**Three of four doses are not recommendations.** Only pulcherrimin has an
interior optimum, pushed off its cap by the S6 reversal term, so more genuinely
would be worse. Xylolipid, lyso-ornithine lipid and hyaluronic acid sit on
their caps because nothing inside the validity domain penalises them. That is
the model saying we ran out of evidence, not that this is the right amount. The
lyso-ornithine lipid cap is different in kind again: 0.010 mg/mL is what the
strain can make, not what anyone measured.

**The dose barely responds to the environment.** The "Does the dose move?" tab
runs the optimiser across a grid and prints the spread. Three coordinates never
move at all; pulcherrimin moves about 12% of its cap across a sweep from 20 to
300 ug/m3 and phototype I to VI. So the feature recommends close to one
formulation and predicts how well it will do today. `G` and the absolute `RSD`
genuinely are personalised. "The model computes your personal optimal dose"
would overclaim. If you want the dose to actually respond to exposure, you need
a penalty that bites inside the domain: lower `C_crit` below the `X` cap, or
give hyaluronic acid a self-toxicity term.

## Model structure

Part 1, exposure to damage, everything at zero dose:

```
f_PT  ->  p, z, u  ->  R_gen  ->  kappa_endo, Phi_scav  ->  R
                       delta_T ->  B
                       RSD_0 = w_ox R + (1 - w_ox) B
```

Part 2, six stages, each acting at a different point in the causal chain, which
is why they multiply rather than add:

| stage | what it does | compounds |
|---|---|---|
| S1a | interfacial shielding, competitive Langmuir over one surface | L, X |
| S1b | UV screen, Mansur SPF linear in dose | P |
| S2 | Fenton suppression, iron chelation upstream of scavenging | P |
| S3 | competitive radical scavenging, one shared denominator | X, H, and the skin itself |
| S4 | barrier modulation, humectant gain and stripping penalty | H, L, X |
| S5 | delivery timing, the price of sensing before secreting | P, X, H |
| S6 | self-toxicity, the reason a recommended dose exists at all | P, L, X |

```
RSD(e, d) = w_ox R' + (1 - w_ox) B' + Omega D_s
G(e, d)   = 1 - RSD(e, d) / RSD(e, 0)
```

Setting `d = 0` recovers Part 1 exactly, to 1e-12, at an arbitrary exposure and
not just at reference. `test_model.py` asserts it.

## Evidence classes

32 constants, 11 of them class C, meaning nobody measured them and we chose a
value. Four class C constants carry most of the spread: `theta_max` (22.6
points), `assay_xfer` (9.0), `C_crit` (7.8), `kappa_0` (4.8). The Uncertainty
tab recomputes this ranking for whatever environment is loaded, so the
experiment priorities are derived rather than quoted.

The conservative preset pushes every class C constant to its pessimistic end
and gives about 28% instead of 55.8%. The worst case preset also pushes the
class A and B constants, which double-counts pessimism across measurements
unlikely to all be wrong at once, and gives about 17%.

## Known limits, stated up front

- Both biosurfactants are extrapolated from studies about fruit juice and crude
  oil. Neither has touched a keratinocyte in the published record, yet together
  they carry most of the headline number.
- The Milani agreement (18.7% predicted against 19% measured) is not a
  validation of the shielding mechanism. Milani's serum contains no film-forming
  surfactant, so the two numbers are not the same number. It validates the scale
  of the barrier channel and nothing more.
- Concentration in a film is assumed to be the right variable, rather than a
  surface density in mg/cm2, and film thickness is assumed identical across
  people. Neither assumption is examined in any source.
- Channels are added, not multiplied, so UV and ozone synergy is ignored. This
  under-predicts damage under simultaneous exposure.
- `alpha_PM` is fitted over roughly 30 to 100 ug/m3. At Delhi winter levels the
  barrier channel is extrapolating.
- The live feed is a CAMS grid cell, not the air this person is breathing.
  Indoors, in traffic, or twenty floors up it is wrong in a direction the app
  cannot see.
- `UVI_REF = 8.0` sets the scale of the entire UV channel and nobody measured
  it.
- ITA is illuminant-dependent: an uncorrected white balance moves it by tens of
  degrees. ITA and Fitzpatrick also agree only loosely in the literature, since
  ITA measures pigment and the Fitzpatrick items measure how skin reacts to sun.
- The photo barrier proxies have never been put next to a TEWL probe. They move
  in the right direction on synthetic tests, and that is the whole claim.
- Self-reported phototype is known to skew light above type IV, which is why a
  disagreement between the two routes is settled toward the darker answer rather
  than averaged.

Model outputs are predictions from a parameterised cascade. They are not
clinical guidance and not a formulation instruction.

# DermaSense v3

Streamlit front end for the DermaSense v3 unified model: pollution exposure to
skin damage, and how much of it four engineered compounds take back.

iGEM IIT Delhi 2026, modelling.

## What it does

Two modes, both driven by the same cascade.

**Inverse (default).** Five environment inputs go in, a dose comes out.

| in | | ref |
|---|---|---|
| `C_PM` | PM2.5, ug/m3 | 80 |
| `C_O3` | ozone, ug/m3 | 60 |
| `I_UV` | UV dose, dimensionless 0 to 1 | 1.0 |
| `PT` | Fitzpatrick phototype, 1 to 6 | I |
| `A0` | barrier state, binary | 1.00 healthy |

Out: `d* = (P, X, L, H)` in mg/mL, protection `G`, absolute `RSD` before and
after, absolute TEWL before and after.

**Forward.** You supply the dose as well, and the model scores it.

At the reference day the app reproduces the specification exactly:
`d* = (0.086, 0.300, 0.010, 1.000)`, `G = 55.8%`, `RSD 1.000 -> 0.442`,
`TEWL 20.54 -> 16.69 g/m2/h` (18.7%).

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
| `test_model.py` | the spec's claims, asserted rather than described |
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

Model outputs are predictions from a parameterised cascade. They are not
clinical guidance and not a formulation instruction.

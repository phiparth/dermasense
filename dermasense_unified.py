"""
Command line entry point. Prints the numbers the wiki quotes, so nothing on the
wiki is typed by hand.

    python dermasense_unified.py                 reference day
    python dermasense_unified.py --pm 180 --o3 70 --uv 0.8 --pt 4 --compromised
    python dermasense_unified.py --sensitivity   experiment ranking
    python dermasense_unified.py --grid          does the recommended dose move
"""

from __future__ import annotations

import argparse

import numpy as np

from dermasense_model import (
    A0_COMPROMISED, A0_HEALTHY, CAPS, COMPOUND_NAME, DOSE_ORDER,
    Dose, Environment, NOMINAL,
    conservative_params, dose_spread, evaluate, monte_carlo, optimal_dose,
    per_compound_optima, sensitivity, single_compound_optima,
)


def rule(title: str) -> None:
    print(f"\n{title}\n" + "-" * max(28, len(title)))


def report(e: Environment) -> None:
    d, g = optimal_dose(e)
    r = evaluate(e, d)

    rule("Environment")
    print(f"  PM2.5 {e.C_PM:g} ug/m3   O3 {e.C_O3:g} ug/m3   UV {e.I_UV:g}")
    print(f"  phototype {'I II III IV V VI'.split()[e.PT - 1]}   "
          f"barrier {'healthy' if e.A0 == A0_HEALTHY else 'compromised'}   "
          f"lag {e.lag_h:g} h")

    rule("Recommended dose, mg/mL")
    for k in DOSE_ORDER:
        v = getattr(d, k)
        pos = "at cap" if v >= CAPS[k] - 1e-6 else "interior optimum"
        print(f"  {COMPOUND_NAME[k]:<22} {v:7.4f}   cap {CAPS[k]:<6g} {pos}")
    print("  three of these are the edge of the evidence, not a recommendation")

    rule("Outputs")
    print(f"  protection G            {g * 100:6.1f} %")
    print(f"  absolute RSD            {r['RSD0']:.4f} -> {r['RSD']:.4f}")
    print(f"  absolute TEWL, g/m2/h   {r['T']:.2f} -> {r['T_prime']:.2f} "
          f"({r['TEWL_improvement'] * 100:.1f} % improvement)")
    print(f"  self-toxicity Omega     {r['Omega']:.4f}  (penalty {r['penalty']:.4f})")
    print(f"  kappa endo              {r['kappa_endo']:.3f} -> {r['kappa_endo_t']:.3f}"
          "  (rises: the reserve is less depleted under protection)")
    print("  report G and absolute RSD together, always")

    rule("Sub-additivity")
    groups = single_compound_optima(e)
    for lab, v in groups.items():
        print(f"  {lab:<34} {v * 100:5.1f} %")
    naive = sum(per_compound_optima(e).values())
    print(f"  {'all four together':<34} {g * 100:5.1f} %")
    print(f"  {'naive sum of single compounds':<34} {naive * 100:5.1f} %")

    rule("Uncertainty")
    print(f"  nominal                 {g * 100:5.1f} %")
    print(f"  conservative (class C)  {optimal_dose(e, conservative_params('C'))[1] * 100:5.1f} %")
    print(f"  worst case (all bands)  {optimal_dose(e, conservative_params('all'))[1] * 100:5.1f} %")
    draws = monte_carlo(e, n=600, d_fixed=d)
    lo, med, hi = np.percentile(draws, [5, 50, 95])
    print(f"  Monte Carlo 90% interval {lo * 100:.0f} to {hi * 100:.0f} %"
          f"  (median {med * 100:.1f} %)")


def main() -> None:
    ap = argparse.ArgumentParser(description="DermaSense v3")
    ap.add_argument("--pm", type=float, default=80.0)
    ap.add_argument("--o3", type=float, default=60.0)
    ap.add_argument("--uv", type=float, default=1.0)
    ap.add_argument("--pt", type=int, default=1, choices=range(1, 7))
    ap.add_argument("--compromised", action="store_true")
    ap.add_argument("--lag", type=float, default=0.0, help="sensor lag, hours")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()

    e = Environment(a.pm, a.o3, a.uv, a.pt,
                    A0_COMPROMISED if a.compromised else A0_HEALTHY, a.lag)
    report(e)

    if a.sensitivity:
        rule("Experiment priorities, ranked by swing in G*")
        for row in sensitivity(e)[:10]:
            print(f"  {row['parameter']:<14} class {row['class']}  "
                  f"band {row['low']:g} to {row['high']:g}   "
                  f"{row['swing_pts']:5.1f} pts")

    if a.grid:
        rule("Does the recommended dose move with the environment")
        exps = [Environment(pm, a.o3, a.uv, pt,
                            A0_COMPROMISED if a.compromised else A0_HEALTHY, a.lag)
                for pm in (20, 80, 180, 300) for pt in (1, 4, 6)]
        rows = dose_spread(exps)
        for k in DOSE_ORDER:
            vals = [r[k] for r in rows]
            span = (max(vals) - min(vals)) / CAPS[k] * 100
            print(f"  {COMPOUND_NAME[k]:<22} {min(vals):.4f} to {max(vals):.4f}"
                  f"   {span:5.1f} % of cap")
        gs = [r["G"] for r in rows]
        print(f"  protection G ranges {min(gs) * 100:.1f} to {max(gs) * 100:.1f} %")
        print("  the formulation is close to fixed; G and absolute RSD are what "
              "actually personalise")


if __name__ == "__main__":
    main()

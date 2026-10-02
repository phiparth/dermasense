"""
Properties the spec claims, asserted rather than described.

Run:  python -m pytest test_model.py -q
      or just: python test_model.py
"""

import numpy as np

import dermasense_model as dm
from dermasense_model import (
    CAPS, DOSE_ORDER, NOMINAL, A0_COMPROMISED,
    Dose, Environment,
    conservative_params, evaluate, optimal_dose, optimal_scale, protection,
    per_compound_optima, single_compound_optima,
)

REF = Environment()                       # 80, 60, 1.0, type I, healthy, lag 0
REC = Dose(0.0862, 0.300, 0.010, 1.000)   # the spec's recommended dose


# ---------------------------------------------------------------- fixed scale

def test_reference_scores_exactly_one():
    assert abs(evaluate(REF, Dose())["RSD"] - 1.0) < 1e-12


def test_clean_air_scores_exactly_zero():
    clean = Environment(C_PM=0.0, C_O3=0.0, I_UV=0.0)
    assert abs(evaluate(clean, Dose())["RSD"]) < 1e-12


# ------------------------------------------------------------------ reduction

def test_zero_dose_recovers_part_one_at_arbitrary_exposure():
    """Part 2 is a modification of Part 1, not a separate model sharing
    notation. Any future edit that breaks this has introduced a bug."""
    rng = np.random.default_rng(7)
    for _ in range(50):
        e = Environment(
            C_PM=float(rng.uniform(0, 400)),
            C_O3=float(rng.uniform(0, 300)),
            I_UV=float(rng.uniform(0, 3)),
            PT=int(rng.integers(1, 7)),
            A0=float(rng.choice([1.0, A0_COMPROMISED])),
            lag_h=float(rng.uniform(0, 12)),
        )
        r = evaluate(e, Dose())
        assert abs(r["RSD"] - r["RSD0"]) < 1e-12


# ---------------------------------------------------------------- boundedness

def test_no_dose_drives_damage_negative():
    rng = np.random.default_rng(11)
    caps = np.array([CAPS[k] for k in DOSE_ORDER])
    worst = np.inf
    for _ in range(4000):
        d = Dose.from_array(rng.uniform(0, 1, 4) * caps)
        worst = min(worst, evaluate(REF, d)["RSD"])
    assert worst > 0.0


# ------------------------------------------------------- the worked trace, S11

def test_worked_trace_matches_the_specification():
    r = evaluate(REF, REC)
    expected = {
        "theta": 0.963, "sigma_PM": 0.567, "sigma_gas": 0.870,
        "C_PM_t": 45.33, "C_O3_t": 52.20,
        "SPF": 3.047, "tau_UV": 0.328,
        "chi": 0.570, "fenton": 0.715,
        "R_gen_t": 0.549,
        "kappa_endo": 1.645, "kappa_endo_t": 1.979, "kappa_app": 1.908,
        "N_R": 0.378, "R": 1.000, "R_prime": 0.297,
        "dT_ref": 4.040, "dT_t": 2.471, "Gamma_H": 0.880, "Psi": 1.000,
        "B": 1.000, "B_prime": 0.538,
        "Omega": 0.0451, "D_s": 0.549, "penalty": 0.0248,
        "RSD0": 1.0000, "RSD": 0.4424, "G": 0.5576,
        "T": 20.54, "T_prime": 16.69,
    }
    for k, want in expected.items():
        got = r[k]
        tol = max(abs(want) * 5e-3, 5e-4)
        assert abs(got - want) < tol, f"{k}: got {got:.6f}, spec says {want}"


def test_tewl_improvement_is_18_7_percent():
    assert abs(evaluate(REF, REC)["TEWL_improvement"] - 0.187) < 2e-3


# ------------------------------------------------------------------- optimum

def test_optimum_reproduces_the_recommended_dose():
    d, g = optimal_dose(REF)
    assert abs(g - 0.5576) < 2e-3
    assert abs(d.P - 0.0862) < 1e-3
    assert abs(d.X - CAPS["X"]) < 1e-6
    assert abs(d.L - CAPS["L"]) < 1e-9
    assert abs(d.H - CAPS["H"]) < 1e-6


def test_only_pulcherrimin_has_an_interior_optimum():
    d, _ = optimal_dose(REF)
    assert d.P < CAPS["P"] - 1e-3
    for k in ("X", "L", "H"):
        assert getattr(d, k) >= CAPS[k] - 1e-6


def test_unimodal_along_a_ray():
    ratio = np.array([CAPS[k] for k in DOSE_ORDER])
    ss = np.linspace(1e-6, 1.0, 4000)
    g = np.array([protection(REF, Dose.from_array(s * ratio)) for s in ss])
    sign_changes = np.sum(np.diff(np.sign(np.diff(g))) != 0)
    assert sign_changes <= 1
    s_star, _, g_star = optimal_scale(REF, ratio)
    assert abs(g_star - g.max()) < 1e-4


# ------------------------------------------------------------ sub-additivity

def test_naive_sum_of_singles_far_exceeds_the_joint_result():
    per_c = per_compound_optima(REF)
    _, g_joint = optimal_dose(REF)
    assert sum(per_c.values()) > 1.0
    assert g_joint < 0.60


# ------------------------------------------------- the interpretation trap

def test_compromised_skin_gets_higher_G_and_higher_absolute_damage():
    """The reason G must never be reported alone."""
    delhi_h = Environment(180, 70, 0.8, 4, 1.00)
    delhi_c = Environment(180, 70, 0.8, 4, A0_COMPROMISED)
    d_h, g_h = optimal_dose(delhi_h)
    d_c, g_c = optimal_dose(delhi_c)
    r_h, r_c = evaluate(delhi_h, d_h), evaluate(delhi_c, d_c)
    assert g_c > g_h
    assert r_c["RSD0"] > r_h["RSD0"]


# ---------------------------------------------------------------- sensor lag

def test_lag_costs_about_three_points_over_six_hours():
    g0 = optimal_dose(Environment(lag_h=0.0))[1]
    g6 = optimal_dose(Environment(lag_h=6.0))[1]
    assert 0.02 < g0 - g6 < 0.04


def test_recommended_pulcherrimin_barely_moves_with_lag():
    p0 = optimal_dose(Environment(lag_h=0.0))[0].P
    p6 = optimal_dose(Environment(lag_h=6.0))[0].P
    assert abs(p6 - p0) < 0.002


# ------------------------------------------------------- exposure invariance

def test_three_of_four_coordinates_are_exposure_invariant():
    """X, L and H sit on their caps for every exposure, because nothing inside
    the domain penalises them. Stated as a test so nobody can quietly claim the
    model personalises all four."""
    for pm in (20, 80, 180, 300):
        for pt in (1, 4, 6):
            d, _ = optimal_dose(Environment(C_PM=pm, PT=pt))
            for k in ("X", "L", "H"):
                assert abs(getattr(d, k) - CAPS[k]) < 1e-6


def test_pulcherrimin_is_the_only_coordinate_that_responds_to_exposure():
    """And it responds weakly: the S6 reversal threshold has no environment
    dependence, so P* only moves through the D_s / RSD0 ratio and the phototype
    modifier. Across a wide grid that is a small fraction of its cap."""
    ps = []
    for pm in (20, 80, 180, 300):
        for pt in (1, 4, 6):
            ps.append(optimal_dose(Environment(C_PM=pm, PT=pt))[0].P)
    spread = (max(ps) - min(ps)) / CAPS["P"]
    assert 0.01 < spread < 0.20, spread


# ---------------------------------------------------------------- pessimism

def test_conservative_preset_is_much_worse_than_nominal():
    g_nom = optimal_dose(REF)[1]
    g_con = optimal_dose(REF, conservative_params("C"))[1]
    assert g_con < g_nom - 0.15


# ------------------------------------------------------------ the control arm

def test_control_is_off_by_default():
    """Every number in the specification has to survive the new coordinate. The
    S slot is zero unless something asks for it, so it cannot move a spec value."""
    assert Dose().S == 0.0
    assert abs(evaluate(REF, REC)["RSD"] - 0.4424) < 5e-3
    d, g = optimal_dose(REF)
    assert d.S == 0.0
    assert abs(g - 0.5576) < 2e-3


def test_control_runs_through_the_same_equations():
    """S is not a bolted-on term: it enters the same Langmuir denominator, the
    same stripping sum and the same self-toxicity sum as the lab surfactants.
    Giving it the lab constants must make it behave exactly like xylolipid."""
    p = NOMINAL.with_values(CMC_S=NOMINAL.K_X_ads, C_crit_S=NOMINAL.C_crit,
                            strip_S=NOMINAL.strip)
    as_X = evaluate(REF, Dose(X=0.2), p)
    as_S = evaluate(REF, Dose(S=0.2), p)
    for k in ("theta", "sigma_PM", "sigma_gas", "Psi", "B_prime"):
        assert abs(as_X[k] - as_S[k]) < 1e-12, k
    # the one asymmetry is deliberate: xylolipid also scavenges (S3), the
    # benchmark surfactant is a film former only.
    assert as_X["kappa_app"] > as_S["kappa_app"]


def test_film_formers_compete_for_one_interface():
    """Coverage is sub-additive across all three, because there is one surface."""
    p = dm.benchmark_params("sophorolipid")
    th_lab = evaluate(REF, Dose(X=CAPS["X"], L=CAPS["L"]), p)["theta"]
    th_both = evaluate(REF, Dose(X=CAPS["X"], L=CAPS["L"], S=0.5), p)["theta"]
    assert th_both > th_lab
    assert th_both < 1.0
    assert th_both - th_lab < 0.05      # a saturated film has little left to give


def test_coverage_equivalence_is_exact():
    """Langmuir occupancies add, so the mass of control that reproduces the lab
    pair's coverage can be solved in closed form. The page quotes that number,
    so it has to match what the model actually does."""
    p = dm.benchmark_params("sophorolipid")
    eq = dm.control_equivalent_dose(p)
    got = evaluate(REF, Dose(S=eq["S_equivalent"]), p)["theta"]
    assert abs(got - eq["theta_lab"]) < 1e-9


def test_sds_has_no_window_between_film_and_harm():
    """The negative control has to actually behave like one: the poorest film
    former of the lot, among the two least tolerated (surfactin, a lipopeptide
    with a keratinocyte LC50 of 0.08 mg/mL, is as toxic), and so last on
    protection."""
    sds = dm.BENCHMARKS["sds"]
    others = [b for k, b in dm.BENCHMARKS.items() if k != "sds"]
    assert all(sds.CMC_S > b.CMC_S for b in others)
    tolerance_rank = sorted(dm.BENCHMARKS.values(), key=lambda b: b.C_crit_S)
    assert sds in tolerance_rank[:2]
    g = {k: optimal_dose(REF, dm.benchmark_params(k), ("S",))[1]
         for k in dm.BENCHMARKS}
    assert g["sds"] == min(g.values())


def test_potent_biosurfactants_trade_efficiency_for_toxicity():
    """Rhamnolipid and surfactin form a film at a fraction of the mass and kill
    keratinocytes at a fraction of the dose, so they are the benchmarks with a
    genuine interior optimum. Sophorolipid is gentle and weak, so it just runs
    to the cap. That trade is what an interior optimum is made of."""
    b = dm.BENCHMARKS
    assert b["surfactin"].CMC_S < b["rhamnolipid"].CMC_S < b["sophorolipid"].CMC_S
    assert b["surfactin"].C_crit_S < b["rhamnolipid"].C_crit_S < b["sophorolipid"].C_crit_S
    for k in ("rhamnolipid", "surfactin"):
        p = dm.benchmark_params(k)
        d, _ = optimal_dose(REF, p, ("S",))
        assert d.S < 0.5 * p.S_cap, k
    p = dm.benchmark_params("sophorolipid")
    assert optimal_dose(REF, p, ("S",))[0].S >= p.S_cap - 1e-6


def test_the_control_adds_almost_nothing_once_the_lab_film_is_saturated():
    """With X and L at their caps the film is ~96% covered, so a third film
    former has little left to buy. Stated as the size of the gain rather than as
    where the optimiser lands, because on a plateau the landing point is
    arbitrary."""
    for k in dm.BENCHMARKS:
        p = dm.benchmark_params(k)
        g_spec = optimal_dose(REF, p, DOSE_ORDER)[1]
        g_all = optimal_dose(REF, p, dm.ALL_KEYS)[1]
        assert g_all - g_spec < 0.002, k


def test_every_arm_is_scored_on_its_own_best_dose():
    """A comparison where one arm was handed a bad dose is not a comparison."""
    p = dm.benchmark_params("sophorolipid")
    rows = {r["arm"]: r for r in dm.control_comparison(REF, p)}
    for label, keys in dm.CONTROL_ARMS.items():
        r = rows[label]
        for k in dm.ALL_KEYS:
            if k not in keys:
                assert r[k] == 0.0, f"{label} used {k}"
        assert abs(r["G"] - optimal_dose(REF, p, keys)[1]) < 1e-9


def test_adding_arms_never_lowers_the_best_achievable_protection():
    """Superset of coordinates, superset of options. If this fails the
    optimiser is getting stuck, not the model saying something interesting."""
    p = dm.benchmark_params("sophorolipid")
    g_lab = optimal_dose(REF, p, dm.LAB_KEYS)[1]
    g_spec = optimal_dose(REF, p, DOSE_ORDER)[1]
    g_all = optimal_dose(REF, p, dm.ALL_KEYS)[1]
    assert g_spec >= g_lab - 1e-6
    assert g_all >= g_spec - 1e-6


# ------------------------------------------------------------ the market

def test_percent_to_model_units():
    """1 % w/w is 10 mg/mL, so a 1 % hyaluronic acid serum is ten times the
    model's own hyaluronic acid ceiling. That one conversion is the headline of
    the market comparison, so it is pinned."""
    assert dm.PCT_TO_MG_PER_ML == 10.0
    assert abs(dm.MARKET_RANGES["H"].mg_per_ml("high") - 20.0) < 1e-12
    assert dm.MARKET_RANGES["H"].mg_per_ml("low") == CAPS["H"]


def test_typical_is_inside_the_published_range():
    for r in dm.MARKET_RANGES.values():
        assert r.low < r.typical < r.high


def test_domain_labels_and_clipping():
    p = dm.benchmark_params("sophorolipid")
    inside = Dose(H=0.5, S=0.5)
    assert dm.domain_label(dm.domain_factor(inside, p)) == "inside the validity domain"
    big = Dose(H=20.0, S=50.0)
    assert dm.domain_label(dm.domain_factor(big, p)) == "outside the model"
    c = dm.clipped(big, p)
    assert c.H == CAPS["H"] and c.S == p.S_cap


def test_market_low_end_is_inside_the_model_and_high_end_is_not():
    rows = dm.market_scenarios(REF, dm.benchmark_params("sophorolipid"))
    by = {r["level"]: r for r in rows}
    assert by["low"]["domain"] == "inside the validity domain"
    assert by["high"]["domain"] == "outside the model"
    # inside the domain there is nothing to clip
    assert abs(by["low"]["G_raw"] - by["low"]["G_clipped"]) < 1e-12


def test_clipped_market_score_never_exceeds_the_lab_optimum():
    """Capped at the evidence, a market formulation cannot beat the optimum the
    optimiser finds inside the same caps: that is what optimal means."""
    p = dm.benchmark_params("sophorolipid")
    best = optimal_dose(REF, p, dm.ALL_KEYS)[1]
    for r in dm.market_scenarios(REF, p):
        assert r["G_clipped"] <= best + 1e-9


def test_formulation_ratio_counts_the_right_things():
    d = Dose(P=0.1, X=0.2, L=0.01, H=1.0, S=0.3)
    r = dm.formulation_ratio(d)
    assert abs(r["antioxidant"] - 1.1) < 1e-12
    assert abs(r["surfactant"] - 0.51) < 1e-12
    assert abs(r["total"] - 1.61) < 1e-12


def test_ratio_sweep_at_the_optimum_mass_reproduces_the_optimum():
    d, g = optimal_dose(REF)
    tot = sum(d.as_array(dm.ALL_KEYS))
    got = dm.ratio_sweep(REF, NOMINAL, d, [tot])[0]
    assert abs(got - g) < 1e-9


def test_grid_matches_direct_evaluation():
    p = dm.benchmark_params("sophorolipid")
    g = dm.grid_G(REF, p, "S", np.array([1.0]), np.array([0.5]))[0, 0]
    assert abs(g - protection(REF, Dose(H=1.0, S=0.5), p)) < 1e-12
    x_share = dm.LAB_FILM_SHAPE[0] / sum(dm.LAB_FILM_SHAPE)
    g2 = dm.grid_G(REF, p, "XL", np.array([1.0]), np.array([0.31]))[0, 0]
    d = Dose(H=1.0, X=0.31 * x_share, L=0.31 * (1 - x_share))
    assert abs(g2 - protection(REF, d, p)) < 1e-12


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")

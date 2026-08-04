"""
Properties the spec claims, asserted rather than described.

Run:  python -m pytest test_model.py -q
      or just: python test_model.py
"""

import numpy as np

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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")

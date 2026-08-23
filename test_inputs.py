"""
Properties the input layer has to have, asserted rather than hoped for.

The photo tests run on synthetic skin patches, because the failure modes worth
guarding are geometric, not aesthetic: does a darker patch score a higher
phototype, does sensor noise read as flaking, does the edge of the frame read
as a rash. A real face would test none of those any better and could not be
committed to a public repo.

Run:  python test_inputs.py       or: python -m pytest test_inputs.py -q
"""

import numpy as np
from PIL import Image

import skin_inputs as si
from env_api import LiveEnvironment, Place, UVI_REF, domain_warnings

# ---------------------------------------------------------------- synthetics

TONES = {
    1: (247, 217, 205), 2: (228, 190, 168), 3: (205, 160, 130),
    4: (178, 133, 104), 5: (126, 88, 66), 6: (74, 50, 38),
}


def skin_patch(tone, texture=0.02, redness=0.0, noise=2.0, background=True,
               size=320, seed=0):
    """A flat skin colour, multiplicative texture at a few-pixel scale, optional
    blotchy erythema, sensor noise, and a neutral grey band to stand in for a
    background."""
    r = np.random.default_rng(seed)
    a = np.zeros((size, size, 3)) + np.array(tone, dtype=float)

    field = r.normal(0, 1, (size, size))
    field = (field - field.min()) / max(field.max() - field.min(), 1e-9) * 255
    field = np.asarray(Image.fromarray(field.astype(np.uint8))
                       .resize((size // 3, size // 3)).resize((size, size)), dtype=float)
    field = (field - field.mean()) / (field.std() or 1.0)
    a *= (1.0 + texture * field)[..., None]

    if redness > 0:
        ys, xs = np.mgrid[0:size, 0:size]
        blob = np.zeros((size, size))
        rr = np.random.default_rng(seed + 5)
        for _ in range(4):
            cy, cx = rr.integers(0, size), rr.integers(0, size)
            blob += np.exp(-(((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * (size / 12) ** 2)))
        blob = np.clip(blob, 0, 1)
        a[..., 0] += 40 * redness * blob
        a[..., 1] -= 16 * redness * blob
        a[..., 2] -= 12 * redness * blob

    a += r.normal(0, noise, a.shape)
    if background:
        a[:size // 5, :, :] = 150 + r.normal(0, 3, (size // 5, size, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# ------------------------------------------------------------ questionnaires

def test_fitzpatrick_extremes_hit_the_end_types():
    palest = {q.key: q.options[0][0] for q in si.FITZPATRICK_QUESTIONS}
    darkest = {q.key: q.options[-1][0] for q in si.FITZPATRICK_QUESTIONS}
    assert si.phototype_from_quiz(palest).value == 1
    assert si.phototype_from_quiz(darkest).value == 6


def test_fitzpatrick_is_monotone_in_the_score():
    seen = []
    for i in range(5):
        ans = {q.key: q.options[i][0] for q in si.FITZPATRICK_QUESTIONS}
        r = si.phototype_from_quiz(ans)
        seen.append((r.total, r.value))
    totals = [t for t, _ in seen]
    types = [v for _, v in seen]
    assert totals == sorted(totals)
    assert types == sorted(types)


def test_poem_zero_is_healthy_and_saturates_at_compromised():
    clear = {q.key: q.options[0][0] for q in si.POEM_QUESTIONS}
    worst = {q.key: q.options[-1][0] for q in si.POEM_QUESTIONS}
    assert si.barrier_from_quiz(clear).value == si.A0_HEALTHY
    assert si.barrier_from_quiz(worst).value == si.A0_COMPROMISED


def test_poem_is_monotone_and_stays_inside_the_two_anchors():
    prev = 2.0
    for i in range(5):
        ans = {q.key: q.options[i][0] for q in si.POEM_QUESTIONS}
        a0 = si.barrier_from_quiz(ans).value
        assert si.A0_COMPROMISED <= a0 <= si.A0_HEALTHY
        assert a0 <= prev
        prev = a0


def test_override_forces_compromised():
    clear = {q.key: q.options[0][0] for q in si.POEM_QUESTIONS}
    assert si.barrier_from_quiz(clear, disrupted_override=True).value == si.A0_COMPROMISED


def test_snap_binary_returns_only_the_specs_two_values():
    for a0 in np.linspace(0.65, 1.0, 15):
        assert si.snap_binary(float(a0)) in (si.A0_HEALTHY, si.A0_COMPROMISED)


# -------------------------------------------------------------------- photos

def test_darker_patch_never_scores_a_lighter_phototype():
    pts = [si.analyze_photo(skin_patch(TONES[k])).PT for k in sorted(TONES)]
    assert pts == sorted(pts), pts
    assert pts[0] == 1 and pts[-1] == 6


def test_ita_falls_monotonically_with_tone():
    itas = [si.analyze_photo(skin_patch(TONES[k])).ita for k in sorted(TONES)]
    assert itas == sorted(itas, reverse=True), itas


def test_calm_skin_reads_as_a_healthy_barrier_at_every_tone():
    """The fairness test. A dark photo is noisier per unit signal and has a
    harder edge against the background; neither is a skin finding."""
    for k in sorted(TONES):
        r = si.analyze_photo(skin_patch(TONES[k]))
        assert r.A0 > 0.93, (k, r.A0, r.texture, r.erythema_spread)


def test_sensor_noise_alone_does_not_move_the_barrier():
    quiet = si.analyze_photo(skin_patch(TONES[3], noise=1.0))
    loud = si.analyze_photo(skin_patch(TONES[3], noise=6.0))
    assert abs(quiet.A0 - loud.A0) < 0.05


def test_flaking_and_redness_lower_the_barrier_at_every_tone():
    for k in sorted(TONES):
        calm = si.analyze_photo(skin_patch(TONES[k]))
        bad = si.analyze_photo(skin_patch(TONES[k], texture=0.09, redness=1.0, seed=3))
        assert bad.A0 < calm.A0 - 0.1, (k, calm.A0, bad.A0)
        assert bad.severity > 0.3


def test_barrier_stays_inside_the_two_anchors():
    for tex in (0.0, 0.05, 0.2, 0.6):
        r = si.analyze_photo(skin_patch(TONES[2], texture=tex, redness=1.0))
        assert si.A0_COMPROMISED <= r.A0 <= si.A0_HEALTHY


def test_phototype_survives_a_bad_barrier():
    """Redness moves a*, and a* is in the ITA denominator only through b*, so a
    rash must not be read as a change of ethnicity."""
    for k in (2, 5):
        calm = si.analyze_photo(skin_patch(TONES[k]))
        bad = si.analyze_photo(skin_patch(TONES[k], texture=0.09, redness=1.0, seed=3))
        assert abs(calm.PT - bad.PT) <= 1, (k, calm.PT, bad.PT)


def test_a_frame_with_no_skin_warns_rather_than_inventing_an_answer():
    grey = Image.fromarray(np.full((200, 200, 3), 128, dtype=np.uint8))
    r = si.analyze_photo(grey)
    assert r.warnings
    assert 1 <= r.PT <= 6


def test_no_deep_model_installed_is_a_quiet_none():
    assert si.deep_phototype(np.zeros((64, 64, 3))) is None or si.deep_available()


# ------------------------------------------------------------- reconciliation

def test_reconcile_takes_the_darker_type():
    assert si.reconcile(2, 5)[0] == 5
    assert si.reconcile(5, 2)[0] == 5
    assert si.reconcile(None, 3)[0] == 3
    assert si.reconcile(3, None)[0] == 3


# --------------------------------------------------------------- environment

def _live(pm=80.0, o3=60.0, uv_now=1.0, uv_peak=8.0):
    return LiveEnvironment(place=Place("Test", 0.0, 0.0), observed_at="now",
                           pm25=pm, o3=o3, uv_now=uv_now, uv_peak_today=uv_peak)


def test_feed_units_pass_through_untouched():
    mi = _live().model_inputs()
    assert mi["C_PM"] == 80.0 and mi["C_O3"] == 60.0


def test_uv_normalisation_is_the_only_conversion():
    assert _live(uv_peak=UVI_REF).model_inputs("peak")["I_UV"] == 1.0
    assert _live(uv_now=4.0).model_inputs("now")["I_UV"] == 4.0 / UVI_REF


def test_peak_is_the_default_because_now_is_zero_at_night():
    night = _live(uv_now=0.0, uv_peak=9.0)
    assert night.model_inputs()["I_UV"] > 1.0
    assert night.model_inputs("now")["I_UV"] == 0.0


def test_domain_warnings_fire_outside_the_fitted_range():
    assert domain_warnings({"C_PM": 350.0, "C_O3": 60.0, "I_UV": 1.0})
    assert not domain_warnings({"C_PM": 80.0, "C_O3": 60.0, "I_UV": 1.0})


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"pass  {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL  {name}: {exc}")
    print("\nall input tests passed" if not fails else f"\n{fails} failed")
    raise SystemExit(1 if fails else 0)

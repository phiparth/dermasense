"""
Skin capture: the two person-level inputs the model needs, obtained from a
person rather than from a slider.

    PT   Fitzpatrick phototype, 1-6   -> acts on the UV channel only, f_PT
    A0   barrier state, 1.00 or 0.65  -> scales the endogenous antioxidant reserve

Two routes to each, and they disagree on purpose:

  questionnaire   the Fitzpatrick self-report items (Fitzpatrick 1988) for PT,
                  POEM (Charman 2004) for the barrier. Both are instruments
                  people already use, scored exactly as published, so the
                  numbers are defensible even though the mapping POEM -> A0 is
                  ours.

  photo           ITA degrees from CIE L*a*b* for PT (Chardon 1991, classes from
                  Del Bino 2013), plus two barrier proxies, erythema spread and
                  surface texture, that nobody has calibrated against TEWL.
                  Deterministic, no weights, no upload: the arithmetic runs
                  wherever the app runs.

Evidence, honestly: the Fitzpatrick items and the POEM items are class A as
instruments. ITA and its class boundaries are class A. Everything that maps an
instrument onto a model input - POEM -> A0, ITA class -> phototype, the two
photo barrier proxies - is class C, our choice, and every function here returns
that label with its number so the interface can say so.

No Streamlit import: usable from a notebook or the CLI.

    python skin_inputs.py --photo cheek.jpg
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

A0_HEALTHY = 1.00
A0_COMPROMISED = 0.65
PT_ROMAN = ("I", "II", "III", "IV", "V", "VI")


def clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


# ===========================================================================
# 1. questionnaires
# ===========================================================================

@dataclass(frozen=True)
class Question:
    key: str
    group: str
    prompt: str
    options: Tuple[Tuple[str, int], ...]   # (label shown, score)

    @property
    def labels(self) -> List[str]:
        return [o[0] for o in self.options]

    def score(self, label: str) -> int:
        for lab, s in self.options:
            if lab == label:
                return s
        raise KeyError(f"{label!r} is not an option of {self.key}")


# --- Fitzpatrick self-report ------------------------------------------------
# Fitzpatrick TB, Arch Dermatol 1988;124(6):869-71, in the ten-item self-report
# form that survived into practice: four constitutional items, three on the
# reaction to sun, three on habitual behaviour. Each 0-4, total 0-40.
# Self-report overestimates type on darker skin (Roberts 2009; Ho 2020), which
# is exactly the direction that would under-dose the UV screen, so the app says
# so next to the answer.

_FITZ_OPTS = {
    "eye": (("Light blue, light grey, light green", 0), ("Blue, grey, green", 1),
            ("Hazel, light brown", 2), ("Dark brown", 3), ("Brownish black, black", 4)),
    "hair": (("Red, light blond", 0), ("Blond, light brown", 1), ("Dark blond, brown", 2),
             ("Dark brown", 3), ("Black", 4)),
    "skin": (("Reddish, very pale", 0), ("Very pale", 1), ("Pale with a beige tint", 2),
             ("Light brown", 3), ("Dark brown, black", 4)),
    "freckles": (("Many", 0), ("Several", 1), ("Few", 2), ("Very few", 3), ("None", 4)),
    "burn": (("Painful burn, blistering, peeling", 0), ("Blistering then peeling", 1),
             ("Burn, sometimes peeling", 2), ("Rare burn", 3), ("Never burn", 4)),
    "tan": (("Never, I stay pale", 0), ("Rarely", 1), ("Sometimes", 2),
            ("Often", 3), ("Always", 4)),
    "brown": (("Never", 0), ("Seldom", 1), ("Sometimes", 2), ("Often", 3), ("Always", 4)),
    "face_react": (("Very sensitive", 0), ("Sensitive", 1), ("Normal", 2),
                   ("Very resistant", 3), ("Never had a problem", 4)),
    "last_exposed": (("More than 3 months ago", 0), ("2-3 months ago", 1),
                     ("1-2 months ago", 2), ("Less than a month ago", 3),
                     ("Less than 2 weeks ago", 4)),
    "face_exposed": (("Never", 0), ("Hardly ever", 1), ("Sometimes", 2),
                     ("Often", 3), ("Always", 4)),
}

FITZPATRICK_QUESTIONS: Tuple[Question, ...] = (
    Question("eye", "Constitution", "Eye colour", _FITZ_OPTS["eye"]),
    Question("hair", "Constitution", "Natural hair colour", _FITZ_OPTS["hair"]),
    Question("skin", "Constitution", "Natural skin colour, where the sun never reaches", _FITZ_OPTS["skin"]),
    Question("freckles", "Constitution", "Freckles on unexposed skin", _FITZ_OPTS["freckles"]),
    Question("burn", "Reaction to sun", "After a long stretch in the sun your skin", _FITZ_OPTS["burn"]),
    Question("tan", "Reaction to sun", "Do you tan?", _FITZ_OPTS["tan"]),
    Question("brown", "Reaction to sun", "Does your skin turn brown?", _FITZ_OPTS["brown"]),
    Question("face_react", "Habits", "Your face reacts to the sun as", _FITZ_OPTS["face_react"]),
    Question("last_exposed", "Habits", "When were you last out in strong sun?", _FITZ_OPTS["last_exposed"]),
    Question("face_exposed", "Habits", "Do you expose your face to the sun?", _FITZ_OPTS["face_exposed"]),
)

# published band edges for the ten-item form
FITZ_BANDS: Tuple[Tuple[int, int, int], ...] = (
    (0, 6, 1), (7, 13, 2), (14, 20, 3), (21, 27, 4), (28, 34, 5), (35, 40, 6),
)

# --- POEM, for the barrier --------------------------------------------------
# Charman CR, Venn AJ, Williams HC, Arch Dermatol 2004;140(12):1513-9.
# Seven items, "how many days in the last week", 0-4 each, total 0-28.
# Bands: 0-2 clear, 3-7 mild, 8-16 moderate, 17-24 severe, 25-28 very severe.
# We use it as a barrier-symptom score, not as an eczema diagnosis: the items
# ask about dryness, cracking and flaking, which is what A0 is standing in for.

_POEM_OPTS = (("No days", 0), ("1-2 days", 1), ("3-4 days", 2),
              ("5-6 days", 3), ("Every day", 4))

POEM_QUESTIONS: Tuple[Question, ...] = tuple(
    Question(k, "Last 7 days", p, _POEM_OPTS) for k, p in (
        ("itch", "Days your skin was itchy"),
        ("sleep", "Days your sleep was disturbed by your skin"),
        ("bleed", "Days your skin was bleeding"),
        ("weep", "Days your skin was weeping or oozing clear fluid"),
        ("crack", "Days your skin was cracked"),
        ("flake", "Days your skin was flaking off"),
        ("dry", "Days your skin felt dry or rough"),
    )
)

POEM_BANDS: Tuple[Tuple[int, int, str], ...] = (
    (0, 2, "clear or almost clear"), (3, 7, "mild"), (8, 16, "moderate"),
    (17, 24, "severe"), (25, 28, "very severe"),
)

# class C. The model's A0 is binary, 1.00 or 0.65, and Kim 2016 gave the
# direction only. We read the POEM total as a continuous position between the
# two, saturating at 16, the top of the moderate band, because a moderate POEM
# is already a barrier a dermatologist would call disrupted. Nothing downstream
# requires A0 to be binary; snap_binary() is there for anyone who wants the
# spec's literal two values.
POEM_SATURATES_AT = 16.0


@dataclass(frozen=True)
class QuizResult:
    total: int
    max_total: int
    value: float                    # PT (1-6) or A0 (0.65-1.00)
    label: str
    per_group: Dict[str, int] = field(default_factory=dict)
    evidence: str = "A instrument, C mapping"
    notes: List[str] = field(default_factory=list)


def _band(total: int, bands) -> Tuple[int, int, object]:
    for lo, hi, v in bands:
        if lo <= total <= hi:
            return lo, hi, v
    return bands[-1]


def phototype_from_quiz(answers: Dict[str, str]) -> QuizResult:
    """answers maps question key -> the option label the person picked."""
    per_group: Dict[str, int] = {}
    total = 0
    for q in FITZPATRICK_QUESTIONS:
        if q.key not in answers:
            raise KeyError(f"missing answer for {q.key}")
        s = q.score(answers[q.key])
        total += s
        per_group[q.group] = per_group.get(q.group, 0) + s
    lo, hi, pt = _band(total, FITZ_BANDS)
    notes = [
        "Self-reported phototype tracks measured pigmentation poorly above type "
        "IV; if the photo route disagrees, trust the darker of the two.",
    ]
    if total >= 21:
        notes.append(
            "Types IV to VI: f_PT cuts the UV channel by 14% per step, so this "
            "answer alone lowers the recommended pulcherrimin."
        )
    return QuizResult(total=total, max_total=40, value=float(pt),
                      label=f"Type {PT_ROMAN[pt - 1]} (score {total}, band {lo}-{hi})",
                      per_group=per_group, notes=notes)


def barrier_from_quiz(answers: Dict[str, str], disrupted_override: bool = False) -> QuizResult:
    """POEM total -> A0. disrupted_override forces the compromised value for the
    barrier insults POEM does not ask about: an active retinoid or acid, a fresh
    peel, a shave rash, a week of cold dry wind."""
    total = 0
    for q in POEM_QUESTIONS:
        if q.key not in answers:
            raise KeyError(f"missing answer for {q.key}")
        total += q.score(answers[q.key])
    lo, hi, sev = _band(total, POEM_BANDS)

    frac = clamp(total / POEM_SATURATES_AT, 0.0, 1.0)
    a0 = A0_HEALTHY - (A0_HEALTHY - A0_COMPROMISED) * frac
    notes = []
    if disrupted_override:
        a0 = min(a0, A0_COMPROMISED)
        notes.append("Forced to the compromised value by the override.")
    if total >= 8:
        notes.append(
            "A compromised barrier scores a HIGHER protection G at a HIGHER "
            "absolute damage. Read the two together or the app will look like "
            "it is recommending less product for the skin that needs more."
        )
    return QuizResult(total=total, max_total=28, value=round(a0, 4),
                      label=f"POEM {total}/28, {sev} -> A0 {a0:.2f}",
                      notes=notes)


def snap_binary(a0: float, cut: float = 0.85) -> float:
    """Collapse a continuous A0 back onto the spec's two literal values."""
    return A0_HEALTHY if a0 >= cut else A0_COMPROMISED


# ===========================================================================
# 2. photo route
# ===========================================================================
#
# What model to deploy for the image, and why this one:
#
#   ITA degrees, what is implemented here. Chardon's individual typology angle
#   is the instrument dermatology and cosmetic science actually use for
#   constitutive pigmentation, its class edges are published (Del Bino 2013),
#   it is about forty lines of arithmetic, it has no training set to inherit
#   bias from, it runs in milliseconds on a free Streamlit dyno, and every step
#   can be shown on the wiki. Its weakness is illumination: an uncorrected
#   white balance moves ITA by tens of degrees, so the app warns on colour cast
#   and offers a grey-world correction.
#
#   A CNN, e.g. MobileNetV3-Small or EfficientNet-B0 fine-tuned on
#   Fitzpatrick17k and exported to int8 ONNX (about 6 MB, ~30 ms on CPU under
#   onnxruntime, which does fit the free tier). Not the default, for three
#   reasons: Fitzpatrick17k labels are annotator-assigned rather than measured
#   and are noisy above type IV, its images are clinical dermatology photos and
#   ours are phone selfies, and a black box that cannot be justified on the
#   wiki is worse than arithmetic that can. The hook is below: drop an ONNX
#   file in and it is used, with its confidence shown next to the ITA answer.
#
#   Face or skin segmentation, e.g. BiSeNet or MediaPipe FaceMesh, would beat
#   the colour-rule mask below and is the first upgrade worth the dependency.
#
# Barrier from a photo is the weaker half and stays labelled that way. Erythema
# spread and surface texture move in the right direction for a disrupted
# barrier; neither has been put next to a TEWL probe on our subjects.

# class C thresholds, all of them. Both proxies are computed at a spatial scale
# that ignores single pixels: sensor noise is not a skin finding, and a blotch
# of erythema or a patch of scale is several pixels across at any sane framing.
ERY_SPREAD_FLOOR = 1.2     # a* units, p90 - p50 on a smoothed a* map, calm skin
ERY_SPREAD_FULL = 7.0      # ... and visibly blotchy skin
TEXTURE_FLOOR = 0.012      # band-pass Weber contrast of calm skin
TEXTURE_FULL = 0.060       # ... and of visibly flaking skin
BARRIER_PHOTO_WEIGHTS = (0.5, 0.5)   # erythema, texture

# Del Bino S, Bernerd F, Br J Dermatol 2013;169(s3):33-40
ITA_CLASSES: Tuple[Tuple[float, str, int], ...] = (
    (55.0, "very light", 1),
    (41.0, "light", 2),
    (28.0, "intermediate", 3),
    (10.0, "tan", 4),
    (-30.0, "brown", 5),
    (-1e9, "dark", 6),
)


@dataclass(frozen=True)
class PhotoResult:
    PT: int
    A0: float
    ita: float
    ita_class: str
    L: float
    a_star: float
    b_star: float
    erythema_spread: float
    texture: float
    skin_fraction: float
    n_pixels: int
    colour_cast: float
    white_balance: str
    severity: float
    deep: Optional[Dict[str, float]] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def pt_roman(self) -> str:
        return PT_ROMAN[self.PT - 1]


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) sRGB in 0-255 -> (..., 3) CIE L*a*b*, D65."""
    c = np.asarray(rgb, dtype=float) / 255.0
    lin = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = (0.2126 * r + 0.7152 * g + 0.0722 * b)
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
    def f(t):
        return np.where(t > 0.008856, np.cbrt(np.maximum(t, 1e-12)), 7.787 * t + 16.0 / 116.0)
    fx, fy, fz = f(x), f(y), f(z)
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def _skin_mask(arr: np.ndarray) -> np.ndarray:
    """Two classic colour rules, unioned, then specular and shadow pixels cut.

    Kovac 2003 RGB rule for uniform daylight, plus the Chai/Ngan YCbCr box.
    Neither is a segmenter; together they are enough to keep a cheek and drop a
    wall, which is all the ITA median needs.
    """
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx, mn = arr.max(axis=2), arr.min(axis=2)
    kovac = ((r > 95) & (g > 40) & (b > 20) & ((mx - mn) > 15) &
             (np.abs(r - g) > 15) & (r > g) & (r > b))
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    ycbcr = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
    # the Kovac rule needs |r-g| > 15, which deep skin tones in shade fail, so
    # the YCbCr box carries them
    return kovac | ycbcr


def _box_blur(img: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean filter via an integral image. Keeps scipy out of this file."""
    pad = k // 2
    p = np.pad(img, pad, mode="edge")
    s = np.cumsum(np.cumsum(p, axis=0), axis=1)
    s = np.pad(s, ((1, 0), (1, 0)), mode="constant")
    h, w = img.shape
    tot = s[k:k + h, k:k + w] - s[0:h, k:k + w] - s[k:k + h, 0:w] + s[0:h, 0:w]
    return tot / float(k * k)


def _mad_sigma(v: np.ndarray) -> float:
    """Median-absolute-deviation estimate of sigma, 1.4826 * MAD."""
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med)))


def _grey_world(arr: np.ndarray) -> np.ndarray:
    """Assumes the frame averages to neutral. True when a neutral surface is in
    shot, false for a full-frame cheek, which is why it is opt-in."""
    m = arr.reshape(-1, 3).mean(axis=0)
    m = np.where(m < 1.0, 1.0, m)
    scale = m.mean() / m
    return np.clip(arr * scale, 0, 255)


def _centre_crop(arr: np.ndarray, frac: float) -> np.ndarray:
    if frac >= 0.999:
        return arr
    h, w = arr.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return arr[y0:y0 + ch, x0:x0 + cw]


def analyze_photo(image, white_balance: str = "none", crop_frac: float = 1.0,
                  max_side: int = 480) -> PhotoResult:
    """A skin photo -> PT and A0, with everything that went into them.

    image: a PIL.Image, or anything numpy can turn into an (H, W, 3) array.
    """
    try:                                   # PIL without importing it at module load
        arr = np.asarray(image.convert("RGB"), dtype=float)
    except AttributeError:
        arr = np.asarray(image, dtype=float)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("need an RGB image")
    arr = arr[..., :3]

    arr = _centre_crop(arr, clamp(crop_frac, 0.1, 1.0))
    h, w = arr.shape[:2]
    if max(h, w) > max_side:               # downscale by striding, no filter needed
        step = int(np.ceil(max(h, w) / max_side))
        arr = arr[::step, ::step]

    warnings: List[str] = []
    if white_balance == "greyworld":
        arr = _grey_world(arr)

    mask = _skin_mask(arr)
    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    total_px = mask.size
    skin_fraction = float(mask.sum()) / total_px

    if mask.sum() < 200:
        warnings.append(
            "Almost nothing in this frame looks like skin, so every pixel was used. "
            "Fill the frame with a cheek or a forehead and try again."
        )
        mask = np.ones_like(mask, dtype=bool)

    # drop blown highlights and deep shadow: both are illumination, not pigment
    lm = lum[mask]
    hi_cut, lo_cut = np.percentile(lm, 97), np.percentile(lm, 5)
    core = mask & (lum < hi_cut) & (lum > lo_cut)
    if core.sum() < 100:
        core = mask
    n_px = int(core.sum())

    lab = _srgb_to_lab(arr[core])
    L = float(np.median(lab[:, 0]))
    a_star = float(np.median(lab[:, 1]))
    b_star = float(np.median(lab[:, 2]))
    ita = float(np.degrees(np.arctan2(L - 50.0, b_star))) if abs(b_star) > 1e-6 else 90.0

    ita_class, pt = "dark", 6
    for edge, name, p in ITA_CLASSES:
        if ita > edge:
            ita_class, pt = name, p
            break

    # barrier proxies, both band-passed so pixel noise does not read as disease.
    # An 11 px window straddling the edge of the face reads as violent texture
    # and as a red streak, so step back from the mask boundary first.
    inner = core & (_box_blur(core.astype(float), 11) > 0.999)
    if inner.sum() < 100:
        inner = core

    a_map = np.full(lum.shape, a_star, dtype=float)   # fill, not zero, so the
    a_map[core] = lab[:, 1]                           # blur has nothing to pull to
    a_smooth = _box_blur(a_map, 5)[inner]
    ery_spread = float(np.percentile(a_smooth, 90) - np.percentile(a_smooth, 50))

    fine = _box_blur(lum, 3)                       # kill single-pixel noise
    coarse = _box_blur(lum, 11)                    # ... and the illumination gradient
    denom = np.maximum(coarse, 1.0)
    band = ((fine - coarse) / denom)[inner]        # Weber, so tone drops out
    noise = ((lum - fine) / denom)[inner]          # what is left is mostly sensor
    # Robust scale, because a nostril or a stray hair inside the mask is an
    # outlier and std would hand it the whole measurement.
    s_band, s_noise = _mad_sigma(band), _mad_sigma(noise)
    # Subtract the noise floor. A 3x3 mean divides noise variance by 9 and
    # (lum - fine) keeps 8/9 of it, so the share still inside `band` is
    # var(noise)/8. Without this a photo of deep skin scores as flaking, purely
    # because the same sensor noise is a larger fraction of a smaller signal.
    texture = float(np.sqrt(max(s_band ** 2 - s_noise ** 2 / 8.0, 0.0)))

    s_ery = clamp((ery_spread - ERY_SPREAD_FLOOR) / (ERY_SPREAD_FULL - ERY_SPREAD_FLOOR),
                  0.0, 1.0)
    s_tex = clamp((texture - TEXTURE_FLOOR) / (TEXTURE_FULL - TEXTURE_FLOOR), 0.0, 1.0)
    w_e, w_t = BARRIER_PHOTO_WEIGHTS
    severity = w_e * s_ery + w_t * s_tex
    a0 = clamp(A0_HEALTHY - (A0_HEALTHY - A0_COMPROMISED) * severity,
               A0_COMPROMISED, A0_HEALTHY)

    # Colour cast is judged on what is NOT skin. Skin is warm by definition, so
    # measuring the cast on the cheek itself would report every face as lit by a
    # tungsten bulb. With no background in shot there is nothing to judge it on,
    # and the interface says that instead of guessing.
    non_skin = ~mask
    if float(non_skin.mean()) > 0.20:
        ch_mean = arr[non_skin].mean(axis=0)
        colour_cast = float(ch_mean.max() / max(ch_mean.mean(), 1e-6) - 1.0)
    else:
        colour_cast = float("nan")

    if skin_fraction < 0.25:
        warnings.append(
            f"Only {skin_fraction * 100:.0f}% of the frame reads as skin. ITA is a "
            "median over what is left, so a background of skin-coloured wood or a "
            "warm wall will pull it."
        )
    if np.isnan(colour_cast):
        warnings.append(
            "The frame is almost all skin, so there is no neutral surface to check "
            "the white balance against. ITA is illuminant-dependent; shoot in "
            "daylight and leave some background in shot."
        )
    elif colour_cast > 0.18 and white_balance == "none":
        warnings.append(
            f"Colour cast of {colour_cast * 100:.0f}% on the background. ITA is an "
            "illuminant-dependent measurement; shoot in daylight, or try the "
            "grey-world correction."
        )
    if float(np.mean(lum > 250)) > 0.06:
        warnings.append("Blown highlights over 6% of the frame. Move out of direct flash.")
    if pt >= 5:
        warnings.append(
            "Deep skin tones: a* is a poor erythema signal there, so the barrier "
            "half of this answer is weaker than the phototype half. Cross-check "
            "with POEM."
        )

    deep = deep_phototype(arr)
    if deep is not None and int(deep["PT"]) != pt:
        warnings.append(
            f"The optional classifier says type {PT_ROMAN[int(deep['PT']) - 1]} "
            f"(p={deep['confidence']:.2f}) where ITA says {PT_ROMAN[pt - 1]}. "
            "ITA is what the readout uses."
        )

    return PhotoResult(
        PT=pt, A0=round(a0, 4), ita=ita, ita_class=ita_class,
        L=L, a_star=a_star, b_star=b_star,
        erythema_spread=ery_spread, texture=texture,
        skin_fraction=skin_fraction, n_pixels=n_px, colour_cast=colour_cast,
        white_balance=white_balance, severity=severity, deep=deep,
        warnings=warnings,
    )


# --- optional CNN hook ------------------------------------------------------
# Set DERMASENSE_PHOTOTYPE_ONNX, or drop the file at models/phototype.onnx.
# Expected: input (1, 3, 224, 224) float32, ImageNet normalisation, output six
# logits ordered type I to VI. Absent onnxruntime or absent file, the app is
# unchanged, which is the point of keeping it optional.

DEEP_MODEL_PATH = os.environ.get("DERMASENSE_PHOTOTYPE_ONNX", "models/phototype.onnx")
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_deep_session = None
_deep_tried = False


def deep_available() -> bool:
    if not os.path.exists(DEEP_MODEL_PATH):
        return False
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


def _resize_nn(arr: np.ndarray, size: int = 224) -> np.ndarray:
    """Nearest-neighbour resize, so the hook needs nothing beyond numpy."""
    h, w = arr.shape[:2]
    yi = (np.arange(size) * h / size).astype(int).clip(0, h - 1)
    xi = (np.arange(size) * w / size).astype(int).clip(0, w - 1)
    return arr[yi][:, xi]


def deep_phototype(arr: np.ndarray) -> Optional[Dict[str, float]]:
    """Returns {"PT": 1-6, "confidence": p} or None when no model is installed."""
    global _deep_session, _deep_tried
    if _deep_session is None:
        if _deep_tried or not deep_available():
            return None
        _deep_tried = True
        try:
            import onnxruntime as ort
            _deep_session = ort.InferenceSession(DEEP_MODEL_PATH,
                                                 providers=["CPUExecutionProvider"])
        except Exception:
            return None
    try:
        x = _resize_nn(np.asarray(arr, dtype=np.float32), 224) / 255.0
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)
        name = _deep_session.get_inputs()[0].name
        logits = np.asarray(_deep_session.run(None, {name: x})[0]).reshape(-1)
        e = np.exp(logits - logits.max())
        prob = e / e.sum()
        k = int(np.argmax(prob))
        return {"PT": float(k + 1), "confidence": float(prob[k])}
    except Exception:
        return None


# ===========================================================================
# 3. reconciling the two routes
# ===========================================================================

def reconcile(quiz_pt: Optional[int], photo_pt: Optional[int]) -> Tuple[Optional[int], str]:
    """When both routes ran, take the darker phototype.

    Not a compromise: f_PT lowers the UV channel as the phototype rises, so the
    darker answer is the one that recommends LESS pulcherrimin. Picking it means
    a disagreement never inflates the dose on the strength of a self-report the
    literature says is optimistic.
    """
    if quiz_pt is None:
        return photo_pt, "photo only"
    if photo_pt is None:
        return quiz_pt, "questionnaire only"
    if quiz_pt == photo_pt:
        return quiz_pt, "both routes agree"
    pt = max(quiz_pt, photo_pt)
    return pt, (f"questionnaire {PT_ROMAN[quiz_pt - 1]} vs photo "
                f"{PT_ROMAN[photo_pt - 1]}, taking the darker")


def default_answers(questions: Sequence[Question], index: int = 2) -> Dict[str, str]:
    """Middle option for every item, for a first render or a test."""
    return {q.key: q.options[min(index, len(q.options) - 1)][0] for q in questions}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="skin inputs from a quiz or a photo")
    ap.add_argument("--photo")
    ap.add_argument("--wb", choices=["none", "greyworld"], default="none")
    ap.add_argument("--crop", type=float, default=1.0)
    a = ap.parse_args()

    q = phototype_from_quiz(default_answers(FITZPATRICK_QUESTIONS))
    b = barrier_from_quiz(default_answers(POEM_QUESTIONS, index=0))
    print(f"quiz, middle answers   -> PT {int(q.value)}  ({q.label})")
    print(f"POEM, all 'no days'    -> {b.label}")

    if a.photo:
        from PIL import Image
        r = analyze_photo(Image.open(a.photo), white_balance=a.wb, crop_frac=a.crop)
        print(f"\nphoto {a.photo}")
        print(f"  ITA {r.ita:6.1f} deg  ({r.ita_class})  -> type {r.pt_roman}")
        print(f"  L* {r.L:.1f}  a* {r.a_star:.1f}  b* {r.b_star:.1f}   "
              f"skin {r.skin_fraction * 100:.0f}% of frame, {r.n_pixels} px used")
        print(f"  erythema spread {r.erythema_spread:.2f}  texture {r.texture:.3f}  "
              f"-> severity {r.severity:.2f}  A0 {r.A0:.2f}")
        if r.deep:
            print(f"  classifier: type {PT_ROMAN[int(r.deep['PT']) - 1]} "
                  f"p={r.deep['confidence']:.2f}")
        for w in r.warnings:
            print(f"  ! {w}")

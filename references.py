"""
Every source the model and the site lean on, with its link.

Kept as data rather than as page markup so a reference can be cited from more
than one page, and so the "which of these did anybody actually read" question
has an answer in the code instead of in somebody's memory.

`checked` records how far verification got, and nothing else:
    "read"      the full text was opened and the numbers below were read off it
    "matched"   the citation was located and its reported details match what the
                model claims, but the full text sits behind a publisher block
    "missing"   inherited from the v3 specification and not identifiable; the
                constant it sets is in the model with nobody able to point at
                where the number came from

No Streamlit import, so this is usable from a notebook or a test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Reference:
    key: str
    cited_as: str            # the shorthand the app and the spec use
    title: str
    where: str               # journal, volume, year
    sets: str                # what it fixes in the model
    checked: str             # read | matched | missing
    links: Tuple[Tuple[str, str], ...] = ()   # (label, url)
    note: str = ""


# ---------------------------------------------------------------------------
# the model's own constants
# ---------------------------------------------------------------------------

MODEL_PAGE: List[Reference] = [
    Reference(
        key="kregiel",
        cited_as="Kregiel 2024",
        title="Photoprotective Effects of Yeast Pulcherrimin",
        where="Molecules 29(20):4873, 2024",
        sets="Pulcherrimin UV screen, iron chelation and self-toxicity (S1b, S2, S6)",
        checked="matched",
        links=(("DOI 10.3390/molecules29204873", "https://doi.org/10.3390/molecules29204873"),
               ("PMC", "https://pmc.ncbi.nlm.nih.gov/articles/PMC11510698/")),
        note="SPF 20 by the Mansur equation and HaCaT viability by PrestoBlue are "
             "both confirmed in the abstract. The concentration the SPF was measured "
             "at is not, and the model needs it.",
    ),
    Reference(
        key="he2006",
        cited_as="He et al. 2006",
        title="Effects of environmentally realistic levels of ozone on stratum "
              "corneum function",
        where="Int J Cosmet Sci 28(5), 2006",
        sets="Depletion of the skin's own antioxidant reserve under ozone (Part 1)",
        checked="matched",
        links=(("DOI 10.1111/j.1467-2494.2006.00347.x", "https://doi.org/10.1111/j.1467-2494.2006.00347.x"),),
        note="0.8 ppm ozone over 2 h cut stratum corneum vitamin E by 70% and raised "
             "lipid hydroperoxides 2.3-fold. That 70% is where dep_max comes from.",
    ),
    Reference(
        key="milani",
        cited_as="Milani",
        title="A serum giving a 19% reduction in transepidermal water loss",
        where="Clin Cosmet Investig Dermatol, 2019",
        sets="External check on the predicted TEWL improvement (How the model works)",
        checked="matched",
        links=(("DOI 10.2147/CCID.S204905", "https://doi.org/10.2147/CCID.S204905"),),
        note="Not a validation of the shielding mechanism. Milani's serum contains "
             "no film-forming surfactant, so the two numbers are not the same number.",
    ),
    Reference(
        key="li2024",
        cited_as="Li 2024",
        title="Optimizing Systems for Robust Heterologous Production of "
              "Biosurfactants Rhamnolipid and Lyso-Ornithine Lipid in "
              "Pseudomonas putida KT2440",
        where="Molecules 29(14):3288, 2024",
        sets="Lyso-ornithine lipid CMC and its 0.010 mg/mL dose cap (S1a)",
        checked="matched",
        links=(("DOI 10.3390/molecules29143288", "https://doi.org/10.3390/molecules29143288"),
               ("PMC", "https://pmc.ncbi.nlm.nih.gov/articles/PMC11279095/")),
        note="Reports CMC 0.5 mg/L and a 10 mg/L titre, which are exactly the two "
             "numbers the model uses. The cap is what the strain can make, not an "
             "efficacy ceiling.",
    ),
    Reference(
        key="nageshwar",
        cited_as="Nageshwar",
        title="Anti-oxidative property of xylolipid produced by Lactococcus lactis "
              "LNH70 and its potential use as fruit juice preservative",
        where="Lactococcus lactis LNH70, 2022",
        sets="Xylolipid radical scavenging rate (S3)",
        checked="matched",
        links=(("PMC", "https://pmc.ncbi.nlm.nih.gov/articles/PMC9679099/"),
               ("PubMed", "https://pubmed.ncbi.nlm.nih.gov/36219343/")),
        note="The model assumes 70% DPPH scavenging at 1 mg/mL. The abstract reports "
             "a range beginning at 60.76%. If 60.76% is the figure, k_X is "
             "overstated, and k_X is one of the constants the film-side result "
             "leans on.",
    ),
    Reference(
        key="huang2020",
        cited_as="Huang 2020",
        title="Traffic-derived air pollution compromises skin barrier function and "
              "stratum corneum redox status: a population study",
        where="J Cosmet Dermatol, 2020",
        sets="Particulate effect on barrier water loss (S4)",
        checked="matched",
        links=(("DOI 10.1111/jocd.13194", "https://doi.org/10.1111/jocd.13194"),),
        note="Fitted over roughly 30-100 ug/m3. At Delhi winter levels the barrier "
             "channel is extrapolating.",
    ),
    Reference(
        key="delbino",
        cited_as="Del Bino 2013",
        title="Variations in skin colour and the biological consequences of "
              "ultraviolet radiation exposure",
        where="Br J Dermatol 169(s3), 2013",
        sets="Phototype from a photo, ITA class edges (Dose simulator)",
        checked="matched",
        links=(("DOI 10.1111/bjd.12529", "https://doi.org/10.1111/bjd.12529"),),
        note="Very light above 55 deg, light 55-41, intermediate 41-28, tan 28-10, "
             "brown 10 to -30, dark below -30. Verified against Crossref: an earlier "
             "draft of this page carried bjd.12536, which is a different paper.",
    ),
    Reference(
        key="charman",
        cited_as="Charman 2004",
        title="The Patient-Oriented Eczema Measure: Development and Initial "
              "Validation of a New Tool for Measuring Atopic Eczema Severity From "
              "the Patients' Perspective",
        where="Arch Dermatol 140(12), 2004",
        sets="Barrier questionnaire, POEM items (Dose simulator)",
        checked="matched",
        links=(("JAMA Dermatology",
                "https://jamanetwork.com/journals/jamadermatology/fullarticle/480876"),
               ("POEM at Nottingham",
                "https://www.nottingham.ac.uk/research/groups/cebd/resources/poem.aspx")),
        note="Used here as a barrier-symptom score, not as an eczema diagnosis. The "
             "mapping from POEM to A0 is ours and is class C.",
    ),
]


# ---------------------------------------------------------------------------
# the control arm
# ---------------------------------------------------------------------------

BENCHMARK_REFS: List[Reference] = [
    Reference(
        key="jstage",
        cited_as="BPB Reports 7(2)",
        title="The Lower Toxicity and Wider Safety Range of Acidic Sophorolipid "
              "Compared to Surfactin and Rhamnolipid as Biosurfactants toward the "
              "HaCaT, THP-1, and RAW 264.7",
        where="Biol Pharm Bull Reports 7(2)",
        sets="CMC and keratinocyte tolerance of sophorolipid, rhamnolipid, surfactin (Lab vs controls)",
        checked="read",
        links=(("J-STAGE",
                "https://www.jstage.jst.go.jp/article/bpbreports/7/2/7_56/_html/-char/en"),),
        note="The one source here that was opened and read in full. CMC 1,000 / 38 / "
             "16 mg/L and HaCaT LC50 20,930 / 165.2 / 80.4 mg/L respectively. One "
             "study and one method, so the three are comparable to each other.",
    ),
    Reference(
        key="fermentation",
        cited_as="Fermentation 2023",
        title="Purified Acidic Sophorolipid Biosurfactants in Skincare "
              "Applications: An Assessment of Cytotoxic Effects in Comparison with "
              "Synthetic Surfactants Using a 3D In Vitro Human Skin Model",
        where="Fermentation 9(11):985, 2023",
        sets="Sophorolipid skin safety (Lab vs controls)",
        checked="matched",
        links=(("DOI 10.3390/fermentation9110985", "https://doi.org/10.3390/fermentation9110985"),),
    ),
    Reference(
        key="glycolipid_kera",
        cited_as="Glycolipids on keratinocytes",
        title="Characterisation of cytotoxicity and immunomodulatory effects of "
              "glycolipid biosurfactants on human keratinocytes",
        where="2022",
        sets="Sophorolipid keratinocyte tolerance (Lab vs controls)",
        checked="matched",
        links=(("PMC", "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9750927/"),),
    ),
    Reference(
        key="decyl_cmc",
        cited_as="Decyl glucoside CMC",
        title="Decyl glucoside as a sustainable surfactant for cosmetic "
              "formulations: environmental stability under hardness, pH and "
              "temperature variations",
        where="RSC Advances",
        sets="Decyl glucoside CMC (Lab vs controls)",
        checked="matched",
        links=(("RSC",
                "https://pubs.rsc.org/ra/article/16/2/1600/909057/Decyl-glucoside-as-a-sustainable-surfactant-for"),),
    ),
    Reference(
        key="cir_glucoside",
        cited_as="CIR alkyl glucosides",
        title="Safety Assessment of Decyl Glucoside and Other Alkyl Glucosides as "
              "Used in Cosmetics",
        where="Cosmetic Ingredient Review Expert Panel, 2013",
        sets="Decyl glucoside tolerance (Lab vs controls)",
        checked="matched",
        links=(("CIR report (PDF)",
                "https://www.cir-safety.org/sites/default/files/decylg122011FRx.pdf"),
               ("DOI 10.1177/1091581813497764", "https://doi.org/10.1177/1091581813497764")),
    ),
    Reference(
        key="sds_hacat",
        cited_as="SDS on HaCaT",
        title="Sodium Dodecyl Sulfate Cytotoxicity towards HaCaT Keratinocytes: "
              "Comparative Analysis of Methods for Evaluation of Cell Viability",
        where="2017",
        sets="SDS toxicity, negative control (Lab vs controls)",
        checked="matched",
        links=(("PubMed", "https://pubmed.ncbi.nlm.nih.gov/28726210/"),),
        note="SDS is the OECD and ICCVAM reference cytotoxic substance for in vitro "
             "skin tests. The 0.1 mg/mL onset the model uses is a rounded value "
             "consistent with 25 ug/mL being sub-toxic over 48 h; nobody fitted it.",
    ),
]


# ---------------------------------------------------------------------------
# commercial concentrations
# ---------------------------------------------------------------------------

MARKET_REFS: List[Reference] = [
    Reference(
        key="paulas",
        cited_as="Hyaluronic acid use levels",
        title="Hyaluronic Acid Skin Care Myths, citing the Cosmetic Ingredient "
              "Review 2021 use survey",
        where="Paula's Choice",
        sets="Hyaluronic acid range in commercial products (Lab vs market)",
        checked="matched",
        links=(("paulaschoice.com",
                "https://www.paulaschoice.com/expert-advice/skincare-advice/skin-care-myths/hyaluronic-acid-skin-care-myths.html"),),
        note="A retailer's education page rather than a primary source. It cites the "
             "CIR survey, which is primary; the CIR document itself was not opened.",
    ),
    Reference(
        key="ep0820273",
        cited_as="EP0820273B1",
        title="Use of sophorolipids in cosmetic and dermatological compositions",
        where="European patent",
        sets="Biosurfactant range in cosmetics (Lab vs market)",
        checked="matched",
        links=(("Google Patents", "https://patents.google.com/patent/EP0820273B1/en"),),
    ),
    Reference(
        key="us6455055",
        cited_as="US6455055",
        title="Cosmetic compositions",
        where="US patent",
        sets="Surfactant range in leave-on products (Lab vs market)",
        checked="matched",
        links=(("USPTO",
                "https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6455055"),),
    ),
    Reference(
        key="skinceuticals",
        cited_as="C E Ferulic",
        title="SkinCeuticals C E Ferulic: 15% L-ascorbic acid, 1% alpha-tocopherol, "
              "0.5% ferulic acid",
        where="Manufacturer's stated formulation",
        sets="Vitamin C and E levels in commercial serums (Lab vs market)",
        checked="matched",
        links=(("skinceuticals.com",
                "https://www.skinceuticals.com/skincare/antioxidants-serums"),),
    ),
]


# ---------------------------------------------------------------------------
# the gaps
# ---------------------------------------------------------------------------

UNIDENTIFIED: List[Reference] = [
    Reference(key="shaheen", cited_as="Shaheen", title="",
              where="inherited from the v3 specification",
              sets="Hyaluronic acid scavenging rate, k_H (S3)",
              checked="missing",
              note="Quoted in the spec as 66% scavenging at 1 mg/mL. Searching on "
                   "that figure returns hyaluronic acid antioxidant studies, none "
                   "of them obviously the one meant."),
    Reference(key="sundaram", cited_as="Sundaram 2017", title="",
              where="inherited from the v3 specification",
              sets="Hyaluronic acid reduction in water loss, eta_H (S4)",
              checked="missing",
              note="A 15.6% TEWL reduction for high-molecular-weight hyaluronic acid "
                   "does appear in a 2016 explant study, but under different "
                   "authorship."),
    Reference(key="kim2016", cited_as="Kim 2016", title="",
              where="inherited from the v3 specification",
              sets="Compromised barrier value, A0 (Part 1)",
              checked="missing",
              note="The direction is well supported across the atopic dermatitis "
                   "literature; the specific magnitude is not traceable to a paper."),
    Reference(key="oh2021", cited_as="Oh 2021", title="",
              where="inherited from the v3 specification",
              sets="Ozone effect on barrier water loss, alpha_O3 (S4)",
              checked="missing",
              note="The spec's own margin carries a '[check]' note against this one."),
]


SECTIONS: Tuple[Tuple[str, str, List[Reference]], ...] = (
    ("Model constants", "", MODEL_PAGE),
    ("Benchmark surfactants", "", BENCHMARK_REFS),
    ("Commercial concentrations", "", MARKET_REFS),
    ("From the v3 specification", "", UNIDENTIFIED),
)

CHECK_LABEL: Dict[str, str] = {
    "read": "read in full",
    "matched": "located, full text blocked",
    "missing": "not identified",
}


def all_references() -> List[Reference]:
    out: List[Reference] = []
    for _, _, refs in SECTIONS:
        out.extend(refs)
    return out


def counts() -> Dict[str, int]:
    c: Dict[str, int] = {"read": 0, "matched": 0, "missing": 0}
    for r in all_references():
        c[r.checked] = c.get(r.checked, 0) + 1
    return c


__all__ = ["Reference", "MODEL_PAGE", "BENCHMARK_REFS", "MARKET_REFS",
           "UNIDENTIFIED", "SECTIONS", "CHECK_LABEL", "all_references", "counts"]

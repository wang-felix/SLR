import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from evaluation.extraction import extraction_utils as utils


def test_wilson_interval_has_positive_upper_bound_at_zero_errors():
    lo, hi = utils.wilson(0, 10)

    assert lo == pytest.approx(0)
    assert hi == pytest.approx(0.27754, abs=1e-5)


def test_coating_material_normalizes_formulas_synonyms_and_percentages():
    assert utils.material_components("Al₂O₃ coating with 13 wt% TiO₂") == {
        "Al2O3", "TiO2"
    }
    status, overlap, _, _ = utils.score_material("氧化铝和氧化钛涂层", "Al2O3-TiO2")
    assert (status, overlap) == ("exact", 1.0)


def test_material_scoring_ignores_substrate_and_electrolyte():
    answer = (
        "Al2O3-CePO4-MoS2 composite coating formed on 6061-T6 Al alloy "
        "in an alkaline electrolyte (4 g/L NaOH)."
    )

    status, overlap, actual, expected = utils.score_material(answer, "Al2O3-CePO4-MoS2")

    assert (status, overlap) == ("exact", 1.0)
    assert actual == expected == {"Al2O3", "CePO4", "MoS2"}


def test_coating_based_on_a_compound_is_not_substrate_context():
    answer = "Ceramic coating based on MgO and MgF2 deposited on AZ31 Mg alloy."

    assert utils.material_components(answer) == {"MgO", "MgF2"}


def test_material_scoring_drops_electrolyte_recipe_and_voltage_units():
    answer = (
        "Coating formed on Mg alloy in alkaline electrolyte (NaOH, KF) "
        "with graphene additive; coatings consist of MgO only at 200 V."
    )

    assert utils.material_components(answer) == {"MgO", "graphene"}


def test_dopant_amount_does_not_double_count_an_element_in_the_formula():
    text = "AlSiN coating with different Si doping contents (1 wt.% and 2 wt.% Si)"

    assert utils.material_components(text) == {"AlSiN"}
    assert utils.material_components(text.replace("AlSiN", "AlN")) == {"AlN", "Si"}


def test_material_scoring_keeps_formula_groups_and_decimals():
    status, overlap, actual, expected = utils.score_material(
        "Hf₀.₅Zr₀.₅O₂ and Ca(OH)2", "Hf0.5Zr0.5O2/Ca(OH)2"
    )

    assert (status, overlap) == ("exact", 1.0)
    assert actual == expected == {"Hf0.5Zr0.5O2", "Ca(OH)2"}


def test_material_scoring_detects_real_omissions():
    answer = (
        "SiC particles reinforced a coating on AZ31 Mg alloy; "
        "fabricated in a phosphate electrolyte (4 g/L Na3PO4)."
    )

    status, overlap, actual, expected = utils.score_material(answer, "MgO-Mg3(PO4)2-SiC")

    assert status == "partial"
    assert overlap == pytest.approx(1 / 3)
    assert expected - actual == {"MgO", "Mg3(PO4)2"}
    assert not actual - expected


@pytest.mark.parametrize(
    ("answer", "reference"),
    [
        ("MgO-MgAl2O4", "氧化物陶瓷"),
        ("Oxide coatings, e.g. MgO", "MgO"),
        ("Ti/TiO2 duplex coating", "TiO2"),
    ],
)
def test_material_scoring_defers_ambiguous_results(answer, reference):
    assert utils.score_material(answer, reference)[:2] == ("review", None)


@pytest.mark.parametrize(
    ("description", "category"),
    [
        ("HVOF thermal spray", "超音速火焰喷涂(HVOF/HVAF)"),
        ("微弧氧化后磁控溅射", "复合工艺"),
        ("未说明", "未说明"),
    ],
)
def test_preparation_method_category(description, category):
    assert utils.classify_method(description)[0] == category


@pytest.mark.parametrize(
    ("excerpt", "status", "verified"),
    [
        ("The coating was deposited by magnetron sputtering", "pass", 1),
        ("Samples were cured in a sealed vessel overnight", "fail", 0),
        ("薄膜样品采用磁控溅射工艺完成沉积并进行表面分析", "language_mismatch", 0),
    ],
)
def test_excerpt_status(excerpt, status, verified):
    text = utils.norm_search(
        "The coating was deposited by magnetron sputtering at room temperature."
    )

    result = utils.check_excerpt(text, excerpt)

    assert result[:3] == (status, 1, verified)
    assert bool(result[3]) == (status != "pass")


def test_excerpt_checker_distinguishes_locations_and_checks_only_the_quote(tmp_path):
    script = Path(__file__).resolve().parents[1] / "evaluation" / "extraction" / "check_excerpts.py"
    location_only = 'Section 2.1 "Preparation of the MAO layer" (Experimental materials), text excerpt'
    quote = "The oxide layer remained intact."
    location = "Section 2.1 xxx"
    table = tmp_path / "extraction.csv"
    pd.DataFrame(
        {
            "Filename": ["location.pdf", "not_found.pdf", "quote.pdf"],
            "Source Section Table or Text Excerpt": [location_only, "NOT_FOUND", f"{quote} || {location}"],
        }
    ).to_csv(table, index=False)
    (tmp_path / "quote.txt").write_text(f"After immersion, the sample was inspected. {quote}", encoding="utf-8")

    subprocess.run(
        [sys.executable, str(script), str(table), "--pdf-dir", str(tmp_path)],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )

    checked = pd.read_csv(tmp_path / "excerpt_check.csv", keep_default_na=False)
    assert checked["status"].tolist() == ["location_only", "not_found", "pass"]
    assert checked["location"].tolist() == [location_only, "", location]
    assert checked["n_segments"].tolist() == checked["n_verified"].tolist() == [0, 0, 1]


@pytest.mark.parametrize(
    ("reference", "expected_score"),
    [("氧化铝", "exact"), ("氧化物陶瓷", "review")],
)
def test_extraction_sample_and_evaluation_use_human_answers(tmp_path, reference, expected_score):
    scripts = Path(__file__).resolve().parents[1] / "evaluation" / "extraction"
    results = tmp_path / "extraction.csv"
    pd.DataFrame(
        [
            {
                "Filename": "synthetic.pdf",
                "Coating Material": "Al₂O₃",
                "Preparation Method": "HVOF spraying",
                "Source Section Table or Text Excerpt": "A synthetic excerpt for this test",
            }
        ]
    ).to_csv(results, index=False)

    subprocess.run(
        [sys.executable, "-X", "utf8", str(scripts / "make_extraction_sheet.py"),
         str(results), "--n", "1"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    labels_path = tmp_path / "extraction_label_sheet.xlsx"
    labels = pd.read_excel(labels_path, sheet_name="标注表", dtype=object)
    assert labels.loc[0, "文件名"] == "synthetic.pdf"
    assert labels["人工_涂层材料"].isna().all()
    assert labels["人工_制备方法类别"].isna().all()
    labels.loc[0, ["该论文应纳入吗", "人工_涂层材料", "人工_制备方法类别"]] = [
        "是", reference, "超音速火焰喷涂(HVOF/HVAF)"
    ]
    labels.to_excel(labels_path, sheet_name="标注表", index=False)

    subprocess.run(
        [sys.executable, "-X", "utf8", str(scripts / "evaluate_extraction.py"),
         str(labels_path), str(tmp_path / "extraction_key.csv")],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )

    material = pd.read_csv(tmp_path / "material_comparison.csv", keep_default_na=False).iloc[0]
    assert material["score"] == expected_score
    report = (tmp_path / "report_extraction.md").read_text(encoding="utf-8")
    if expected_score == "review":
        assert "自动评分 0 篇，待复核 1 篇" in report
        assert material["review_reason"] and not material["extra"]
    else:
        assert material["matched"] == "Al2O3"
        assert not material["missing"] and not material["extra"]
    assert pd.read_csv(tmp_path / "method_comparison.csv").loc[0, "correct"] == 1

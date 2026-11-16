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
        "al2o3", "tio2"
    }
    status, overlap, _, _ = utils.score_material("氧化铝和氧化钛涂层", "Al2O3-TiO2")
    assert (status, overlap) == ("exact", 1.0)


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


def test_extraction_sample_and_evaluation_use_human_answers(tmp_path):
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
        "是", "氧化铝", "超音速火焰喷涂(HVOF/HVAF)"
    ]
    labels.to_excel(labels_path, sheet_name="标注表", index=False)

    subprocess.run(
        [sys.executable, "-X", "utf8", str(scripts / "evaluate_extraction.py"),
         str(labels_path), str(tmp_path / "extraction_key.csv")],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )

    assert pd.read_csv(tmp_path / "material_comparison.csv").loc[0, "score"] == "exact"
    assert pd.read_csv(tmp_path / "method_comparison.csv").loc[0, "correct"] == 1

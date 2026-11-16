import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from evaluation.screening import evaluate


def test_weighted_screening_metrics():
    rows = pd.DataFrame(
        {
            "ai": [1, 1, 0, 0],
            "human": [1, 0, 1, 0],
            "weight": [2.0, 1.0, 3.0, 4.0],
        }
    )

    result = evaluate.metrics(rows)

    assert {k: result[k] for k in ("TP", "FP", "FN", "TN")} == {
        "TP": 2.0, "FP": 1.0, "FN": 3.0, "TN": 4.0
    }
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 5)
    assert result["kappa"] == pytest.approx(0.2)


def test_screening_sample_sheet_hides_ai_decisions(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "evaluation" / "screening" / "make_label_sheet.py"
    )
    results = tmp_path / "screening.xlsx"
    pd.DataFrame(
        [
            ("a.pdf", "Likely Exclude", 100, "Synthetic exclusion", ""),
            ("b.pdf", "Likely Include", 200, "Synthetic cermet sample", ""),
            ("c.pdf", "Likely Include", 300, "Synthetic inclusion", ""),
        ],
        columns=["Filename", "Decision", "Text Length", "Reasoning", "Notes"],
    ).to_excel(results, index=False)

    subprocess.run([sys.executable, "-X", "utf8", str(script), str(results)],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    labels = pd.read_excel(tmp_path / "label_sheet.xlsx")
    key = pd.read_csv(tmp_path / "sample_key.csv")

    assert set(labels["文件名"]) == set(key["filename"]) == {"a.pdf", "b.pdf", "c.pdf"}
    assert labels["最终人工判断"].isna().all()
    assert "ai_decision" not in labels.columns
    assert set(key["stratum"]) == {"A_exclude", "B_include_borderline", "C_include_clear"}
    assert dict(zip(key["filename"], key["text_length"])) == {
        "a.pdf": 100, "b.pdf": 200, "c.pdf": 300
    }

    used = tmp_path / "used.csv"
    key.loc[key["filename"] == "a.pdf"].to_csv(used, index=False)
    subprocess.run(
        [sys.executable, "-X", "utf8", str(script), str(results),
         "--exclude-key", str(used), "--prefix", "val_"],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    validation = pd.read_csv(tmp_path / "val_sample_key.csv")
    assert set(validation["filename"]) == {"b.pdf", "c.pdf"}
    assert (tmp_path / "val_label_sheet.xlsx").is_file()


@pytest.fixture
def screening_comparison(tmp_path):
    labels = tmp_path / "labels.xlsx"
    key = tmp_path / "sample_key.csv"
    results = tmp_path / "new_results.csv"
    pd.DataFrame(
        {
            "序号": [1, 2, 3],
            "最终人工判断": ["纳入", "排除", "排除"],
            "备注": ["", "", ""],
        }
    ).to_excel(labels, sheet_name="标注表", index=False)
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "filename": ["old-a.pdf", "old-b.pdf", "old-c.pdf"],
            "ai_decision": ["Likely Include", "Likely Include", "Likely Exclude"],
            "text_length": [100, 200, 300],
            "stratum": ["B", "B", "A"],
            "N_stratum": [8, 8, 2],
            "weight": [4, 4, 2],
        }
    ).to_csv(key, index=False)
    pd.DataFrame(
        [
            ("new-c.pdf", 300, "Likely Exclude", "c"),
            ("new-b.pdf", 200, "Error", "b"),
            ("new-a.pdf", 100, "Likely Include", "a"),
        ],
        columns=["Filename", "Text Length", "Decision", "Reasoning"],
    ).to_csv(results, index=False)
    return labels, key, results


def test_matches_renamed_screening_results_by_text_length(screening_comparison):
    labels, key, results = screening_comparison

    aligned = evaluate.load(labels, key, str(results))

    assert aligned["ai_decision"].tolist() == ["Likely Include", "Error", "Likely Exclude"]
    assert aligned["ai_reasoning_new"].tolist() == ["a", "b", "c"]
    assert aligned["ai_baseline"].tolist() == ["Likely Include", "Likely Include", "Likely Exclude"]


def test_rejects_ambiguous_text_lengths(screening_comparison):
    labels, key, results = screening_comparison
    new_results = pd.read_csv(results)
    duplicate_length = new_results.iloc[[2]].assign(Filename="unrelated.pdf")
    pd.concat([new_results, duplicate_length]).to_csv(results, index=False)

    with pytest.raises(SystemExit, match="text lengths missing/ambiguous"):
        evaluate.load(labels, key, str(results))


@pytest.mark.parametrize(
    ("mode", "scored", "matrix_row", "disagreement_ids"),
    [
        ("drop", 2, "| AI:纳入 | 8.0 | 0.0 |", []),
        ("flag", 3, "| AI:纳入 | 4.0 | 4.0 |", [2]),
    ],
)
def test_screening_error_policy(
    mode, scored, matrix_row, disagreement_ids, screening_comparison, tmp_path
):
    labels, key, results = screening_comparison
    report = tmp_path / "report.md"
    subprocess.run(
        [sys.executable, "-X", "utf8", str(Path(evaluate.__file__).resolve()),
         str(labels), str(key), "--results", str(results), "--errors", mode,
         "--out", str(report)],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )

    content = report.read_text(encoding="utf-8")
    assert f"参与指标 {scored}" in content
    assert f"AI 处理失败(Decision=Error):1 篇;处理方式 --errors {mode}" in content
    assert matrix_row in content
    assert "| AI:排除 | 0.0 | 2.0 |" in content
    assert pd.read_csv(tmp_path / "disagreements.csv")["id"].tolist() == disagreement_ids

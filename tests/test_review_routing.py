import csv
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openpyxl import load_workbook

import workspace_store
from housing_enhanced import SystematicReviewAutomation, TEXT_INPUT_COLUMNS
from WebApp.services import processing_service


@pytest.fixture
def make_automation(monkeypatch, tmp_path):
    def make(threshold=None, scope="all"):
        auto = SystematicReviewAutomation(
            api_key="fixture-only",
            pdf_folder=tmp_path / "pdfs",
            output_folder=tmp_path / "output",
            advanced_config={
                "flag_confidence_threshold": threshold,
                "flag_confidence_scope": scope,
            },
        )
        monkeypatch.setattr(auto, "_llm_call", Mock())
        return auto

    monkeypatch.setattr(SystematicReviewAutomation, "_init_llm", lambda self: SimpleNamespace())
    return make


def audit_rows(auto):
    return [json.loads(line) for line in auto.audit_ledger.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize(
    "threshold, scope, stage, decision, confidence, normalized, routed",
    [
        ("low", "all", "Full-text", "Likely Include", " lOw ", "Low", True),
        ("low", "all", "Full-text", "Likely Exclude", "Medium", "Medium", False),
        ("medium", "exclude_only", "Full-text", "Likely Exclude", "MEDIUM", "Medium", True),
        ("medium", "exclude_only", "Full-text", "Likely Include", "Low", "Low", False),
        ("medium", "all", "Full-text", "Likely Exclude", "High", "High", False),
        ("medium", "all", "Full-text", "Likely Exclude", ["Low"], None, False),
        ("medium", "all", "Full-text", "Likely Exclude", "unsure", None, False),
        ("medium", "all", "Full-text", "Likely Exclude", None, None, False),
        ("medium", "all", "Title/Abstract", "Likely Exclude", "Low", "Low", False),
        ("medium", "all", "Full-text", "Flag for Human Review", "Low", "Low", False),
        ("medium", "all", "Full-text", "Flag for Review", "Low", "Low", False),
        (None, "all", "Full-text", "Likely Exclude", "Low", "Low", False),
    ],
)
def test_confidence_routing(make_automation, threshold, scope, stage, decision, confidence, normalized, routed):
    auto = make_automation(threshold, scope)
    auto._llm_call.return_value = (
        json.dumps({"decision": decision, "notes": "uncertain", "confidence": confidence}),
        3,
    )

    result = auto.screen_article("paper text", "paper.pdf", stage=stage)

    assert result.confidence == normalized
    assert result.decision == ("Flag for Human Review" if routed else decision)
    assert result.review_trigger == ("confidence_threshold" if routed else None)
    assert result.raw_decision == (decision if threshold and stage == "Full-text" else None)
    assert result.notes == "uncertain"
    auto._llm_call.assert_called_once()


@pytest.mark.parametrize("threshold, scope", [("high", "all"), (None, "include_only")])
def test_invalid_routing_settings(make_automation, threshold, scope):
    with pytest.raises(ValueError, match="flag_confidence_"):
        make_automation(threshold, scope)


def test_cached_decision_is_routed_with_current_settings(make_automation):
    original = make_automation("medium")
    original._llm_call.side_effect = [
        ('{"decision":"Likely Exclude","confidence":"Medium"}', 3),
        ('{"decision":"Likely Exclude"}', 5),
    ]
    assert original.screen_article("paper text", "original.pdf").decision == "Flag for Human Review"
    original.screen_article("old cache text", "old.pdf")

    cache_key = audit_rows(original)[0]["cache_key"]
    cache = json.loads((original.cache_folder / f"screening_{cache_key}.json").read_text(encoding="utf-8"))
    assert cache["decision"] == "Likely Exclude"
    assert "raw_decision" not in cache and "review_trigger" not in cache

    stricter = make_automation("low")
    result = stricter.screen_article("paper text", "cached.pdf")
    event, = audit_rows(stricter)
    assert (result.decision, result.confidence, result.api_tokens_used) == ("Likely Exclude", "Medium", 0)
    assert event["cache_hit"] is True
    assert event["cache_key"] == cache_key
    assert event["raw_decision"] == "Likely Exclude"
    assert event["confidence"] == "Medium"
    assert event["review_trigger"] is None
    stricter._llm_call.assert_not_called()

    current = make_automation("medium")
    assert current.screen_article("paper text", "cached.pdf").review_trigger == "confidence_threshold"
    old = current.screen_article("old cache text", "old.pdf")
    assert old.decision == "Likely Exclude" and old.confidence is None
    current._llm_call.assert_not_called()


def test_routed_paper_skips_extraction_and_reaches_workspace(make_automation, monkeypatch, tmp_path):
    root = tmp_path / "workspace"
    handle = workspace_store.create_workspace(root)
    pdf = root / "pdfs" / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    workspace_store.register_pdf(root, "pdfs/paper.pdf", original_filename=pdf.name)
    auto = make_automation("low")
    auto.pdf_folder = pdf.parent
    auto._llm_call.return_value = ('{"decision":"Likely Include","confidence":"Low"}', 3)
    monkeypatch.setattr(auto, "extract_text_from_pdf", lambda path: ("paper text", True))
    monkeypatch.setattr(auto, "extract_data", Mock())

    summary = auto.process_pdfs()

    auto.extract_data.assert_not_called()
    assert summary["prisma"]["flagged"] == 1
    assert summary["extracted_count"] == 0
    assert summary["review_routing"]["confidence_threshold_count"] == 1
    assert "HUMAN REVIEW ROUTING" in auto.summary_report.read_text(encoding="utf-8")
    config, event = audit_rows(auto)
    assert config["review_routing"] == {"flag_confidence_threshold": "low", "flag_confidence_scope": "all"}
    assert (event["decision"], event["raw_decision"], event["review_trigger"]) == (
        "Flag for Human Review", "Likely Include", "confidence_threshold",
    )
    with auto.screening_csv.open(encoding="utf-8", newline="") as handle_csv:
        csv_reader = csv.DictReader(handle_csv)
        csv_result, = list(csv_reader)
        headers = csv_reader.fieldnames
    assert headers[-8:] == ["confidence", "raw_decision", "review_trigger", *TEXT_INPUT_COLUMNS]
    assert csv_result["confidence"] == "Low"
    workbook = load_workbook(auto.screening_excel, read_only=True)
    try:
        excel_headers, excel_result = workbook.active.values
        assert [header.lower().replace(" ", "_") for header in excel_headers] == headers
        assert excel_result[-8:-5] == ("Low", "Likely Include", "confidence_threshold")
    finally:
        workbook.close()

    processing_service.persist_workspace_processing_suggestions(handle, auto, display_names={})
    item, = workspace_store.get_review_queue(root, stage="full_text")
    assert item["latest_ai_suggestion"]["decision"] == "flag"
    assert item["latest_ai_suggestion"]["confidence"] == "Low"

import csv
import json
import sys
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openpyxl import load_workbook

import workspace_store
from evaluation.screening import evaluate, make_label_sheet
from housing_enhanced import SystematicReviewAutomation, TEXT_INPUT_COLUMNS
from WebApp.services import processing_service


@pytest.fixture
def make_automation(monkeypatch, tmp_path):
    def make(threshold=None, scope="all", *, passes=1, consensus_scope="all"):
        auto = SystematicReviewAutomation(
            api_key="fixture-only",
            pdf_folder=tmp_path / "pdfs",
            output_folder=tmp_path / "output",
            advanced_config={
                "flag_confidence_threshold": threshold,
                "flag_confidence_scope": scope,
                "screening_consensus_passes": passes,
                "consensus_scope": consensus_scope,
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


def reply(decision, confidence=None):
    return json.dumps({"decision": decision, "confidence": confidence}), 3


@pytest.mark.parametrize("passes", [True, 0, 6, 2.0, "2", None])
def test_invalid_consensus_pass_count(make_automation, passes):
    with pytest.raises(ValueError, match="screening_consensus_passes"):
        make_automation(passes=passes)


def test_invalid_consensus_scope(make_automation):
    with pytest.raises(ValueError, match="consensus_scope"):
        make_automation(consensus_scope="include_only")


def test_p012_disagreement_is_routed_without_extraction(make_automation, monkeypatch):
    auto = make_automation(passes=2)
    auto.pdf_folder.mkdir()
    (auto.pdf_folder / "P012.pdf").touch()
    monkeypatch.setattr(auto, "extract_text_from_pdf", lambda path: ("P012", True))
    monkeypatch.setattr(auto, "extract_data", Mock())
    auto._llm_call.side_effect = [reply("Likely Include", "High"), reply("Likely Exclude", "High")]

    summary = auto.process_pdfs()

    auto.extract_data.assert_not_called()
    result, = auto.screening_results
    assert result.decision == "Flag for Human Review"
    assert result.consensus_decisions == "Likely Include | Likely Exclude"
    assert (result.review_trigger, result.api_tokens_used) == ("pass_disagreement", 6)
    config, first, second = audit_rows(auto)
    assert config["review_routing"]["consensus_passes"] == 2
    assert [(row["consensus_pass"], row["decision"]) for row in (first, second)] == [
        (1, "Likely Include"), (2, "Likely Exclude"),
    ]
    assert summary["review_routing"]["extra_passes_run"] == 1


def test_consensus_cache_isolated_and_reused_pass_by_pass(make_automation):
    single = make_automation()
    single._llm_call.return_value = reply("Likely Exclude")
    single.screen_article("paper", "before.pdf")
    original, = audit_rows(single)
    assert "consensus_pass" not in original

    consensus = make_automation(passes=2)
    consensus._llm_call.return_value = reply("Likely Include")
    for filename in ("first.pdf", "second.pdf"):
        assert consensus.screen_article("paper", filename).review_trigger == "pass_disagreement"
    first, second, repeat_first, repeat_second = audit_rows(consensus)
    assert first["cache_key"] == repeat_first["cache_key"] == original["cache_key"]
    assert second["cache_key"] == repeat_second["cache_key"] != first["cache_key"]
    assert [row["cache_hit"] for row in (first, second, repeat_first, repeat_second)] == [True, False, True, True]
    consensus._llm_call.assert_called_once()


def test_exclude_only_uses_raw_first_decision_and_skips_title_abstract(make_automation):
    auto = make_automation("low", passes=2, consensus_scope="exclude_only")
    auto._llm_call.side_effect = [
        reply("Likely Include", "Low"),
        reply("Likely Exclude", "Low"), reply("Likely Include"),
        reply("Likely Exclude", "Low"),
    ]
    include = auto.screen_article("include", "include.pdf")
    exclude = auto.screen_article("exclude", "exclude.pdf")
    abstract = auto.screen_article("abstract", "abstract.pdf", stage="Title/Abstract")

    assert include.review_trigger == "confidence_threshold"
    assert include.consensus_decisions == "Likely Include"
    assert exclude.review_trigger == "pass_disagreement"
    assert abstract.decision == "Likely Exclude"
    assert "consensus_decisions" not in asdict(abstract)
    assert auto._llm_call.call_count == 4


@pytest.mark.parametrize(
    "responses, expected_decision, expected_trigger, expected_confidence",
    [
        (["Error", "Likely Exclude"], "Flag for Human Review", "partial_error", "Medium"),
        (["Flag for Review", "Error"], "Flag for Human Review", "partial_error", "High"),
        (["Error", "Error"], "Error", None, None),
        (["Flag for Review", "Likely Exclude"], "Flag for Human Review", "model_flag", "Medium"),
        (["Likely Exclude", "Likely Exclude"], "Flag for Human Review", "confidence_threshold", "Medium"),
        (["Likely Include", "Likely Include"], "Flag for Human Review", "confidence_threshold", "Medium"),
    ],
)
def test_consensus_decision_priority(make_automation, responses, expected_decision, expected_trigger, expected_confidence):
    auto = make_automation("medium", passes=2)
    auto._llm_call.side_effect = [
        reply(decision, confidence) if decision != "Error" else RuntimeError("simulated request failure")
        for decision, confidence in zip(responses, ("High", "Medium"))
    ]
    result = auto.screen_article("paper", "paper.pdf")
    first, second = audit_rows(auto)

    assert (result.decision, result.review_trigger, result.confidence) == (
        expected_decision, expected_trigger, expected_confidence,
    )
    assert result.consensus_decisions == " | ".join(responses)
    assert [row["status"] for row in (first, second)] == [
        "error" if decision == "Error" else "ok" for decision in responses
    ]
    assert result.api_tokens_used == sum(row["api_tokens_used"] for row in (first, second))


@pytest.mark.parametrize("stop_after", [1, 2])
def test_consensus_observes_stop_even_when_next_pass_is_cached(make_automation, monkeypatch, stop_after):
    warm = make_automation(passes=3)
    warm._llm_call.return_value = reply("Likely Exclude")
    warm.screen_article("paper", "cached.pdf")
    auto = make_automation(passes=3)
    append = auto._append_audit_event

    def stop_after_pass(event):
        append(event)
        if event["consensus_pass"] == stop_after:
            auto.stop_event.set()

    monkeypatch.setattr(auto, "_append_audit_event", stop_after_pass)
    with pytest.raises(InterruptedError):
        auto.screen_article("paper", "stopped.pdf")
    assert [row["consensus_pass"] for row in audit_rows(auto)] == list(range(1, stop_after + 1))
    assert auto._generate_summary()["review_routing"]["extra_passes_run"] == stop_after - 1
    auto._llm_call.assert_not_called()


def test_routing_summary_and_evaluation_accept_consensus_export(make_automation, monkeypatch, tmp_path):
    auto = make_automation("low", passes=2)
    assert auto._generate_summary()["review_routing"]["flag_rate"] == 0.0
    auto._llm_call.side_effect = [
        reply("Likely Exclude", "Medium"), reply("Likely Exclude", "Low"),
        reply("Flag for Review", "High"), reply("Likely Include", "High"),
        reply("Likely Exclude"), reply("Likely Exclude"),
    ]
    auto.screening_results = [
        auto.screen_article(text, filename)
        for text, filename in (("first", "a.pdf"), ("second", "b.pdf"), ("third text", "c.pdf"))
    ]
    summary = auto._generate_summary()["review_routing"]
    assert summary["screened_count"] == 3
    assert summary["flagged_count"] == 2
    assert summary["flag_rate"] == 0.6667
    assert summary["triggers"] == {
        "model_flag": 1, "confidence_threshold": 1, "pass_disagreement": 0, "partial_error": 0,
    }
    assert summary["confidence_counts"] == {"High": 1, "Medium": 0, "Low": 1, "unspecified": 1}
    assert summary["extra_passes_run"] == 3
    assert "0.6667" in auto.summary_report.read_text(encoding="utf-8")

    auto.write_screening_csv()
    auto.write_screening_excel()
    with auto.screening_csv.open(encoding="utf-8", newline="") as handle:
        headers = next(csv.reader(handle))
    assert headers[-9:] == ["confidence", "raw_decision", "review_trigger", "consensus_decisions", *TEXT_INPUT_COLUMNS]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_label_sheet.py", str(auto.screening_excel)])
    make_label_sheet.main()
    labels = tmp_path / "label_sheet.xlsx"
    key = tmp_path / "sample_key.csv"
    compared = evaluate.load(labels, key, str(auto.screening_excel)).set_index("filename")
    assert compared["ai_decision"].to_dict() == {
        "a.pdf": "Flag for Human Review", "b.pdf": "Flag for Human Review", "c.pdf": "Likely Exclude",
    }
    assert compared["ai"].to_dict() == {"a.pdf": 1, "b.pdf": 1, "c.pdf": 0}

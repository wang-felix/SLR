import csv
import hashlib
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openpyxl import load_workbook

from housing_enhanced import SystematicReviewAutomation


@pytest.fixture
def batch(monkeypatch, tmp_path):
    client = Mock(profile={}, get_last_call_metadata=Mock(return_value={}))
    client.chat_completion_with_tokens.return_value = (
        '{"decision":"Likely Include","reasoning":"fixture","notes":""}', 2,
    )
    client.chat_completion_structured.return_value = (
        SimpleNamespace(model_dump=lambda: {"material": "ceramic"}), 3,
    )
    monkeypatch.setattr(SystematicReviewAutomation, "_init_llm", lambda self: client)
    processors = []

    def make(documents, *, cache_enabled=False, parallel_processing=False, **kwargs):
        auto = SystematicReviewAutomation(
            api_key="",
            llm_model="offline-fixture",
            pdf_folder=str(tmp_path / "pdfs"),
            output_folder=str(tmp_path / "output"),
            cache_enabled=cache_enabled,
            parallel_processing=parallel_processing,
            screening_prompt="Screen: {text}",
            extraction_fields=["material"],
            **kwargs,
        )
        processors.append(auto)
        for filename in documents:
            path = auto.pdf_folder / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        monkeypatch.setattr(auto, "_extract_pdf_text", lambda path: documents[path.name])
        return auto, client

    yield make
    for auto in processors:
        auto.close()


def _group(text, *filenames):
    return {
        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "count": len(filenames),
        "filenames": list(filenames),
    }


@pytest.mark.parametrize("cache_enabled, calls_per_stage", [(True, 2), (False, 4)])
def test_identical_text_is_reported_without_changing_model_calls_or_rows(
    batch, cache_enabled, calls_per_stage,
):
    text = "Identical extracted text.\r\n  With spacing."
    names = ["a.pdf", "b.pdf", "c.pdf", "other.pdf"]
    auto, client = batch(
        {**dict.fromkeys(names[:3], (text, True)), "other.pdf": ("Unrelated text.", True)},
        cache_enabled=cache_enabled,
    )

    summary = auto.process_pdfs()

    group = _group(text, *names[:3])
    assert summary["full_text_duplicates"] == {
        "files_with_text": 4,
        "unique_text_count": 2,
        "duplicate_group_count": 1,
        "duplicate_extra_file_count": 2,
        "groups": [group],
    }
    assert group["text_hash"] == auto.get_paper_text_metadata("a.pdf")["text_hash"]
    assert client.chat_completion_with_tokens.call_count == calls_per_stage
    assert client.chat_completion_structured.call_count == calls_per_stage
    assert summary["screened_count"] == summary["extracted_count"] == 4
    assert summary["prisma"] == {
        "identified": 4, "screened": 4, "included": 4, "excluded": 0,
        "flagged": 0, "errors": 0, "extracted": 4,
    }
    audit = [json.loads(line) for line in auto.audit_ledger.read_text(encoding="utf-8").splitlines()]
    for stage in ("screening", "extraction"):
        events = [row for row in audit if row["kind"] == stage]
        assert [row["filename"] for row in events] == names
        assert [row["cache_hit"] for row in events] == [False, cache_enabled, cache_enabled, False]

    reports = (
        (auto.screening_csv, auto.screening_excel, [
            "filename", "decision", "stage", "reasoning", "notes", "processing_time",
            "text_length", "api_tokens_used",
        ]),
        (auto.extraction_csv, auto.extraction_excel, [
            "filename", "material", "processing_time", "api_tokens_used",
        ]),
    )
    metadata_headers = [
        "input_text_chars", "prepared_text_chars", "text_truncated", "text_limit_chars", "text_input_state",
    ]
    for csv_path, xlsx_path, headers in reports:
        headers = headers + metadata_headers
        with csv_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == headers
            assert [row["filename"] for row in reader] == names
        workbook = load_workbook(xlsx_path, read_only=True)
        try:
            columns, *rows = workbook.active.values
            assert list(columns) == [header.replace("_", " ").title() for header in headers]
            assert [row[0] for row in rows] == names
        finally:
            workbook.close()

    report = auto.summary_report.read_text(encoding="utf-8")
    assert (
        "FULL-TEXT DUPLICATES\n"
        "  Files with text       : 4\n"
        "  Unique texts          : 2\n"
        "  Duplicate groups      : 1\n"
        "  Extra duplicate files : 2\n"
        f"  {group['text_hash'][:12]} (3 files)\n"
        "    a.pdf\n    b.pdf\n    c.pdf\n\n"
        "TEXT TRUNCATION\n"
    ) in report


@pytest.mark.parametrize("left, right", [
    ("ABCD", "WXYZ"),
    ("Shared prefix followed by apples", "Shared prefix followed by pears"),
    ("Shared\r\ntext  with spacing", "Shared\ntext with spacing"),
], ids=["same-length", "same-truncated-prefix", "same-normalized-text"])
def test_only_exact_untruncated_text_matches(batch, left, right):
    auto, client = batch(
        {"a.pdf": (left, True), "b.pdf": (right, True)},
        advanced_config={"max_text_chars": 6, "enable_smart_truncation": False},
    )

    assert auto.process_pdfs()["full_text_duplicates"] == {
        "files_with_text": 2, "unique_text_count": 2, "duplicate_group_count": 0,
        "duplicate_extra_file_count": 0, "groups": [],
    }
    prompts = [call.args[0][-1]["content"] for call in client.chat_completion_with_tokens.call_args_list]
    assert prompts == [f"Screen: {left[:6]}", f"Screen: {right[:6]}"]


def test_parallel_groups_follow_discovery_order_not_worker_completion(batch, monkeypatch):
    first, second = "First group text", "Second group text"
    auto, _ = batch(
        {"a.pdf": (first, True), "b.pdf": (second, True),
         "c.pdf": (first, True), "d.pdf": (second, True)},
        parallel_processing=True,
        max_workers=4,
        advanced_config={"dry_run": True},
    )
    names = [path.name for path in auto._discover_pdf_files()]
    gates = dict(zip(names, (threading.Event() for _ in names)))
    predecessor = dict(zip(names[1:], names))
    completed = []
    process_one = auto._process_one

    def reverse_workers(path):
        assert gates[path.name].wait(timeout=5)
        result = process_one(path)
        completed.append(path.name)
        if path.name in predecessor:
            gates[predecessor[path.name]].set()
        return result

    gates[names[-1]].set()
    monkeypatch.setattr(auto, "_process_one", reverse_workers)

    summary = auto.process_pdfs()

    assert completed == list(reversed(names))
    assert summary["full_text_duplicates"]["groups"] == [
        _group(first, "a.pdf", "c.pdf"), _group(second, "b.pdf", "d.pdf"),
    ]


def test_failed_or_empty_dry_run_conversions_do_not_count(batch):
    auto, client = batch(
        {"fail-a.pdf": ("Conversion failed", False), "fail-b.pdf": ("Conversion failed", False),
         "empty-a.pdf": ("", True), "empty-b.pdf": ("", True), "blank.pdf": (" \n", True)},
        advanced_config={"dry_run": True},
    )

    summary = auto.process_pdfs()

    assert summary["screened_count"] == 5
    assert summary["prisma"]["errors"] == 2
    assert summary["full_text_duplicates"] == {
        "files_with_text": 0, "unique_text_count": 0, "duplicate_group_count": 0,
        "duplicate_extra_file_count": 0, "groups": [],
    }
    client.chat_completion_with_tokens.assert_not_called()


def test_cancelled_batch_counts_converted_text_even_without_a_screening_result(batch, monkeypatch):
    text = "Identical before cancellation"
    auto, client = batch(dict.fromkeys(("a.pdf", "b.pdf", "c.pdf"), (text, True)))
    converted = []

    def convert(path):
        converted.append(path.name)
        if path.name == "b.pdf":
            auto.stop_event.set()
        return text, True

    monkeypatch.setattr(auto, "_extract_pdf_text", convert)

    summary = auto.process_pdfs()

    assert converted == ["a.pdf", "b.pdf"]
    assert summary["screened_count"] == 1
    client.chat_completion_with_tokens.assert_called_once()
    assert summary["full_text_duplicates"] == {
        "files_with_text": 2, "unique_text_count": 1, "duplicate_group_count": 1,
        "duplicate_extra_file_count": 1, "groups": [_group(text, "a.pdf", "b.pdf")],
    }


def test_reusing_a_processor_clears_previous_batch_hashes(batch, tmp_path, monkeypatch):
    text = "Text reused across batches"
    auto, _ = batch(dict.fromkeys(("a.pdf", "b.pdf"), (text, True)), cache_enabled=True)
    first = auto.process_pdfs()
    next_folder = tmp_path / "second-batch"
    next_folder.mkdir()
    (next_folder / "c.pdf").touch()
    auto.pdf_folder = next_folder
    monkeypatch.setattr(auto, "_extract_pdf_text", lambda path: (text, True))

    second = auto.process_pdfs()

    assert first["full_text_duplicates"]["groups"] == [_group(text, "a.pdf", "b.pdf")]
    assert second["full_text_duplicates"] == {
        "files_with_text": 1, "unique_text_count": 1, "duplicate_group_count": 0,
        "duplicate_extra_file_count": 0, "groups": [],
    }

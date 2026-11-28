import csv
import hashlib
import json
import sys
from unittest.mock import Mock

import pytest
from openpyxl import load_workbook

from evaluation.extraction.extraction_utils import pick_col, read_table
from evaluation.screening import evaluate, make_label_sheet
from housing_enhanced import SystematicReviewAutomation


METADATA_COLUMNS = [
    "input_text_chars", "prepared_text_chars", "text_truncated", "text_limit_chars", "text_input_state",
]


@pytest.fixture
def make_batch(monkeypatch, tmp_path):
    client = Mock(profile={})
    client.get_last_call_metadata.return_value = {}
    client.chat_completion_structured.return_value = (None, 0)
    monkeypatch.setattr(SystematicReviewAutomation, "_init_llm", lambda self: client)
    processors = []

    def make(documents, *, values=None, decision="Likely Include", **options):
        values = {"material": "ceramic"} if values is None else values
        pdfs = tmp_path / "pdfs"
        pdfs.mkdir()
        for name in documents:
            (pdfs / name).touch()

        def reply(messages, **kwargs):
            if messages[0]["content"].startswith("You extract"):
                return json.dumps(values), 3
            return json.dumps({"decision": decision, "reasoning": "fixture", "notes": "fixture note"}), 2

        client.chat_completion_with_tokens.side_effect = reply
        advanced = {"max_text_chars": 8, "two_stage_chars": 4, "enable_smart_truncation": False}
        advanced.update(options.pop("advanced_config", {}))
        auto = SystematicReviewAutomation(
            api_key="", llm_model="offline-fixture", pdf_folder=str(pdfs),
            output_folder=str(tmp_path / "out"),
            cache_enabled=options.pop("cache_enabled", False),
            parallel_processing=options.pop("parallel_processing", False),
            screening_prompt="Screen {text}", extraction_prompt="Extract {text}",
            extraction_fields=list(values), advanced_config=advanced, **options,
        )
        processors.append(auto)
        monkeypatch.setattr(auto, "_extract_pdf_text", lambda path: documents[path.name])
        return auto, client

    yield make
    for auto in processors:
        auto.close()


def _audit(auto):
    return [json.loads(line) for line in auto.audit_ledger.read_text(encoding="utf-8").splitlines()]


def _counts(prepared=0, truncated=0, cache_hit=0):
    return {"prepared_count": prepared, "truncated_count": truncated, "cache_hit_count": cache_hit}


def _xlsx_rows(path):
    workbook = load_workbook(path, read_only=True)
    try:
        return list(workbook.active.values)
    finally:
        workbook.close()


def _csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def _assert_tail(csv_path, excel_path, columns, expected):
    csv_headers, *csv_rows = _csv_rows(csv_path)
    excel_headers, *excel_rows = _xlsx_rows(excel_path)
    width = len(columns)
    assert csv_headers[-width:] == columns
    assert list(excel_headers[-width:]) == [name.replace("_", " ").title() for name in columns]
    assert [list(row[-width:]) for row in excel_rows] == expected
    assert [row[-width:] for row in csv_rows] == [
        ["" if value is None else str(value) for value in row] for row in expected
    ]


def test_screening_counts_short_exact_and_long_inputs_in_parallel(make_batch):
    texts = {"exact.pdf": "12345678", "long.pdf": "long fixture text", "short.pdf": "abc"}
    auto, _ = make_batch(
        {name: (text, True) for name, text in texts.items()},
        decision="Likely Exclude", parallel_processing=True,
    )

    summary = auto.process_pdfs()

    expected = []
    for result in auto.screening_results:
        length = len(texts[result.filename])
        assert result.text_length == length
        expected.append([length, min(length, 8), length > 8, 8, "prepared"])
    _assert_tail(auto.screening_csv, auto.screening_excel, METADATA_COLUMNS, expected)
    stage_events = [event for event in _audit(auto) if event["kind"] == "screening"]
    assert sorted(event["filename"] for event in stage_events) == sorted(texts)
    for event in stage_events:
        length = len(texts[event["filename"]])
        assert event["text_input"] == {
            "input_text_chars": length, "prepared_text_chars": min(length, 8),
            "text_truncated": length > 8, "text_limit_chars": 8, "state": "prepared",
        }
    assert summary["text_truncation"] == {
        "limit_chars": {"title_abstract": 4, "full_text": 8, "extraction": 8},
        "stages": {
            "title_abstract": _counts(), "full_text": _counts(3, 1), "extraction": _counts(),
        },
        "extraction_metadata_columns": dict(zip(METADATA_COLUMNS, METADATA_COLUMNS)),
    }
    report = auto.summary_report.read_text(encoding="utf-8")
    assert report.index("FULL-TEXT DUPLICATES") < report.index("TEXT TRUNCATION") < report.index("OUTPUT FILES")
    section = report.split("TEXT TRUNCATION\n", 1)[1].split("OUTPUT FILES", 1)[0]
    assert "Title/Abstract: prepared=0, truncated=0, cache_hit=0, limit_chars=4" in section
    assert "Full-text: prepared=3, truncated=1, cache_hit=0, limit_chars=8" in section
    assert "Extraction: prepared=0, truncated=0, cache_hit=0, limit_chars=8" in section
    assert "long fixture text" not in section
    assert str(auto.pdf_folder) not in section


def test_two_stage_counts_cover_early_exclusion_and_final_screening_row(make_batch):
    auto, client = make_batch(
        {"exclude.pdf": ("DROP this paper", True), "keep.pdf": ("KEEP this paper", True)},
        two_stage_screening=True,
    )
    client.chat_completion_with_tokens.side_effect = [
        ('{"decision":"Likely Exclude"}', 2),
        ('{"decision":"Flag for Review"}', 2),
        ('{"decision":"Likely Exclude"}', 2),
    ]

    summary = auto.process_pdfs()

    first, second = auto.screening_results
    assert (first.stage, first.text_length, first.prepared_text_chars) == ("Title/Abstract", 15, 4)
    assert (second.stage, second.text_length, second.prepared_text_chars) == ("Full-text", 15, 8)
    assert summary["text_truncation"]["stages"] == {
        "title_abstract": _counts(2, 2), "full_text": _counts(1, 1), "extraction": _counts(),
    }
    calls = client.chat_completion_with_tokens.call_args_list
    assert [call.args[0][-1]["content"] for call in calls] == ["Screen DROP", "Screen KEEP", "Screen KEEP thi"]
    assert [(row["stage"], row["text_input"]["text_limit_chars"]) for row in _audit(auto)[1:]] == [
        ("Title/Abstract", 4), ("Title/Abstract", 4), ("Full-text", 8),
    ]


@pytest.mark.parametrize("structured", [False, True], ids=["plain-json", "instructor"])
def test_extraction_reports_the_actual_smart_prepared_length(make_batch, structured):
    text = "Results\n\nAn oversized paragraph for this budget"
    auto, client = make_batch(
        {"paper.pdf": (text, True)}, advanced_config={"enable_smart_truncation": True},
    )
    if structured:
        client.chat_completion_structured.side_effect = lambda messages, schema, **kwargs: (
            schema(material="ceramic"), 3,
        )

    summary = auto.process_pdfs()

    _assert_tail(auto.extraction_csv, auto.extraction_excel, METADATA_COLUMNS,
                 [[len(text), len("Results"), True, 8, "prepared"]])
    assert client.chat_completion_with_tokens.call_count == (1 if structured else 2)
    call = client.chat_completion_structured.call_args if structured else client.chat_completion_with_tokens.call_args
    assert call.args[0][-1]["content"].startswith("Extract Results\n\nApply")
    assert "oversized" not in call.args[0][-1]["content"]
    assert auto.extraction_results[0].fields["material"] == "ceramic"
    event = next(row for row in _audit(auto) if row["kind"] == "extraction")
    assert event["text_input"] == {
        "input_text_chars": len(text), "prepared_text_chars": 7,
        "text_truncated": True, "text_limit_chars": 8, "state": "prepared",
    }
    assert summary["text_truncation"]["stages"]["extraction"] == _counts(1, 1)


def test_cache_hits_keep_cells_empty_and_reset_all_stage_counts(make_batch, monkeypatch):
    text = "longer than the budget"
    auto, client = make_batch(
        {"paper.pdf": (text, True)}, cache_enabled=True, two_stage_screening=True,
    )
    first = auto.process_pdfs()
    assert first["text_truncation"]["stages"] == dict.fromkeys(
        ("title_abstract", "full_text", "extraction"), _counts(1, 1),
    )
    for path in auto.cache_folder.glob("screening_*.json"):
        assert not set(METADATA_COLUMNS).intersection(json.loads(path.read_text(encoding="utf-8")))
    client.reset_mock()
    truncate = Mock(side_effect=AssertionError("cache hits must not prepare text"))
    monkeypatch.setattr(auto, "_smart_truncate", truncate)

    second = auto.process_pdfs()

    expected = [[len(text), None, None, 8, "cache_hit"]]
    _assert_tail(auto.screening_csv, auto.screening_excel, METADATA_COLUMNS, expected)
    _assert_tail(auto.extraction_csv, auto.extraction_excel, METADATA_COLUMNS, expected)
    assert auto.screening_results[0].text_length == len(text)
    assert second["text_truncation"]["stages"] == dict.fromkeys(
        ("title_abstract", "full_text", "extraction"), _counts(cache_hit=1),
    )
    assert first["text_truncation"]["stages"]["full_text"] == _counts(1, 1)
    truncate.assert_not_called()
    client.chat_completion_with_tokens.assert_not_called()
    client.chat_completion_structured.assert_not_called()
    for row, limit in zip(_audit(auto)[-3:], [4, 8, 8]):
        assert row["cache_hit"] is True
        assert row["text_input"] == {
            "input_text_chars": len(text), "prepared_text_chars": None,
            "text_truncated": None, "text_limit_chars": limit, "state": "cache_hit",
        }


def test_conversion_failure_dry_run_and_worker_error_never_prepare_text(make_batch, monkeypatch):
    auto, client = make_batch(
        {"broken.pdf": ("conversion failure", False), "dry.pdf": ("valid text", True)},
        advanced_config={"dry_run": True},
    )
    (auto.pdf_folder / "worker.pdf").touch()
    converter = auto._extract_pdf_text

    def convert(path):
        if path.name == "worker.pdf":
            raise RuntimeError("fixture worker error")
        return converter(path)

    monkeypatch.setattr(auto, "_extract_pdf_text", convert)

    summary = auto.process_pdfs()

    _assert_tail(auto.screening_csv, auto.screening_excel, METADATA_COLUMNS, [
        [None, None, None, None, state] for state in ("conversion_failed", "dry_run", "worker_error")
    ])
    assert all(counts == _counts() for counts in summary["text_truncation"]["stages"].values())
    assert [event["kind"] for event in _audit(auto)] == ["run_config"]
    client.chat_completion_with_tokens.assert_not_called()


def test_extraction_header_collisions_preserve_existing_user_fields(make_batch):
    values = {
        "Input Text Chars": "user input count",
        "slr_meta_input_text_chars": "user prefixed count",
        "text_input_state": "user status",
    }
    auto, _ = make_batch({"paper.pdf": ("long fixture", True)}, values=values)

    summary = auto.process_pdfs()

    metadata_columns = summary["text_truncation"]["extraction_metadata_columns"]
    assert metadata_columns == dict(zip(METADATA_COLUMNS, [
        "slr_meta_input_text_chars_2", "prepared_text_chars", "text_truncated", "text_limit_chars",
        "slr_meta_text_input_state",
    ]))
    _assert_tail(auto.extraction_csv, auto.extraction_excel, list(metadata_columns.values()),
                 [[12, 8, True, 8, "prepared"]])
    columns, row = _csv_rows(auto.extraction_csv)
    assert columns[:6] == ["filename", *values, "processing_time", "api_tokens_used"]
    assert row[1:4] == list(values.values())
    excel_columns, excel_row = _xlsx_rows(auto.extraction_excel)
    assert len(set(excel_columns)) == len(excel_columns)
    assert list(excel_row[1:4]) == list(values.values())


def test_existing_evaluation_readers_accept_new_report_columns(make_batch, tmp_path, monkeypatch):
    values = {"coating_material": "ceramic", "preparation_method": "PVD", "source_excerpt": "quote"}
    text = "A source paper longer than the budget"
    auto, _ = make_batch({"paper.pdf": (text, True)}, values=values)
    auto.process_pdfs()

    monkeypatch.setattr(sys, "argv", [
        "make_label_sheet.py", str(auto.screening_excel), "--prefix", str(tmp_path / "review_"),
    ])
    make_label_sheet.main()
    key_path = tmp_path / "review_sample_key.csv"
    label_path = tmp_path / "review_label_sheet.xlsx"
    workbook = load_workbook(label_path)
    try:
        workbook.active["G2"] = "\u7eb3\u5165"
        workbook.save(label_path)
    finally:
        workbook.close()

    key_text = key_path.read_text(encoding="utf-8-sig")
    key_path.write_text(key_text.replace("paper.pdf", "previous-name.pdf"), encoding="utf-8-sig")
    evaluated = evaluate.load(label_path, key_path, str(auto.screening_excel))
    assert evaluated["text_length"].tolist() == [len(text)]
    assert evaluated["ai_decision"].tolist() == ["Likely Include"]
    assert evaluated["human"].tolist() == [1]
    for path in (auto.extraction_csv, auto.extraction_excel):
        table = read_table(path)
        assert table.columns.is_unique
        for field, hints in (
            ("filename", ["filename"]), ("coating_material", ["coating_material"]),
            ("preparation_method", ["preparation_method"]), ("source_excerpt", ["excerpt", "source"]),
        ):
            assert pick_col(table, None, hints, field) == field
            assert table[field].tolist() == [values.get(field, "paper.pdf")]


@pytest.mark.parametrize("text, limit, expected", [
    ("lead\n\nMethods A\n\nbody\n\nResults B\n\nan oversized paragraph at the end", 32,
     "lead\n\nMethods A\n\nbody\n\nResults B"),
    ("Methods first\n\nResults second\n\nDiscussion third", 18, "Methods first\n\nRes"),
    ("an oversized paragraph\n\nok", 3, "ok"),
    ("an oversized paragraph\n\nanother oversized paragraph", 3, "an "),
    ("Methods\n\n\n\n \n\nResults\n\ntail", 16, "Methods\n\nResults"),
    ("Methods A\n\nResults B", 11, "Methods A"),
    ("front\n\nMethods", 1, "M"),
    ("front\n\nMethods", 0, ""),
    ("front\n\nMethods", -2, ""),
    ("绪论\n\n研究结果\n\n附录段落很长", 8, "绪论\n\n研究结果"),
    (" A\n\n\n\nB ", 9, " A\n\n\n\nB "),
], ids=[
    "interleaved-order", "partial-priority-budget", "skip-oversized-regular", "prefix-fallback",
    "empty-paragraphs", "separator-boundary", "one-char", "zero-budget", "negative-budget",
    "unicode-chars", "unchanged-short-text",
])
def test_smart_truncation_preserves_order_and_budget(make_batch, text, limit, expected):
    auto, _ = make_batch({}, advanced_config={
        "enable_smart_truncation": True,
        "preserve_sections": ["method", "result", "discussion", "研究"],
    })

    actual = auto._smart_truncate(text, limit)

    assert actual == expected
    assert len(actual) <= max(limit, 0)


@pytest.mark.parametrize("smart, text, prepared", [
    (True, "A\r\n文  ", "A\r\n文  "),
    (True, "12345678", "12345678"),
    (False, "lead\n\nMethods A", "lead\n\nMe"),
], ids=["short-smart", "exact-limit-smart", "long-smart-disabled"])
def test_unaffected_requests_keep_prompt_bytes_and_cache_context(make_batch, monkeypatch, smart, text, prepared):
    auto, client = make_batch(
        {"paper.pdf": (text, True)}, advanced_config={"enable_smart_truncation": smart},
    )
    stage = "Full-text"
    expected_contexts = [
        auto._cache_key_context(kind="screening", text=text, stage=stage,
                                prompt={"screening_prompt": auto.screening_prompt, "stage": stage}),
        auto._cache_key_context(kind="extraction", text=text, prompt=auto._extraction_prompt_fingerprint_source()),
    ]
    cache_load = Mock(wraps=auto._cache_load)
    monkeypatch.setattr(auto, "_cache_load", cache_load)

    auto.process_pdfs()

    assert [call.kwargs["expected_context"] for call in cache_load.call_args_list] == expected_contexts
    assert all("truncation_fingerprint" not in context for context in expected_contexts)
    screening, extraction = client.chat_completion_with_tokens.call_args_list
    assert screening.args[0][0]["content"].encode("utf-8") == (
        b"You are a systematic review assistant. Only return valid JSON."
    )
    assert extraction.args[0][0]["content"].encode("utf-8") == (
        b"You extract structured data from research papers. Return only valid JSON."
    )
    assert screening.args[0][-1]["content"].encode("utf-8") == f"Screen {prepared}".encode("utf-8")
    expected_extraction = (
        f"Extract {prepared}\n\n"
        "Apply the Quote-Then-Answer method for every field:\n"
        "  1. Find all relevant elements for the field in the paper.\n"
        "  2. For each element, copy the shortest continuous verbatim passage into field_quote; "
        "join nonadjacent fragments with ' ... '. Preserve punctuation and inline citations. "
        "Split at headers or layout interruptions instead of rewriting. For table-only evidence, "
        "copy the header and data rows verbatim, separated by ' ... '.\n"
        "  3. Use the evidence fragments to write a complete answer without inventing unreported facts. "
        "If no relevant text exists, set both field_quote and field to 'Not reported'.\n\n"
        "Return ONLY valid JSON (no prose, no markdown fences):\n"
        '{\n    "material_quote": "verbatim sentence(s) from the text",\n'
        '    "material": "extracted value, or Not reported"\n}'
    )
    assert extraction.args[0][-1]["content"].encode("utf-8") == expected_extraction.encode("utf-8")


def test_long_smart_cache_isolates_legacy_results_without_counting_cache_hits_as_prepared(make_batch, monkeypatch):
    prepared = "lead\n\nMethods A\n\nbody\n\nResults B"
    text = prepared + "\n\nan oversized paragraph at the end"
    limit = len(prepared)
    auto, client = make_batch(
        {"paper.pdf": (text, True)}, two_stage_screening=True, cache_enabled=True,
        advanced_config={"max_text_chars": limit, "enable_smart_truncation": True},
    )
    legacy_contexts = {}
    for stage, decision in (("Title/Abstract", "Flag for Review"), ("Full-text", "Likely Exclude")):
        legacy_context = auto._cache_key_context(
            kind="screening", text=text, stage=stage, prompt={"screening_prompt": auto.screening_prompt, "stage": stage},
        )
        auto._cache_save(legacy_context["cache_key"], "screening", {
            "decision": decision, "stage": stage, "reasoning": "legacy", "notes": "",
            "text_length": len(text), "cache_metadata": legacy_context,
        })
        assert auto._cache_load(legacy_context["cache_key"], "screening", legacy_context)["decision"] == decision
        legacy_contexts[stage] = legacy_context
    legacy_extraction = auto._cache_key_context(
        kind="extraction", text=text, prompt=auto._extraction_prompt_fingerprint_source(),
    )
    auto._cache_save(legacy_extraction["cache_key"], "extraction", {
        "fields": {"material": "legacy"}, "cache_metadata": legacy_extraction,
    })
    assert auto._cache_load(legacy_extraction["cache_key"], "extraction", legacy_extraction)["fields"] == {
        "material": "legacy",
    }
    truncate = Mock(wraps=auto._smart_truncate)
    monkeypatch.setattr(auto, "_smart_truncate", truncate)

    first = auto.process_pdfs()

    assert truncate.call_count == 2
    assert all(call.args == (text, limit) for call in truncate.call_args_list)
    screening, extraction = client.chat_completion_with_tokens.call_args_list
    assert screening.args[0][-1]["content"] == f"Screen {prepared}"
    assert extraction.args[0][-1]["content"].startswith(f"Extract {prepared}\n\nApply")
    assert auto.screening_results[0].decision == "Likely Include"
    assert auto.extraction_results[0].fields["material"] == "ceramic"
    assert first["text_truncation"]["stages"] == {
        "title_abstract": _counts(cache_hit=1), "full_text": _counts(1, 1), "extraction": _counts(1, 1),
    }
    title, full, extracted = _audit(auto)[1:]
    assert title["cache_key"] == legacy_contexts["Title/Abstract"]["cache_key"]
    assert title["cache_hit"] is True
    for event, legacy_context in ((full, legacy_contexts["Full-text"]), (extracted, legacy_extraction)):
        assert event["cache_hit"] is False
        assert event["cache_key"] != legacy_context["cache_key"]
        assert event["text_hash"] == legacy_context["text_hash"]
        assert event["prompt_hash"] == legacy_context["prompt_hash"]
        assert event["text_cache"]["text_hash"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
        cache_file = auto.cache_folder / f"{event['kind']}_{event['cache_key']}.json"
        new_context = json.loads(cache_file.read_text(encoding="utf-8"))["cache_metadata"]
        assert new_context == {
            **legacy_context,
            "cache_key": event["cache_key"],
            "truncation_fingerprint": {
                "algorithm": "order_preserving_budget_v1",
                "snippet_hash": hashlib.sha256(prepared.encode("utf-8")).hexdigest(),
            },
        }
    client.reset_mock()
    truncate.reset_mock()

    second = auto.process_pdfs()

    assert truncate.call_count == 2
    client.chat_completion_with_tokens.assert_not_called()
    client.chat_completion_structured.assert_not_called()
    assert second["text_truncation"]["stages"] == {
        stage: _counts(cache_hit=1) for stage in ("title_abstract", "full_text", "extraction")
    }
    for event, original in zip(_audit(auto)[-3:], (title, full, extracted)):
        assert event["cache_key"] == original["cache_key"]
        assert event["cache_hit"] is True
        assert event["text_input"]["state"] == "cache_hit"
        assert event["text_input"]["prepared_text_chars"] is None
        assert event["text_input"]["text_truncated"] is None
    expected_tail = [[len(text), None, None, limit, "cache_hit"]]
    _assert_tail(auto.screening_csv, auto.screening_excel, METADATA_COLUMNS, expected_tail)
    _assert_tail(auto.extraction_csv, auto.extraction_excel, METADATA_COLUMNS, expected_tail)


def test_smart_precomputation_failure_still_uses_stage_error_audit(make_batch, monkeypatch):
    auto, client = make_batch({}, advanced_config={"enable_smart_truncation": True})
    truncate = Mock(side_effect=ValueError("could not prepare fixture"))
    monkeypatch.setattr(auto, "_smart_truncate", truncate)
    text = "long text fixture"

    screening = auto.screen_article(text, "paper.pdf")
    extraction = auto.extract_data(text, "paper.pdf")

    assert screening.decision == "Error"
    assert extraction.fields == {"material": "Error"}
    assert screening.text_input_state == extraction.text_input_state == "worker_error"
    assert truncate.call_count == 2
    client.chat_completion_with_tokens.assert_not_called()
    assert [event["kind"] for event in _audit(auto)] == ["screening", "extraction"]
    for event in _audit(auto):
        assert event["failure_stage"] == "processing"
        assert event["text_input"]["state"] == "worker_error"

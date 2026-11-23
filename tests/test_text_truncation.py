import csv
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

"""Offline integration tests for the medical benchmark."""

import copy
import importlib
import json
import socket
from decimal import Decimal

import openai
import pymupdf
import pytest
import requests
from openpyxl import load_workbook

from evaluation.medical import check_medical_quotes as quotes
from evaluation.medical import make_blind_review_sheet as blind
from evaluation.medical import run_medical_eval as runner
from evaluation.medical import score_medical_eval as scoring


DESIGN = "This was a randomized parallel group clinical trial."
POPULATION = "The trial randomized sixty adult participants with the condition."
INTERVENTION = "The intervention group received education and the control received usual care."
OUTCOMES = "The investigators measured symptoms at the twelve-week follow-up visit."
RATES = {key: Decimal(value) for key, value in zip(runner.RATE_NAMES, ("1", "2", "2", "4"))}


@pytest.fixture
def dataset(tmp_path):
    directory = tmp_path / "dataset"
    pdfs = directory / "pdfs"
    pdfs.mkdir(parents=True)
    rows = []
    for number in (1, 2):
        name = f"trial{number}.pdf"
        with pymupdf.open() as doc:
            page = doc.new_page()
            for y, text in ((74, DESIGN), (116, POPULATION)):
                page.insert_text((72, y), text, fontsize=11)
            page = doc.new_page()
            for y, text in ((74, INTERVENTION), (116, OUTCOMES)):
                page.insert_text((72, y), text, fontsize=11)
            doc.save(pdfs / name)
        rows.append({
            "paper_id": f"P{number:03}", "pdf_filename": name,
            **dict(zip(runner.GOLD.values(), (DESIGN, POPULATION, INTERVENTION, OUTCOMES))),
        })
    runner.write_csv(directory / "gold_standard.csv", list(rows[0]), rows)
    return directory


@pytest.fixture
def transports(monkeypatch):
    sent = []

    def offline(*args, **kwargs):
        raise AssertionError("Network disabled for the offline benchmark test")

    monkeypatch.setattr(socket.socket, "connect", offline)

    def data(payload):
        assert payload["model"] == runner.DEFAULT_MODEL
        assert payload["temperature"] == .05
        assert payload["max_tokens"] == 4000
        assert payload["thinking"] == {"type": "disabled"}
        structured = "tools" in payload
        sent.append((structured, payload["messages"]))
        fields = dict(zip(runner.FIELDS, ("parallel randomization", "sixty adults",
                                         "education versus usual care", "symptoms at twelve weeks")))
        message = {"role": "assistant", "content": json.dumps(fields)}
        if structured:
            fields.update(dict(zip((f"{f}_quote" for f in runner.FIELDS),
                                   (DESIGN, POPULATION, INTERVENTION, OUTCOMES))))
            message.update(content=None, tool_calls=[{
                "id": "mock_tool", "type": "function", "function": {
                    "name": payload["tools"][0]["function"]["name"],
                    "arguments": json.dumps(fields),
                },
            }])
        return {
            "id": "offline", "object": "chat.completion", "created": 1,
            "model": runner.DEFAULT_MODEL, "choices": [{"index": 0, "message": message,
                                                        "finish_reason": "tool_calls" if structured else "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }

    def plain(client, request, **kwargs):
        assert kwargs["allow_redirects"] is False
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(data(json.loads(request.body))).encode()
        return response

    def sdk(client, request, **kwargs):
        assert kwargs["follow_redirects"] is False
        http = importlib.import_module(type(request).__module__.split(".")[0])
        return http.Response(200, json=data(json.loads(request.content)), request=request)

    monkeypatch.setattr(requests.Session, "send", plain)
    monkeypatch.setattr(openai.DefaultHttpxClient, "send", sdk)
    return sent


def live(dataset, output, **kwargs):
    return runner.run(dataset, output, rates=RATES, api_key="mock-test-key", **kwargs)


def test_full_offline_pipeline_and_resume(dataset, transports, tmp_path):
    output = tmp_path / "run"
    dry = runner.run(dataset, output, dry_run=True)
    assert (dry["converted"], dry["errors"], dry["truncated"]) == (2, 0, 0)
    assert not transports

    smoke = live(dataset, output, limit=1)
    assert [structured for structured, _ in transports] == [False, True]
    assert smoke["groups"]["G2"]["parse_success_rate"] == 1
    assert smoke["groups"]["G1"]["total_tokens"] == 120
    raw, structured = (list(messages) for _, messages in transports)
    assert any(OUTCOMES in msg["content"] for msg in raw if msg.get("content"))
    assert any(OUTCOMES in msg["content"] for msg in structured if msg.get("content"))

    complete = live(dataset, output)
    assert [structured for structured, _ in transports] == [False, True, False, True]
    assert complete["reserved_or_spent_peak_cny"] == pytest.approx(.00112)
    for group in runner.GROUPS:
        assert complete["groups"][group]["prompt_tokens"] == 200
        assert complete["groups"][group]["per_100"]["total_tokens"] == 12000
        assert complete["groups"][group]["per_100"]["offpeak_cny"] == pytest.approx(.014)
        assert complete["groups"][group]["per_100"]["peak_cny"] == pytest.approx(.028)
    assert "mock-test-key" not in (output / "resume_state.json").read_text(encoding="utf-8")
    assert len(transports) == 4

    l1 = quotes.check(dataset, output)
    assert l1["groups"]["G2"]["strict_pass"]["count"] == 8
    assert l1["groups"]["G1"]["verbatim_match"]["rate"] is None
    detail = runner.read_csv(output / "l1_quotes.csv")
    first_opt = next(row for row in detail if row["group"] == "G2")
    assert first_opt["study_design_matched_pages"] == "1"
    assert first_opt["outcomes_measured_matched_pages"] == "2"

    assert blind.create(dataset, output, seed=9) == 16
    workbook = output / "blind_review_sheet.xlsx"
    wb = load_workbook(workbook)
    accuracy = wb[blind.ACCURACY_TAB]
    evidence = wb[blind.EVIDENCE_TAB]
    assert next(accuracy.values) == blind.ACCURACY_HEADERS
    assert not {"quote", "matched_pages", "evidence_grade"} & set(next(accuracy.values))
    assert next(evidence.values) == blind.EVIDENCE_HEADERS
    assert evidence.sheet_state == "hidden"
    assert {str(item.sqref) for item in accuracy.data_validations.dataValidation} == {"F2:F17"}
    assert {str(item.sqref) for item in evidence.data_validations.dataValidation} == {"G2:G9"}
    assert evidence.data_validations.dataValidation[0].formula1 == '"Supported,Weak,Unsupported"'
    assert {row[1] for row in accuracy.iter_rows(min_row=2, values_only=True)} == set(blind.SYSTEMS)
    assert all(case in " ".join(str(row) for row in wb["评分说明"].values)
               for case in ("P003", "P009", "P017"))
    with pytest.raises(ValueError, match="Fill the accuracy grade"):
        scoring.score(output, workbook)
    mapping = {(r["paper_id"], r["system"]): r["group"]
               for r in runner.read_csv(output / "blind_key.csv")}
    for row in accuracy.iter_rows(min_row=2):
        row[5].value = "Exact" if mapping[row[0].value, row[1].value] == "G2" else "Wrong"
    wb.save(workbook)
    with pytest.raises(ValueError, match="Fill the evidence grade"):
        scoring.score(output, workbook)
    for row in evidence.iter_rows(min_row=2):
        assert quotes.has_quote(row[4].value)
        row[6].value = "Supported"
    original = evidence["E2"].value
    evidence["E2"] = "Modified quote"
    wb.save(workbook)
    with pytest.raises(ValueError, match="Evidence quote differs"):
        scoring.score(output, workbook)
    evidence["E2"] = original
    wb.save(workbook)
    wb.close()
    verdict, passed, redlines, operational = scoring.score(output, workbook)
    assert verdict == "PROVISIONAL"  # A smoke subset cannot certify the full run.
    assert all(passed.values()) and all(operational.values()) and not any(redlines.values())
    report = (output / "medical_eval_report.md").read_text(encoding="utf-8")
    assert "Wilson" in report and "Per 100 papers" in report and "hallucination" in report
    assert "| L2 · No_Quote | 8 | 0 |" in report
    assert all(term in report for term in ("strict pass", "fragmented pass", "verbatim match"))


def test_full_size_blind_workbook_has_160_tasks(tmp_path, monkeypatch):
    ids = [f"P{number:03}" for number in range(1, 21)]
    gold = [{"paper_id": pid, "pdf_filename": f"{pid}.pdf",
             **dict.fromkeys(runner.GOLD.values(), "expert value")} for pid in ids]
    runner.write_csv(tmp_path / "gold_standard.csv", list(gold[0]), gold)
    run = {"run_id": "offline", "selected_ids": ids,
           "gold_sha256": runner.sha256(tmp_path / "gold_standard.csv")}
    groups = {group: {pid: {column: ("Not reported" if group == "G1" and column.endswith("_quote")
                                   else "sample value") for column in runner.OUTPUT_COLUMNS[3:]}
                      for pid in ids} for group in runner.GROUPS}
    monkeypatch.setattr(blind, "load_run", lambda output: (run, groups))
    pages = {f"{field}_matched_pages": "2" for field in runner.FIELDS}
    l1 = [{"run_id": "offline", "paper_id": pid, "group": group, **pages}
          for pid in ids for group in runner.GROUPS]
    runner.write_csv(tmp_path / "l1_quotes.csv", list(l1[0]), l1)
    assert blind.create(tmp_path, tmp_path, seed=9) == 160
    workbook = load_workbook(tmp_path / "blind_review_sheet.xlsx")
    try:
        assert workbook.sheetnames == [blind.ACCURACY_TAB, blind.EVIDENCE_TAB, "评分说明"]
        accuracy, evidence = workbook[blind.ACCURACY_TAB], workbook[blind.EVIDENCE_TAB]
        assert accuracy.max_row == 161
        assert evidence.max_row == 81
        assert {str(rule.sqref) for rule in accuracy.data_validations.dataValidation} == {"F2:F161"}
        assert {str(rule.sqref) for rule in evidence.data_validations.dataValidation} == {"G2:G81"}
    finally:
        workbook.close()


def test_budget_rejects_before_http_and_resume_keeps_paid_rows(dataset, transports, tmp_path):
    output = tmp_path / "blocked"
    sample = tmp_path / "reservation_sample"
    live(dataset, sample, limit=1)
    requests_log = json.loads((sample / "resume_state.json").read_text(encoding="utf-8"))["requests"]
    raw, optimized = requests_log
    assert raw["reserved_prompt_tokens"] < optimized["reserved_prompt_tokens"]
    reservation = runner.cost(optimized["reserved_prompt_tokens"], 4000, Decimal("2"), Decimal("4"))
    transports.clear()
    blocked = live(dataset, output, limit=1, budget_cny=reservation + Decimal(".0001"))
    # Only G1 is admitted. Its settled cost plus the next reservation exceeds the limit.
    assert len(transports) == 1
    assert blocked["stop_reason"] and blocked["reserved_or_spent_peak_cny"] == pytest.approx(.00028)
    assert blocked["groups"]["G2"]["attempted"] == 0
    assert len(runner.read_csv(output / "G1_extraction.csv")) == 1
    with pytest.raises(ValueError, match="missing"):
        quotes.check(dataset, output)
    resumed = live(dataset, output, limit=1)
    assert len(transports) == 2 and transports[-1][0]
    assert resumed["stop_reason"] is None


@pytest.mark.parametrize("as_bytes", [False, True])
def test_missing_usage_keeps_exact_utf8_reservation(tmp_path, as_bytes):
    state = {"requests": []}
    path = tmp_path / "state.json"
    body = json.dumps({"model": runner.DEFAULT_MODEL, "max_tokens": 4000,
                       "thinking": {"type": "disabled"},
                       "messages": [{"role": "user", "content": "临床试验" * 5000}]},
                      ensure_ascii=False)
    encoded = body.encode("utf-8")
    expected_prompt = len(encoded) + 4096
    peak = runner.cost(expected_prompt, 4000, Decimal("2"), Decimal("4"))
    offpeak = runner.cost(expected_prompt, 4000, Decimal("1"), Decimal("2"))
    budget = runner.Budget(state, lambda: runner.write_json(path, state), RATES, peak, runner.DEFAULT_MODEL)
    budget.current = "G1", "P001"
    request = requests.Request("POST", "https://api.deepseek.com/chat/completions", data=encoded).prepare()
    response = requests.Response()
    response._content = b'{"error":"an error without usage"}'

    def send(client, prepared, **kwargs):
        row, = json.loads(path.read_text(encoding="utf-8"))["requests"]
        assert row["status"] == "unsettled"
        assert row["body_bytes"] == len(encoded)
        assert row["reserved_prompt_tokens"] == expected_prompt
        assert Decimal(row["peak_cny"]) == peak
        assert Decimal(row["offpeak_cny"]) == offpeak
        return response

    assert budget.send(send, None, request, {}, encoded if as_bytes else body) is response
    assert budget.spent == peak
    with pytest.raises(runner.BudgetStop, match="would exceed"):
        budget.send(send, None, request, {}, body)


def test_abstention_and_nonverbatim_quotes_do_not_raise_match_rate(dataset, transports, tmp_path):
    output = tmp_path / "abstention"
    live(dataset, output, limit=1)
    path = output / "G2_extraction.csv"
    row, = runner.read_csv(path)
    row["study_design"] = row["study_design_quote"] = "Not reported"
    row["population_and_sample_size"] = "Not reported"
    row["population_and_sample_size_quote"] = "Not found"
    row["intervention_and_comparator_quote"] = INTERVENTION.replace("education", "medication")
    runner.write_csv(path, runner.OUTPUT_COLUMNS, [row])
    result = quotes.check(dataset, output)["groups"]["G2"]
    assert result["abstention_consistency"]["count"] == 1
    assert result["abstention_consistency"]["denominator"] == 2
    assert result["verbatim_match"]["count"] == 1
    assert result["verbatim_match"]["denominator"] == 2


def test_typographic_quotes_are_strict_matches_on_the_correct_pdf_page(dataset, monkeypatch, tmp_path):
    pdf = dataset / "pdfs" / "trial1.pdf"
    original = "The participant’s response was recorded as “blue” at the ‘baseline’ visit."
    quote = "The participant's response was recorded as \"blue\" at the 'baseline' visit."
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), DESIGN)
        document.new_page().insert_htmlbox(pymupdf.Rect(72, 72, 530, 145), original)
        document.save(pdf)
    text, pages = quotes.load_pages(pdf)
    assert original in pages[1][1]
    assert quotes.check_excerpt_with_pages(text, quote, pages, near=1.0)[0] != "pass"

    run = {
        "run_id": "typographic-quotes", "selected_ids": ["P001"],
        "gold_sha256": runner.sha256(dataset / "gold_standard.csv"),
        "pdf_sha256": {"P001": runner.sha256(pdf)},
    }
    prediction = dict.fromkeys(runner.OUTPUT_COLUMNS[3:], "")
    prediction.update(outcomes_measured="Response at baseline", outcomes_measured_quote=quote)
    monkeypatch.setattr(quotes, "load_run", lambda _: (run, {"G2": {"P001": prediction}}))

    summary = quotes.check(dataset, tmp_path)
    assert summary["groups"]["G2"]["strict_pass"] == quotes.metric(1, 1)
    detail, = runner.read_csv(tmp_path / "l1_quotes.csv")
    assert detail["outcomes_measured_matched_pages"] == "2"


def test_fragmented_pass_contributes_to_verbatim_and_page_metrics():
    rows = [
        {"status": status, "matched_pages": pages, "has_quote": True, "abstention": "n/a"}
        for status, pages in (("pass", "1"), ("pass_fragmented", "2,3"),
                              ("pass_fragmented", ""), ("fail", "4"))
    ]
    stats = quotes.quote_metrics(rows)
    assert stats["strict_pass"] == quotes.metric(1, 4)
    assert stats["fragmented_pass"] == quotes.metric(2, 4)
    assert stats["verbatim_match"] == quotes.metric(3, 4)
    assert stats["physical_page"] == quotes.metric(2, 4)


def test_frozen_pass_and_stop_boundaries():
    def group(exact, wrong, supported):
        return {"accuracy": {"Exact": quotes.metric(exact, 80), "Wrong": quotes.metric(wrong, 80)},
                "evidence": {"Supported": quotes.metric(supported, 80)}}

    run_group = {"parsed": 20, "requested": 20, "peak_cny": .5,
                 "elapsed_seconds": 20, "usage_complete": True}
    run = {"selected_ids": list(range(20)), "dataset_size": 20,
           "reserved_or_spent_peak_cny": 1, "stop_reason": None,
           "groups": {"G1": dict(run_group), "G2": dict(run_group)}}
    l1 = {"groups": {"G2": {"strict_pass": quotes.metric(60, 80),
                           "fragmented_pass": quotes.metric(4, 80),
                           "verbatim_match": quotes.metric(64, 80),
                           "physical_page": quotes.metric(64, 80)}}}
    scores = {"G1": group(45, 8, 0), "G2": group(45, 4, 68)}
    assert scoring.evaluate_gates(run, l1, scores)[0] == "PASS"
    assert scoring.evaluate_gates(run, l1, {**scores, "G2": group(45, 4, 51)})[0] == "STOP"
    assert scoring.evaluate_gates(run, l1, {**scores, "G2": group(44, 4, 68)})[0] == "STOP"
    slow = copy.deepcopy(run)
    slow["groups"]["G2"]["elapsed_seconds"] = 50.1
    low_gain = {"G1": group(45, 2, 0), "G2": group(46, 1, 68)}
    assert scoring.evaluate_gates(run, l1, low_gain)[0] == "PASS"  # Baseline ≤3%.
    assert scoring.evaluate_gates(slow, l1, low_gain)[0] == "STOP"
    run["groups"]["G2"]["parsed"] = 17
    assert scoring.evaluate_gates(run, l1, scores)[0] == "STOP"

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import housing_enhanced
from housing_enhanced import SystematicReviewAutomation, __version__, _structured_error_response
from llm_interface import LLMCallError


@pytest.fixture
def make_automation(monkeypatch, tmp_path):
    def make(adapter="openai_compatible", **kwargs):
        manager = SimpleNamespace(profile={"adapter": adapter})
        monkeypatch.setattr(SystematicReviewAutomation, "_init_llm", lambda self: manager)
        return SystematicReviewAutomation(
            api_key="fixture-api-key",
            pdf_folder=str(tmp_path / "pdfs"),
            output_folder=str(tmp_path / "output"),
            llm_provider=kwargs.pop("llm_provider", "DeepSeek"),
            llm_model=kwargs.pop("llm_model", "fixture-model"),
            extraction_fields=kwargs.pop("extraction_fields", ["coating_material"]),
            **kwargs,
        )

    return make


def audit_rows(auto):
    return [
        json.loads(line)
        for line in auto.audit_ledger.read_text(encoding="utf-8").splitlines()
    ]


@pytest.mark.parametrize(
    "provider, adapter, limit, screening_limit, structured_temperature",
    [
        ("DeepSeek", "openai_compatible", 4200, 1500, 0.31),
        ("Anthropic", "native_anthropic", 700, 700, "not_sent"),
    ],
)
def test_run_config_is_written_once_before_each_batch(
    make_automation, monkeypatch, provider, adapter, limit, screening_limit, structured_temperature
):
    auto = make_automation(
        adapter=adapter,
        llm_provider=provider,
        screening_prompt="  screen\n{text}\n",
        extraction_prompt=" extract\n{text}  ",
        extraction_fields=["First field", "Second field"],
        advanced_config={"temperature": 0.31, "max_tokens": limit},
        base_url="https://user:fixture-password@example.invalid/v1?api_key=fixture-url-key&keep=1",
    )
    calls = []

    def fake_batch():
        calls.append(len(audit_rows(auto)))
        return {}

    monkeypatch.setattr(auto, "_process_batch", fake_batch)
    auto.process_pdfs()
    auto.process_pdfs()

    assert calls == [1, 2]
    for row in audit_rows(auto):
        assert row["kind"] == "run_config"
        assert row["provider"] == provider
        assert row["model"] == "fixture-model"
        assert row["app_version"] == row["pipeline_version"] == __version__
        assert row["screening_prompt"] == "  screen\n{text}\n"
        assert row["extraction_prompt"] == " extract\n{text}  "
        assert row["extraction_fields"] == ["First field", "Second field"]
        assert row["screening"] == {"temperature": 0.31, "max_tokens": screening_limit}
        assert row["extraction"]["plain_json"] == {"temperature": 0.31, "max_tokens": limit}
        assert row["extraction"]["structured"] == {
            "temperature": structured_temperature, "max_tokens": limit,
        }
        assert row["base_url"] == "https://example.invalid/v1?api_key=%5Bredacted%5D&keep=1"
    ledger = auto.audit_ledger.read_text(encoding="utf-8")
    assert all(secret not in ledger for secret in ("fixture-api-key", "fixture-password", "fixture-url-key"))


def test_run_config_audit_failure_does_not_stop_processing(make_automation, monkeypatch):
    auto = make_automation(screening_prompt="private template {text}", parallel_processing=False)
    auto.pdf_folder.mkdir()
    (auto.pdf_folder / "paper.pdf").touch()
    monkeypatch.setattr(auto, "extract_text_from_pdf", lambda path: ("paper text", True))
    monkeypatch.setattr(auto, "_llm_call", lambda *args, **kwargs: ('{"decision":"Likely Exclude"}', 3))
    warning = Mock()
    monkeypatch.setattr(auto.logger, "warning", warning)
    canonical_json = housing_enhanced._canonical_json

    def fail_run_config(event):
        if isinstance(event, dict) and event.get("kind") == "run_config":
            raise ValueError("fixture-api-key private template")
        return canonical_json(event)

    monkeypatch.setattr(housing_enhanced, "_canonical_json", fail_run_config)
    assert auto.process_pdfs()["screened_count"] == 1
    event, = audit_rows(auto)
    assert event["decision"] == "Likely Exclude"
    warning.assert_called_once_with("Run configuration audit write failed")


def test_run_config_accepts_none_profile(make_automation):
    auto = make_automation()
    auto.llm_manager.profile = None
    auto.process_pdfs()
    row, = audit_rows(auto)
    assert row["kind"] == "run_config"
    assert row["extraction"] == {"plain_json": {"temperature": 0.05, "max_tokens": 4000}}


def test_structured_error_response_handles_bad_chains_and_completions():
    cycle = SimpleNamespace()
    cycle.original_error = cycle
    assert _structured_error_response(cycle) is None
    assert _structured_error_response(SimpleNamespace()) is None
    assert _structured_error_response(SimpleNamespace(last_completion=SimpleNamespace(choices=[]))) is None
    assert _structured_error_response(SimpleNamespace(last_completion=SimpleNamespace(choices=True))) is None

    error = SimpleNamespace(last_completion=SimpleNamespace(content="reply"))
    for _ in range(9):
        error = SimpleNamespace(original_error=error)
    assert _structured_error_response(error) == "reply"
    assert _structured_error_response(SimpleNamespace(original_error=error)) is None


@pytest.mark.parametrize("path", ["screen_article", "extract_data"])
@pytest.mark.parametrize(
    "raw, expected, stage",
    [("x" * 4100, "x" * 4000, "validation"), ("", "", "parse"), (None, None, "parse")],
)
def test_failed_plain_response_audit(make_automation, monkeypatch, path, raw, expected, stage):
    auto = make_automation()
    monkeypatch.setattr(auto, "_generate_dynamic_schema", lambda: None)
    monkeypatch.setattr(auto, "_llm_call", lambda *args, **kwargs: (raw, 3))
    getattr(auto, path)("paper", "paper.pdf")
    event, = audit_rows(auto)
    assert event["status"] == "error"
    assert event["raw_response"] == expected
    assert event["failure_stage"] == stage
    assert "error_category" not in event
    assert event["exception_type"] == "ValueError"


@pytest.mark.parametrize("representation", ["text", "tool_arguments", "unavailable"])
def test_final_structured_failure_extracts_original_reply(make_automation, monkeypatch, representation):
    auto = make_automation()
    calls = []
    provider_error = RuntimeError("fake structured failure")
    if representation == "text":
        message = SimpleNamespace(content="original text", tool_calls=[])
        expected = "original text"
    elif representation == "tool_arguments":
        message = SimpleNamespace(content=None, tool_calls=[
            SimpleNamespace(function=SimpleNamespace(arguments="x" * 4100)),
        ])
        expected = "x" * 4000
    else:
        message = None
        expected = None
    if message is not None:
        provider_error.last_completion = SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def fail_structured(*args):
        calls.append(args)
        raise LLMCallError("fake failure", category="unknown", retry_count=0, original_error=provider_error)

    monkeypatch.setattr(auto, "_generate_dynamic_schema", lambda: object())
    monkeypatch.setattr(auto, "_llm_structured_call", fail_structured)
    auto.extract_data("paper", "paper.pdf")
    event, = audit_rows(auto)
    assert event["status"] == "error"
    assert event["failure_stage"] == "structured"
    assert event["raw_response"] == expected
    assert event["exception_type"] == "LLMCallError"
    assert event["parse_retry_count"] == 0
    assert len(calls) == 1


def test_success_event_has_no_raw_reply(make_automation, monkeypatch):
    auto = make_automation()
    monkeypatch.setattr(auto, "_llm_call", lambda *args, **kwargs: ('{"decision":"Likely Include"}', 3))
    auto.screen_article("paper", "paper.pdf")
    event, = audit_rows(auto)
    assert event["status"] == "ok"
    assert "raw_response" not in event
    assert "failure_stage" not in event
    assert "exception_type" not in event


def test_fixed_cache_keys(make_automation):
    auto = make_automation(
        llm_model="deepseek-flash",
        screening_prompt="Screen this paper:\n{text}",
        extraction_prompt="Extract coating fields:\n{text}",
        extraction_fields=["coating_material", "preparation_method"],
    )
    text = "Cache audit fixture.\nCeramic coating on titanium."
    stage = "Full-text"
    screening = auto._cache_key_context(
        kind="screening", text=text, stage=stage,
        prompt={"screening_prompt": auto.screening_prompt, "stage": stage},
    )
    extraction = auto._cache_key_context(
        kind="extraction", text=text, prompt=auto._extraction_prompt_fingerprint_source(),
    )
    assert screening["cache_key"] == "788354ed11512f6d510a339aff4494bcdfbc3a797fa3c79739192a2c360711ed"
    assert extraction["cache_key"] == "f4b94f1e238298ba749c20bdc0a4b81c89f1e67d7490198a2a85d201bd9fc517"


@pytest.mark.parametrize(
    "path, bad_response",
    [
        ("screen_article", ""),
        ("screen_article", "not JSON " * 600),
        ("screen_article", '{"decision":"Maybe"}'),
        ("screen_article", '{"decision":["Likely Include"]}'),
        ("screen_article", '{"decision":"Likely Include","reasoning":[]}'),
        ("screen_article", '{"decision":"Likely Include","notes":3}'),
        ("extract_data", ""),
        ("extract_data", "not JSON"),
    ],
    ids=[
        "screen-empty", "screen-json", "screen-decision", "screen-decision-type",
        "screen-reasoning", "screen-notes", "extract-empty", "extract-json",
    ],
)
@pytest.mark.parametrize("recovers", [True, False])
def test_plain_parse_retry_uses_identical_request_and_keeps_first_reply(
    make_automation, monkeypatch, path, bad_response, recovers
):
    auto = make_automation()
    monkeypatch.setattr(auto, "_generate_dynamic_schema", lambda: None)
    valid = '{"decision":"Likely Include"}' if path == "screen_article" else '{"coating_material":"TiN"}'
    second = valid if recovers else "still not JSON " * 300
    replies = iter([(bad_response, 8), (second, 13)])
    requests = []

    def fake_call(messages, **parameters):
        requests.append(json.dumps({"messages": messages, **parameters}, sort_keys=True))
        return next(replies)

    monkeypatch.setattr(auto, "_llm_call", fake_call)
    result = getattr(auto, path)("paper", "paper.pdf")
    event, = audit_rows(auto)
    assert len(requests) == 2
    assert requests[0] == requests[1]
    assert event["parse_retry_count"] == 1
    assert event["retry_count"] == 0
    assert event["retry_first_raw_response"] == bad_response[:4000]
    if recovers:
        assert "raw_response" not in event
        assert "exception_type" not in event
    else:
        assert event["raw_response"] == second[:4000]
        assert event["exception_type"] == "ValueError"
    assert event["api_tokens_used"] == result.api_tokens_used == auto.stats["total_api_tokens"] == 21
    assert event["status"] == ("ok" if recovers else "error")
    if path == "screen_article":
        assert result.decision == ("Likely Include" if recovers else "Error")
    else:
        assert result.fields["coating_material"] == ("TiN" if recovers else "Error")
    assert len(list(auto.cache_folder.glob("*.json"))) == int(recovers)


@pytest.mark.parametrize("path", ["screen_article", "extract_data"])
@pytest.mark.parametrize("category", ["timeout", "auth_error"])
@pytest.mark.parametrize("after_parse_failure", [False, True])
def test_provider_failure_does_not_trigger_parse_retry(
    make_automation, monkeypatch, path, category, after_parse_failure
):
    auto = make_automation()
    monkeypatch.setattr(auto, "_generate_dynamic_schema", lambda: None)
    calls = []

    def fail_request(messages, **kwargs):
        calls.append(messages)
        if after_parse_failure and len(calls) == 1:
            return "first reply is not JSON", 2
        auto._llm_thread_local.metadata = {"error_category": category}
        raise LLMCallError(
            "fake provider failure", category=category, retry_count=0, original_error=RuntimeError(),
        )

    monkeypatch.setattr(auto, "_llm_call", fail_request)
    getattr(auto, path)("paper", "paper.pdf")
    event, = audit_rows(auto)
    assert len(calls) == 1 + int(after_parse_failure)
    assert event["status"] == "error"
    assert event["parse_retry_count"] == int(after_parse_failure)
    assert event["raw_response"] is None
    if after_parse_failure:
        assert event["retry_first_raw_response"] == "first reply is not JSON"
    else:
        assert "retry_first_raw_response" not in event
    assert event["error_category"] == category
    assert event["exception_type"] == "LLMCallError"
    assert event["failure_stage"] == "request"
    assert not list(auto.cache_folder.glob("*.json"))


@pytest.mark.parametrize(
    "path, raw",
    [
        ("screen_article", "[]"),
        ("screen_article", '{"decision":"Likely Include","notes":"Malformed LLM response; raw response content was not stored."}'),
        ("extract_data", "[]"),
        ("extract_data", '{"other":"TiN"}'),
    ],
)
def test_other_existing_plain_errors_do_not_trigger_parse_retry(make_automation, monkeypatch, path, raw):
    auto = make_automation()
    monkeypatch.setattr(auto, "_generate_dynamic_schema", lambda: None)
    replies = iter([(raw, 2)])
    monkeypatch.setattr(auto, "_llm_call", lambda *args, **kwargs: next(replies))
    getattr(auto, path)("paper", "paper.pdf")
    event, = audit_rows(auto)
    assert event["status"] == "error"
    assert event["parse_retry_count"] == 0
    assert event["raw_response"] == raw


def test_plain_extraction_still_accepts_non_string_fields(make_automation, monkeypatch):
    auto = make_automation()
    monkeypatch.setattr(auto, "_generate_dynamic_schema", lambda: None)
    replies = iter([('{"coating_material":["TiN"],"coating_material_quote":7}', 2)])
    monkeypatch.setattr(auto, "_llm_call", lambda *args, **kwargs: next(replies))
    result = auto.extract_data("paper", "paper.pdf")
    event, = audit_rows(auto)
    assert result.fields == {"coating_material": ["TiN"], "coating_material_quote": 7}
    assert event["status"] == "ok"
    assert event["parse_retry_count"] == 0
    assert "raw_response" not in event

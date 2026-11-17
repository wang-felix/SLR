import json
from types import SimpleNamespace

import pytest

from housing_enhanced import SystematicReviewAutomation, __version__
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
        ("Anthropic", "native_anthropic", 700, 700, "未传"),
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
    assert event["error_category"] == "ValueError"


@pytest.mark.parametrize("representation", ["text", "tool_arguments", "unavailable"])
def test_final_structured_failure_extracts_original_reply(make_automation, monkeypatch, representation):
    auto = make_automation()
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
        raise LLMCallError("fake failure", category="unknown", retry_count=0, original_error=provider_error)

    monkeypatch.setattr(auto, "_generate_dynamic_schema", lambda: object())
    monkeypatch.setattr(auto, "_llm_structured_call", fail_structured)
    auto.extract_data("paper", "paper.pdf")
    event, = audit_rows(auto)
    assert event["status"] == "error"
    assert event["failure_stage"] == "structured"
    assert event["raw_response"] == expected


def test_success_event_has_no_raw_reply(make_automation, monkeypatch):
    auto = make_automation()
    monkeypatch.setattr(auto, "_llm_call", lambda *args, **kwargs: ('{"decision":"Likely Include"}', 3))
    auto.screen_article("paper", "paper.pdf")
    event, = audit_rows(auto)
    assert event["status"] == "ok"
    assert "raw_response" not in event
    assert "failure_stage" not in event


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
    assert extraction["cache_key"] == "73cde48999a4d73506cc3d769423f67e5f6879a08887c10c528db3096a2744e8"

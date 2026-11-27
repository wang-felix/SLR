import csv
import json
import sys
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openpyxl import load_workbook

import llm_interface
from housing_enhanced import SystematicReviewAutomation
from llm_interface import LLMCallError, LLMManager


MESSAGES = [{"role": "user", "content": "paper text"}]


def make_manager():
    return LLMManager(
        "DeepSeek", "fixture-key", "fixture-model",
        rate_limit_config={"min_interval": 0},
    )


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "compatible"])
def test_plain_provider_token_metadata_keeps_two_value_return(monkeypatch, provider):
    with make_manager() as manager:
        usage = SimpleNamespace(prompt_tokens=11, completion_tokens=4, total_tokens=15)
        if provider == "openai":
            instance = object.__new__(llm_interface.OpenAIProvider)
            instance.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                create=Mock(return_value=SimpleNamespace(
                    usage=usage, choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                )),
            )))
        elif provider == "anthropic":
            instance = object.__new__(llm_interface.AnthropicProvider)
            instance.client = SimpleNamespace(messages=SimpleNamespace(
                create=Mock(return_value=SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=11, output_tokens=4),
                    content=[SimpleNamespace(text="ok")],
                )),
            ))
        elif provider == "gemini":
            instance = object.__new__(llm_interface.GeminiProvider)
            instance._sdk = "google-genai"
            instance.genai_types = SimpleNamespace(GenerateContentConfig=Mock())
            instance._client = SimpleNamespace(models=SimpleNamespace(
                generate_content=Mock(return_value={
                    "text": "ok",
                    "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 3,
                                      "thoughtsTokenCount": 1, "totalTokenCount": 15},
                }),
            ))
        else:
            instance = manager.provider
            monkeypatch.setattr(llm_interface.requests, "post", Mock(return_value=Mock(
                json=Mock(return_value={"usage": vars(usage), "choices": [{"message": {"content": "ok"}}]}),
            )))
        instance.model = "fixture-model"
        manager.provider = instance
        result = manager.chat_completion_with_tokens(MESSAGES)
        assert type(result) is tuple
        assert result == ("ok", 15)
        for metadata in (manager.last_call_metadata, manager.get_last_call_metadata()):
            assert (metadata["prompt_tokens"], metadata["completion_tokens"], metadata["total_tokens"]) == (11, 4, 15)

        # Legacy adapters/mocks that report only the total must not inherit the split.
        monkeypatch.setattr(instance, "chat_completion_with_tokens", Mock(return_value=("legacy", 9)))
        assert manager.chat_completion_with_tokens(MESSAGES) == ("legacy", 9)
        metadata = manager.get_last_call_metadata()
        assert (metadata["prompt_tokens"], metadata["completion_tokens"], metadata["total_tokens"]) == (None, None, 9)


def test_cancelled_request_does_not_reuse_previous_usage(monkeypatch):
    with make_manager() as manager:
        call = Mock(return_value=("previous response", 9))
        monkeypatch.setattr(manager.provider, "chat_completion_with_tokens", call)
        manager.chat_completion_with_tokens(MESSAGES)
        stop = Event()
        stop.set()
        with pytest.raises(InterruptedError):
            manager.chat_completion_with_tokens(MESSAGES, retry_stop_event=stop)
        assert manager.get_last_call_metadata() == {}
        call.assert_called_once()


@pytest.mark.parametrize("adapter", ["native_openai", "openai_compatible", "native_anthropic"])
def test_instructor_reports_aggregate_usage_and_keeps_legacy_return(monkeypatch, adapter):
    import anthropic
    import instructor
    import openai

    def usage(prompt, completion):
        if adapter == "native_anthropic":
            return SimpleNamespace(input_tokens=prompt, output_tokens=completion)
        return SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion)

    parsed = SimpleNamespace(_total_usage=usage(23, 7))
    creator = Mock(return_value=(parsed, SimpleNamespace(usage=usage(13, 4))))
    api = SimpleNamespace(create_with_completion=creator)
    raw = Mock()
    monkeypatch.setattr(openai, "OpenAI", Mock(return_value=raw))
    monkeypatch.setattr(anthropic, "Anthropic", Mock(return_value=raw))
    monkeypatch.setattr(instructor, "from_openai", Mock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=api),
    )))
    monkeypatch.setattr(instructor, "from_anthropic", Mock(return_value=SimpleNamespace(messages=api)))
    with make_manager() as manager:
        manager.profile = {**manager.profile, "adapter": adapter}
        assert manager.chat_completion_structured(MESSAGES, object) == (parsed, 17)
        metadata = manager.get_last_call_metadata()
        assert (metadata["prompt_tokens"], metadata["completion_tokens"], metadata["total_tokens"]) == (23, 7, 30)
    raw.close.assert_called_once()


@pytest.fixture
def build(monkeypatch, tmp_path):
    manager = Mock(profile={"adapter": "openai_compatible"})
    manager.get_last_call_metadata.return_value = {}
    monkeypatch.setattr(SystematicReviewAutomation, "_init_llm", lambda self: manager)
    instances = []

    def create(**config):
        auto = SystematicReviewAutomation(
            api_key="fixture-key", pdf_folder=str(tmp_path / "pdfs"),
            output_folder=str(tmp_path / "output"), cache_folder=str(tmp_path / "cache"),
            extraction_fields=["Study Design"], extraction_prompt="Extract the study design.\n{text}",
            advanced_config=config,
        )
        instances.append(auto)
        return auto, manager

    yield create
    for auto in instances:
        auto.close()


def audit(auto):
    return [json.loads(line) for line in auto.audit_ledger.read_text(encoding="utf-8").splitlines()]


def test_raw_and_optimized_use_disjoint_caches_and_export_quotes(build):
    opt, manager = build()
    raw, _ = build(extraction_mode="raw")
    emitted_quote = "Participants were randomly allocated."
    manager.chat_completion_structured.return_value = (
        SimpleNamespace(model_dump=lambda: {"study_design": "randomized", "study_design_quote": emitted_quote}), 15,
    )
    manager.chat_completion_with_tokens.return_value = ('{"study_design":"plain answer"}', 10)
    manager.get_last_call_metadata.side_effect = [
        {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
        {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    ]
    optimized = opt.extract_data("paper text", "test.pdf")
    baseline = raw.extract_data("paper text", "test.pdf")
    assert optimized.fields["Study Design"] == "randomized"
    assert baseline.fields == {"study_design": "plain answer", "Study Design": "plain answer"}
    assert "Quote-Then-Answer" in manager.chat_completion_structured.call_args.args[0][1]["content"]
    assert manager.chat_completion_with_tokens.call_args.args[0][1]["content"] == "Extract the study design.\npaper text"
    assert opt.extract_data("paper text", "test.pdf").fields == optimized.fields
    assert raw.extract_data("paper text", "test.pdf").fields == baseline.fields
    manager.chat_completion_structured.assert_called_once()
    manager.chat_completion_with_tokens.assert_called_once()

    a, cached_a = audit(opt)
    b, cached_b = audit(raw)
    assert a["cache_key"] != b["cache_key"]
    assert [a["extraction_mode"], b["extraction_mode"]] == ["optimized", "raw"]
    assert [cached_a["cache_key"], cached_b["cache_key"]] == [a["cache_key"], b["cache_key"]]
    assert [a["prompt_tokens"], a["completion_tokens"], a["total_tokens"]] == [11, 4, 15]
    for row in (cached_a, cached_b):
        assert row["cache_hit"]
        assert [row[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")] == [0, 0, 0]

    for include_quotes in (True, False):
        opt.include_extraction_quotes = include_quotes
        opt.extraction_results = [optimized, baseline]
        opt.write_extraction_csv()
        opt.write_extraction_excel()
        with opt.extraction_csv.open(encoding="utf-8", newline="") as file:
            first, second = csv.DictReader(file)
        workbook = load_workbook(opt.extraction_excel, read_only=True)
        try:
            headers, first_values, second_values = workbook.active.values
        finally:
            workbook.close()
        if include_quotes:
            assert first["Study Design_quote"] == emitted_quote
            assert second["Study Design_quote"] == ""
            column = headers.index("Study Design Quote")
            assert first_values[column] == emitted_quote
            assert second_values[column] in (None, "")
        else:
            assert "Study Design_quote" not in first
            assert "Study Design Quote" not in headers


def test_audit_counts_structured_fallback_and_parse_retry(build):
    auto, manager = build()
    manager.chat_completion_structured.side_effect = LLMCallError(
        "schema mode unsupported", category="unsupported_feature",
        retry_count=0, original_error=ValueError(),
    )
    manager.chat_completion_with_tokens.side_effect = [
        ("not JSON", 11), ('{"study_design":"randomized"}', 11),
    ]
    manager.get_last_call_metadata.side_effect = [
        {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
        {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
    ]
    result = auto.extract_data("paper text", "test.pdf")
    event, = audit(auto)
    assert result.fields["Study Design"] == "randomized"
    assert event["parse_retry_count"] == 1
    assert [event[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")] == [18, 7, 25]
    assert result.api_tokens_used == auto.stats["total_api_tokens"] == 25


@pytest.mark.parametrize("converter_type", ["modern", "legacy", "wrapped_layout", "wrapped_legacy"])
def test_pdf_conversion_only_passes_supported_keywords(monkeypatch, build, tmp_path, converter_type):
    import pymupdf

    page_options = {}

    def modern(path, *, use_ocr=True, show_progress=True, pages=None):
        page_options.update(use_ocr=use_ocr, show_progress=show_progress, pages=pages)
        return "fixture extracted text"

    def legacy(path, *, pages=None, show_progress=True):
        page_options.update(pages=pages, show_progress=show_progress)
        return "fixture extracted text"

    implementation = modern if converter_type in {"modern", "wrapped_layout"} else legacy

    def wrapper(*args, **kwargs):
        return implementation(*args, **kwargs)

    monkeypatch.setitem(sys.modules, "pymupdf4llm", SimpleNamespace(
        to_markdown=wrapper if converter_type.startswith("wrapped") else implementation,
        _use_layout=converter_type == "wrapped_layout",
        _layout_to_markdown=modern,
        helpers=SimpleNamespace(pymupdf_rag=SimpleNamespace(to_markdown=legacy)),
    ))
    auto, _ = build(skip_pages=1)
    path = tmp_path / "sample.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.new_page()
        doc.save(path)
    assert auto.extract_text_from_pdf(path) == ("fixture extracted text", True)
    expected = {"pages": [1], "show_progress": False}
    if implementation is modern:
        expected["use_ocr"] = False
    assert page_options == expected

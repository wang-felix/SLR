"""Run the paired medical extraction benchmark with durable request accounting."""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from contextlib import ExitStack, contextmanager
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests

from atomic_files import atomic_open, atomic_write_text
from housing_enhanced import SystematicReviewAutomation

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "test_output" / "medical_eval_v1"
PROMPTS = ROOT / "evaluation" / "prompts"
FIELDS = tuple((PROMPTS / "medical_extraction_fields_v1.txt").read_text(encoding="utf-8").split())
GOLD = {field: "gold_" + field for field in FIELDS}
GOLD["outcomes_measured"] = "gold_outcomes_and_main_findings"
GROUPS = {"G1": "raw", "G2": "optimized"}
LIMIT = 100_000
MAX_OUTPUT = 4000
# Reserve a full 1M-token context, allowing for the binary interpretation of 1M.
MAX_CONTEXT = 1_048_576
BUDGET_DEFAULT = Decimal("8")
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_PROMPT_FILE = PROMPTS / "medical_extraction_prompt_opt_v1.txt"
OFFICIAL_MODELS = {DEFAULT_MODEL, "deepseek-v4-pro"}
SETTINGS = {"temperature": 0.05, "max_tokens": MAX_OUTPUT, "max_text_chars": LIMIT,
            "strip_references": True, "thinking": "disabled"}
RATE_NAMES = ("offpeak_input", "offpeak_output", "peak_input", "peak_output")
OUTPUT_COLUMNS = ("run_id", "paper_id", "status", *FIELDS, *(f"{field}_quote" for field in FIELDS))


class BudgetStop(InterruptedError):
    pass


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, columns, rows):
    with atomic_open(path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def indexed(rows, key):
    result = {}
    for row in rows:
        value = key(row)
        if value in result or not value:
            raise ValueError(f"Duplicate or empty key: {value}")
        result[value] = row
    return result


def paper_path(dataset, paper):
    name = paper["pdf_filename"]
    if name != Path(name).name or not name.lower().endswith(".pdf"):
        raise ValueError(f'Invalid PDF filename for {paper["paper_id"]}')
    directory = (dataset / "pdfs").resolve()
    path = (directory / name).resolve()
    if path.parent != directory:
        raise ValueError(f'PDF outside dataset for {paper["paper_id"]}')
    return path


def load_gold(dataset):
    rows = read_csv(dataset / "gold_standard.csv")
    if not rows or any(not row.get(k) for row in rows for k in ("paper_id", "pdf_filename", *GOLD.values())):
        raise ValueError("Missing gold-standard rows or fields")
    indexed(rows, lambda row: row["paper_id"])
    indexed(rows, lambda row: row["pdf_filename"])
    return rows


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()



def cost(prompt, completion, input_rate, output_rate):
    return (prompt * input_rate + completion * output_rate) / 1_000_000


class Budget:
    """Charge the peak ceiling per HTTP request before sending it."""

    def __init__(self, state, save, rates, limit, model):
        self.records = state["requests"]
        self.save = save
        self.rates = rates
        self.limit = limit
        self.model = model
        self.current = None
        self.reason = None
        self.spent = sum((Decimal(r["peak_cny"]) for r in self.records), Decimal(0))

    def stop(self, reason):
        self.reason = reason
        raise BudgetStop(reason)

    def send(self, send, client, request, kwargs, body):
        if self.reason:
            raise BudgetStop(self.reason)
        url = urlsplit(str(request.url))
        if (self.current is None or request.method != "POST"
                or url.scheme != "https" or url.netloc != "api.deepseek.com"
                or url.path not in ("/chat/completions", "/v1/chat/completions")):
            self.stop("Unexpected HTTP request in the medical evaluation")
        payload = json.loads(body)
        max_tokens = payload.get("max_tokens")
        if (payload.get("model") != self.model or type(max_tokens) is not int
                or not 0 < max_tokens <= MAX_OUTPUT or payload.get("stream")
                or payload.get("thinking") != {"type": "disabled"}):
            self.stop("Request parameters exceed the approved budget configuration")
        body_bytes = len(body.encode("utf-8")) if isinstance(body, str) else len(body)
        reserved_prompt = min(MAX_CONTEXT, body_bytes + 4096)
        peak = cost(reserved_prompt, max_tokens, self.rates["peak_input"], self.rates["peak_output"])
        if self.spent + peak > self.limit:
            self.stop("The next request's peak-price reservation would exceed the budget")
        group, paper_id = self.current
        row = {"group": group, "paper_id": paper_id, "peak_cny": str(peak),
               "offpeak_cny": str(cost(reserved_prompt, max_tokens, self.rates["offpeak_input"],
                                        self.rates["offpeak_output"])),
               "body_bytes": body_bytes, "reserved_prompt_tokens": reserved_prompt,
               "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
               "finish_reason": None, "status": "unsettled"}
        self.records.append(row)
        self.spent += peak
        self.save()

        response = send(client, request, **kwargs)
        try:
            data = response.json()
            usage = data["usage"]
            prompt, completion, total = (usage[key] for key in (
                "prompt_tokens", "completion_tokens", "total_tokens"))
            if any(type(n) is not int or n < 0 for n in (prompt, completion, total)):
                raise ValueError("Invalid usage")
        except (AttributeError, KeyError, TypeError, ValueError):
            return response  # Keep the reservation when billing usage is unavailable.
        if prompt > reserved_prompt or completion > max_tokens or total != prompt + completion:
            self.stop("Provider usage did not match the reserved token bounds")
        off, settled = (cost(prompt, completion, self.rates[f"{period}_input"],
                             self.rates[f"{period}_output"]) for period in ("offpeak", "peak"))
        choices = data.get("choices") or []
        reason = choices[0].get("finish_reason") if choices else None
        self.spent += settled - peak
        row.update(peak_cny=str(settled), offpeak_cny=str(off), prompt_tokens=prompt,
                   completion_tokens=completion, total_tokens=total, status="settled",
                   finish_reason=reason)
        self.save()
        return response

    @contextmanager
    def guard(self):
        # These are the two transports used by the production plain and instructor paths.
        import openai

        sdk_transport = openai.DefaultHttpxClient
        plain_send, sdk_send = requests.Session.send, sdk_transport.send

        def plain(client, request, **kwargs):
            kwargs["allow_redirects"] = False
            return self.send(plain_send, client, request, kwargs, request.body)

        def structured(client, request, **kwargs):
            kwargs["follow_redirects"] = False
            return self.send(sdk_send, client, request, kwargs, request.content)

        with patch.object(requests.Session, "send", plain), patch.object(sdk_transport, "send", structured):
            yield


@contextmanager
def exclusive_run(output):
    # OS advisory lock is released by process exit, including unclean shutdown.
    with (output / ".run.lock").open("a+b") as file:
        if os.name == "nt":
            import msvcrt

            file.seek(0)
            try:
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ValueError("Another evaluation is using this output directory") from exc
        else:
            import fcntl

            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class MedicalAutomation(SystematicReviewAutomation):
    def _llm_call(self, messages, **kwargs):
        # The product already disables thinking for the DeepSeek instructor path.
        return super()._llm_call(messages, thinking={"type": "disabled"}, **kwargs)

    def _strip_references_from_md(self, text):
        return strip_reference_text(text)


def strip_reference_text(text):
    heading = re.search(
        r"(?im)^[ \t]*(?:#{1,6}[ \t]+)?(?:\d+[.)]?[ \t]+)?"
        r"[*_]*(?:references|bibliography|works cited|literature cited)[*_]*[ \t]*:?[ \t]*$",
        text,
    )
    return text[:heading.start()].rstrip() if heading else text


def automation(output, dataset, model, key, mode, prompt=None, *, dry_run=False):
    return MedicalAutomation(
        api_key=key, pdf_folder=str(dataset / "pdfs"), output_folder=str(output),
        llm_provider="DeepSeek", llm_model=model, extraction_fields=list(FIELDS),
        extraction_prompt=prompt, cache_enabled=False,
        advanced_config={**SETTINGS, "dry_run": dry_run, "extraction_mode": mode},
    )


def preflight(dataset, output, papers, model):
    texts, report = {}, []
    auto = automation(output / "preflight", dataset, model, "", "raw", dry_run=True)
    try:
        for paper in papers:
            path = paper_path(dataset, paper)
            text, ok = auto.extract_text_from_pdf(path) if path.is_file() else ("", False)
            metadata = auto._last_text_extraction_metadata.get(auto._pdf_display_name(path), {})
            method = metadata.get("extraction_method")
            ok = bool(ok and method == "pymupdf4llm" and text.strip())
            if ok:
                texts[paper["paper_id"]] = text
            report.append({"paper_id": paper["paper_id"], "status": "ok" if ok else "parse_error",
                           "chars": len(text) if ok else None, "truncated": ok and len(text) > LIMIT})
    finally:
        auto.close()
    lengths = [row["chars"] for row in report if row["status"] == "ok"]
    summary = {"papers": report, "converted": len(lengths), "errors": len(report) - len(lengths),
               "truncated": sum(row["truncated"] for row in report),
               "min_chars": min(lengths, default=None), "max_chars": max(lengths, default=None)}
    return texts, summary


def summarize_group(records, requests, selected):
    n = len(selected)
    known = sum(row["status"] == "settled" for row in requests)
    accounting_complete = known == len(requests)
    counts = {key: sum(row[key] or 0 for row in requests) if accounting_complete else None for key in (
        "prompt_tokens", "completion_tokens", "total_tokens")}
    costs = {key: sum((Decimal(r[key]) for r in requests), Decimal(0))
             for key in ("offpeak_cny", "peak_cny")}
    seconds = sum(r["elapsed_seconds"] for r in records)
    metrics = {**counts, **{k: float(v) for k, v in costs.items()}, "elapsed_seconds": seconds}
    return {"requested": n, "attempted": len(records),
            "parsed": sum(r["status"] == "ok" for r in records),
            "parse_success_rate": sum(r["status"] == "ok" for r in records) / n,
            "api_requests": len(requests), "unsettled_requests": len(requests) - known,
            "usage_complete": accounting_complete, **metrics,
            "per_100": {k: v * 100 / n if v is not None else None for k, v in metrics.items()}
            if len(records) == n else None}


def load_prompt(prompt_file):
    prompt = Path(prompt_file).read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError(f"Optimization prompt file is empty: {prompt_file}")
    return prompt


def prepare_run(dataset, output, selected, model, prompt_file):
    texts, report = preflight(dataset, output, selected, model)
    report["prompt_file"] = Path(prompt_file).name
    write_json(output / "dry_run_summary.json", report)
    return texts, report


def run(dataset, output, *, model=DEFAULT_MODEL, limit=None, dry_run=False,
        rates=None, budget_cny=BUDGET_DEFAULT, api_key="", prompt_file=DEFAULT_PROMPT_FILE):
    if dry_run:
        load_prompt(prompt_file)
    output.mkdir(parents=True, exist_ok=True)
    all_papers = load_gold(dataset)
    selected = all_papers[:limit]
    if not selected:
        raise ValueError("Select at least one paper")
    with exclusive_run(output):
        if dry_run:
            _, report = prepare_run(dataset, output, selected, model, prompt_file)
            return report
        return run_live(dataset, output, all_papers, selected, None, model, rates, budget_cny, api_key,
                        prompt_file=prompt_file)


def run_live(dataset, output, all_papers, selected, texts, model, rates, budget_cny, api_key,
             *, prompt_file=DEFAULT_PROMPT_FILE):
    if model not in OFFICIAL_MODELS or not api_key or not rates:
        raise ValueError("Live runs require an official DeepSeek model, DEEPSEEK_API_KEY and all four rates")
    rates = {key: Decimal(str(rates[key])) for key in RATE_NAMES}
    budget_cny = Decimal(str(budget_cny))
    if any(not value.is_finite() or value <= 0 for value in (*rates.values(), budget_cny)):
        raise ValueError("Rates and budget must be positive finite numbers")
    if any(rates[f"peak_{part}"] < rates[f"offpeak_{part}"] for part in ("input", "output")):
        raise ValueError("Peak rates must be at least the off-peak rates")
    opt_prompt = load_prompt(prompt_file)
    fingerprint = {"model": model, "settings": SETTINGS, "fields": FIELDS, "prompt": opt_prompt,
                   "rated_cny_per_million": {k: str(v) for k, v in rates.items()},
                   "gold": sha256(dataset / "gold_standard.csv"),
                   "pdfs": {row["paper_id"]: sha256(paper_path(dataset, row)) for row in all_papers},
                   "code": {name: sha256(ROOT / name) for name in ("housing_enhanced.py", "llm_interface.py")},
                   "pymupdf4llm": __import__("pymupdf4llm").__version__}
    run_id = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    state_path = output / "resume_state.json"
    state = (json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists()
             else {"run_id": run_id, "requests": [], "results": {group: {} for group in GROUPS}})
    if state["run_id"] != run_id:
        raise ValueError("Model, pricing, dataset or extraction settings changed. Use a new --output directory")

    def save():
        write_json(state_path, state)

    meter = Budget(state, save, rates, budget_cny, model)
    if meter.spent > budget_cny:
        raise ValueError("The recorded spend already exceeds this budget")
    if texts is None:
        texts, report = prepare_run(dataset, output, selected, model, prompt_file)
        if report["errors"] or report["truncated"]:
            return report
    try:
        with ExitStack() as stack:
            stack.enter_context(meter.guard())
            autos = {}
            for group, mode in GROUPS.items():
                auto = automation(output / group, dataset, model, api_key, mode,
                                  opt_prompt if group == "G2" else None)
                stack.callback(auto.close)
                autos[group] = auto
            for paper in selected:
                pid = paper["paper_id"]
                for group, auto in autos.items():
                    if pid in state["results"][group]:
                        continue
                    meter.current = group, pid
                    start = time.perf_counter()
                    result = auto.extract_data(texts[pid], paper["pdf_filename"])
                    if meter.reason:
                        raise BudgetStop(meter.reason)
                    ok = not all(result.fields.get(f) == "Error" for f in FIELDS)
                    columns = (*FIELDS, *(f"{name}_quote" for name in FIELDS)) if group == "G2" else FIELDS
                    state["results"][group][pid] = {
                        "run_id": run_id, "paper_id": pid, "status": "ok" if ok else "parse_error",
                        **{name: result.fields.get(name, "") if ok and name in columns else ""
                           for name in OUTPUT_COLUMNS[3:]},
                        "elapsed_seconds": time.perf_counter() - start,
                    }
                    save()
    except BudgetStop:
        pass
    finally:
        selected_ids = [paper["paper_id"] for paper in selected]
        groups = {}
        for group, results in state["results"].items():
            rows = [results[pid] for pid in selected_ids if pid in results]
            write_csv(output / f"{group}_extraction.csv", OUTPUT_COLUMNS,
                      ({k: row[k] for k in OUTPUT_COLUMNS} for row in rows))
            groups[group] = summarize_group(
                rows, [r for r in meter.records if r["group"] == group and r["paper_id"] in selected_ids],
                selected_ids,
            )
        summary = {"run_id": run_id, "model": model, "settings": SETTINGS, "groups": groups,
                   "prompt_file": Path(prompt_file).name,
                   "selected_ids": selected_ids, "dataset_size": len(all_papers),
                   "gold_sha256": fingerprint["gold"], "pdf_sha256": fingerprint["pdfs"],
                   "rates_cny_per_million": fingerprint["rated_cny_per_million"],
                   "budget_cny": float(budget_cny), "reserved_or_spent_peak_cny": float(meter.spent),
                   "stop_reason": meter.reason,
                   "length_limited_requests": sum(r["finish_reason"] == "length" for r in meter.records)}
        write_json(output / "run_summary.json", summary)
    return summary


def paths(parser):
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    paths(parser)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE,
                        help="Prompt file for the optimized group (defaults to v1)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(OFFICIAL_MODELS))
    parser.add_argument("--budget-cny", type=Decimal, default=BUDGET_DEFAULT)
    for name in RATE_NAMES:
        parser.add_argument("--" + name.replace("_", "-") + "-cny-per-million", type=Decimal, dest=name)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    rates = {name: getattr(args, name) for name in RATE_NAMES}
    if not args.dry_run and (not os.getenv("DEEPSEEK_API_KEY") or any(v is None for v in rates.values())):
        parser.error("Set DEEPSEEK_API_KEY and all four rates before a live run")
    report = run(args.dataset, args.output or args.dataset / "evaluation_v1", model=args.model,
                 limit=args.limit, dry_run=args.dry_run, rates=rates, budget_cny=args.budget_cny,
                 prompt_file=args.prompt_file,
                 api_key=os.getenv("DEEPSEEK_API_KEY", "") if not args.dry_run else "")
    if "groups" not in report:
        print(f'PDFs converted={report["converted"]}, errors={report["errors"]}, '
              f'truncated={report["truncated"]}, characters={report["min_chars"]}–{report["max_chars"]}')
        return int(bool(report["errors"] or report["truncated"]))
    print("Run summary written. " + (report["stop_reason"] or "Selected pairs processed."))
    return 2 if report["stop_reason"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

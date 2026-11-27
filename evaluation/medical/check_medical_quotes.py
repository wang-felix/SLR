"""Verify verbatim medical extraction quotes and locate physical PDF pages."""

import argparse
import json
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pymupdf4llm

from evaluation.extraction.extraction_utils import check_excerpt_with_pages, is_missing, wilson
from evaluation.medical.run_medical_eval import (
    FIELDS, GROUPS, indexed, load_gold, paper_path, paths, read_csv, sha256, strip_reference_text,
    write_csv, write_json,
)

_TYPOGRAPHIC_QUOTES = str.maketrans("\u2018\u2019\u201c\u201d", "''\"\"")


def normalize_typographic_quotes(text):
    """Map typographic quotes one-to-one, preserving character offsets."""
    return text.translate(_TYPOGRAPHIC_QUOTES)


def metric(count, denominator):
    return {"count": count, "denominator": denominator,
            "rate": count / denominator if denominator else None,
            "wilson_95": list(wilson(count, denominator)) if denominator else None}


def not_reported(value):
    return str(value or "").strip().casefold() == "not reported"


def has_quote(value):
    return not is_missing(value) and str(value).strip().casefold().replace("_", " ") not in {
        "not found", "no quote",
    }


def load_run(output):
    run = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    ids = run["selected_ids"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Invalid selected IDs in run summary")
    groups = {}
    for group in GROUPS:
        rows = indexed(read_csv(output / f"{group}_extraction.csv"), lambda row: row["paper_id"])
        if set(rows) != set(ids) or any(row["run_id"] != run["run_id"] for row in rows.values()):
            raise ValueError(f"{group} has missing, extra or mismatched rows")
        groups[group] = rows
    return run, groups


def load_pages(pdf):
    chunks = pymupdf4llm.to_markdown(
        str(pdf), page_chunks=True, use_ocr=False, show_progress=False,
    )
    raw = [chunk["text"].rstrip("\n") + "\n" for chunk in chunks]
    body = strip_reference_text("".join(raw))
    if not body.strip():
        raise ValueError(f"No text available for quote validation in {pdf.name}")
    pages = []
    offset = 0
    for number, chunk in enumerate(raw, 1):
        if offset >= len(body):
            break
        part = body[offset:offset + len(chunk)]
        pages.append((number, part))
        offset += len(chunk)
    return body, pages


def quote_metrics(rows):
    present = [row for row in rows if row["has_quote"]]
    abstentions = [row for row in rows if row["abstention"] != "n/a"]
    strict = sum(row["status"] == "pass" for row in present)
    fragmented = sum(row["status"] == "pass_fragmented" for row in present)
    return {
        "quote_coverage": metric(len(present), len(rows)),
        "strict_pass": metric(strict, len(present)),
        "fragmented_pass": metric(fragmented, len(present)),
        "verbatim_match": metric(strict + fragmented, len(present)),
        "physical_page": metric(
            sum(row["status"] in {"pass", "pass_fragmented"} and bool(row["matched_pages"])
                for row in present),
            len(present),
        ),
        "abstention_consistency": metric(
            sum(row["abstention"] == "consistent" for row in abstentions), len(abstentions),
        ),
    }


def check(dataset, output):
    run, groups = load_run(output)
    if sha256(dataset / "gold_standard.csv") != run["gold_sha256"]:
        raise ValueError("Gold table changed since the extraction run")
    gold = indexed(load_gold(dataset), lambda row: row["paper_id"])
    rows = []
    for pid in run["selected_ids"]:
        pdf = paper_path(dataset, gold[pid])
        if sha256(pdf) != run["pdf_sha256"][pid]:
            raise ValueError(f"PDF changed after the extraction run: {pid}")
        text, pages = load_pages(pdf)
        text = normalize_typographic_quotes(text)
        pages = [(number, normalize_typographic_quotes(content)) for number, content in pages]
        for group, predictions in groups.items():
            row = {"run_id": run["run_id"], "paper_id": pid, "group": group}
            for field in FIELDS:
                value = predictions[pid][field]
                quote = predictions[pid][field + "_quote"]
                present = has_quote(quote)
                if present:
                    status, _, _, _, matched = check_excerpt_with_pages(
                        text, normalize_typographic_quotes(quote), pages, near=1.0,
                    )
                else:
                    status, matched = "no_excerpt", ""
                abstention = "n/a"
                if not_reported(value):
                    abstention = "consistent" if not_reported(quote) else "inconsistent"
                elif not is_missing(value) and not_reported(quote):
                    abstention = "inconsistent"
                record = {"status": status, "matched_pages": matched, "abstention": abstention,
                          "has_quote": present}
                row.update({f"{field}_{key}": item for key, item in record.items()})
            rows.append(row)
    write_csv(output / "l1_quotes.csv", list(rows[0]), rows)
    summary = {"run_id": run["run_id"], "selected_ids": run["selected_ids"], "groups": {}}
    for group in GROUPS:
        group_rows = [row for row in rows if row["group"] == group]
        field_records = {
            field: [{name: row[f"{field}_{name}"] for name in (
                "status", "matched_pages", "abstention", "has_quote",
            )} for row in group_rows] for field in FIELDS
        }
        summary["groups"][group] = {
            **quote_metrics([record for records in field_records.values() for record in records]),
            "by_field": {field: quote_metrics(records) for field, records in field_records.items()},
        }
    write_json(output / "l1_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    paths(parser)
    args = parser.parse_args()
    check(args.dataset, args.output or args.dataset / "evaluation_v1")


if __name__ == "__main__":
    main()

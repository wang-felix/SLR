"""Decode completed medical ratings and apply the frozen version-2 thresholds."""

import argparse
import json
import sys
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openpyxl import load_workbook

from atomic_files import atomic_write_text
from evaluation.medical.check_medical_quotes import has_quote, load_run, metric
from evaluation.medical.make_blind_review_sheet import (
    ACCURACY, ACCURACY_HEADERS, ACCURACY_TAB, EVIDENCE, EVIDENCE_HEADERS, EVIDENCE_TAB, SYSTEMS,
)
from evaluation.medical.run_medical_eval import FIELDS, GROUPS, indexed, paths, read_csv

EXPECTED_PAPERS = 20


def read_ratings(workbook, run_id):
    wb = load_workbook(workbook, read_only=True)
    try:
        if wb["评分说明"]["B1"].value != run_id:
            raise ValueError("Blind workbook run ID does not match the extraction run")
        def read_tab(name, headers):
            rows = wb[name].values
            if next(rows, None) != headers:
                raise ValueError(f"Unexpected columns in {name}")
            return indexed(
                (dict(zip(headers, row)) for row in rows if any(value is not None for value in row)),
                lambda row: (row["paper_id"], row["system"], row["field"]),
            )

        return read_tab(ACCURACY_TAB, ACCURACY_HEADERS), read_tab(EVIDENCE_TAB, EVIDENCE_HEADERS)
    finally:
        wb.close()


def decode(output, workbook, run, predictions):
    ratings, evidence_ratings = read_ratings(workbook, run["run_id"])
    key = indexed(read_csv(output / "blind_key.csv"), lambda row: (row["paper_id"], row["system"]))
    expected_key = {(pid, system) for pid in run["selected_ids"] for system in SYSTEMS}
    expected_ratings = {(pid, system, field) for pid, system in expected_key for field in FIELDS}
    if set(key) != expected_key or set(ratings) != expected_ratings:
        raise ValueError("Blind workbook or decoding key has missing or extra rows")
    if any(row["run_id"] != run["run_id"] for row in key.values()):
        raise ValueError("Decoding key belongs to another run")
    for pid in run["selected_ids"]:
        if {key[pid, system]["group"] for system in SYSTEMS} != set(GROUPS):
            raise ValueError(f"Invalid system mapping for {pid}")
    expected_quotes = {
        (pid, system, field) for pid, system, field in expected_ratings
        if has_quote(predictions[key[pid, system]["group"]][pid][field + "_quote"])
    }
    if set(evidence_ratings) != expected_quotes:
        raise ValueError("Evidence worksheet has missing or extra quote rows")

    grouped = {group: [] for group in GROUPS}
    for (pid, system, field), row in ratings.items():
        group = key[pid, system]["group"]
        prediction = predictions[group][pid]
        location = f"{pid} / {system} / {field}"
        if row["accuracy_grade"] not in ACCURACY:
            raise ValueError(f"Fill the accuracy grade using the dropdown for {location}")
        if str(row["prediction"] or "") != prediction[field]:
            raise ValueError(f"Accuracy prediction differs from the extraction for {location}")
        evidence = "No_Quote"
        if (pid, system, field) in expected_quotes:
            evidence_row = evidence_ratings[pid, system, field]
            evidence = evidence_row["evidence_grade"]
            if evidence not in EVIDENCE:
                raise ValueError(f"Fill the evidence grade using the dropdown for {location}")
            if str(evidence_row["prediction"] or "") != prediction[field]:
                raise ValueError(f"Evidence prediction differs from the extraction for {location}")
            if str(evidence_row["quote"] or "") != prediction[field + "_quote"]:
                raise ValueError(f"Evidence quote differs from the extraction for {location}")
        grouped[group].append({**row, "evidence_grade": evidence})
    return grouped


def ratings_summary(rows):
    grades = Counter(row["accuracy_grade"] for row in rows)
    evidence = Counter(row["evidence_grade"] for row in rows)
    eligible = len(rows) - grades["not_in_pdf"]
    with_quote = len(rows) - evidence["No_Quote"]
    return {
        "accuracy": {grade: metric(grades[grade], eligible) for grade in ACCURACY[:-1]},
        "not_in_pdf": grades["not_in_pdf"],
        "evidence": {grade: metric(evidence[grade], with_quote) for grade in EVIDENCE},
        "no_quote": evidence["No_Quote"],
    }


def evaluate_gates(run, l1, scores):
    raw, opt = (scores[group] for group in GROUPS)
    g1, g2 = (run["groups"][group] for group in GROUPS)

    def rate(metrics):
        if metrics["denominator"]:
            return Fraction(metrics["count"], metrics["denominator"])
        return None

    exact1, exact2 = (rate(group["accuracy"]["Exact"]) for group in (raw, opt))
    wrong1, wrong2 = (rate(group["accuracy"]["Wrong"]) for group in (raw, opt))
    quote = rate(l1["groups"]["G2"]["verbatim_match"])
    page = rate(l1["groups"]["G2"]["physical_page"])
    support = rate(opt["evidence"]["Supported"])
    exact_delta = None if exact1 is None or exact2 is None else exact2 - exact1
    wrong_gain = None if wrong1 is None or wrong2 is None else wrong1 - wrong2
    parse1, parse2 = (Fraction(g["parsed"], g["requested"]) for g in (g1, g2))
    more_expensive = Decimal(str(g2["peak_cny"])) > Decimal("2.5") * Decimal(str(g1["peak_cny"]))
    slower = Decimal(str(g2["elapsed_seconds"])) > Decimal("2.5") * Decimal(str(g1["elapsed_seconds"]))
    passed = {
        "Wrong rate improves ≥5 pp, or stays at/below a baseline ≤3%": (
            wrong_gain is not None and (wrong_gain >= Fraction(".05")
                                         or wrong1 <= Fraction(".03") and wrong_gain >= 0)
        ),
        "Exact rate does not decline": exact_delta is not None and exact_delta >= 0,
        "G2 verbatim quote match ≥80%": quote is not None and quote >= Fraction(".80"),
        "G2 physical page match ≥80%": page is not None and page >= Fraction(".80"),
        "G2 human evidence support ≥85%": support is not None and support >= Fraction(".85"),
    }
    redlines = {
        "G2 Exact rate declines or parsing failures exceed 10%": (
            (exact_delta is not None and exact_delta < 0) or parse2 < Fraction(".90")
        ),
        "G2 verbatim quote match ≥80%, but human evidence support <65%": (
            quote is not None and support is not None and quote >= Fraction(".80") and support < Fraction(".65")
        ),
        "G2 peak cost or time exceeds 2.5× and both improvements are <3 pp": (
            (more_expensive or slower) and exact_delta is not None and wrong_gain is not None
            and exact_delta < Fraction(".03") and wrong_gain < Fraction(".03")
        ),
    }
    operational = {
        "Both parse success rates ≥95%": min(parse1, parse2) >= Fraction(".95"),
        "Peak reservation/spend ≤¥8 without a budget stop": (
            Decimal(str(run["reserved_or_spent_peak_cny"])) <= Decimal("8") and not run["stop_reason"]
        ),
        "Every request has complete usage metadata": all(g["usage_complete"] for g in (g1, g2)),
    }
    if len(run["selected_ids"]) != EXPECTED_PAPERS or run["dataset_size"] != EXPECTED_PAPERS:
        verdict = "PROVISIONAL"
    elif any(redlines.values()):
        verdict = "STOP"
    elif all(passed.values()) and all(operational.values()):
        verdict = "PASS"
    else:
        verdict = "NOT PASSED"
    return verdict, passed, redlines, operational


def format_rate(stat):
    if stat["rate"] is None:
        return "N/A (0 eligible)"
    low, high = stat["wilson_95"]
    return (f'{stat["count"]}/{stat["denominator"]} ({stat["rate"]:.1%}) '
            f'[{low:.1%}, {high:.1%}]')


def report_text(run, l1, scores, by_field, outcome):
    verdict, passed, redlines, operational = outcome
    lines = [
        f'# Medical paired extraction evaluation — {verdict}', '',
        f'Model {run["model"]} · {len(run["selected_ids"])} paired papers · {run["run_id"]}', '',
        'G1 uses plain JSON with the default prompt. G2 uses the clinical prompt, quote-first '
        'instructions and structured output when supported.',
        'Intervals are 95% Wilson intervals over field observations. Fields from the same paper '
        'may be correlated, so these intervals are descriptive.',
        'Accuracy excludes not_in_pdf. Quote match and support use nonempty, non-abstention quotes. '
        'G1 normally has no quote, so quote-specific rates may be undefined.', '',
        f'Peak-price reserved or spent: ¥{run["reserved_or_spent_peak_cny"]:.4f}. '
        f'Requests reaching the output limit: {run["length_limited_requests"]}.', '',
        '| Metric (95% interval) | G1 | G2 |', '|---|---|---|',
    ]

    def pair(label, values):
        lines.append(f'| {label} | {format_rate(values[0])} | {format_rate(values[1])} |')

    for grade in ACCURACY[:-1]:
        label = f"Accuracy · {grade}" + (" (severe error / hallucination)" if grade == "Wrong" else "")
        pair(label, [scores[g]["accuracy"][grade] for g in GROUPS])
    for key, label in (
        ("quote_coverage", "L1 · quote coverage"),
        ("strict_pass", "L1 · strict pass"),
        ("fragmented_pass", "L1 · fragmented pass"),
        ("verbatim_match", "L1 · verbatim match"),
        ("physical_page", "L1 · physical PDF page"),
        ("abstention_consistency", "L1 · abstention consistency"),
    ):
        pair(label, [l1["groups"][g][key] for g in GROUPS])
    for grade in EVIDENCE:
        pair(f"L2 · {grade}", [scores[g]["evidence"][grade] for g in GROUPS])
    for key, label in (("not_in_pdf", "Excluded · not_in_pdf"), ("no_quote", "L2 · No_Quote")):
        lines.append(f'| {label} | {scores["G1"][key]} | {scores["G2"][key]} |')
    pair("Parsed successfully", [metric(run["groups"][g]["parsed"], run["groups"][g]["requested"])
                                 for g in GROUPS])

    lines.extend(["", "## Field accuracy", "", "| Field and grade | G1 | G2 |", "|---|---|---|"])
    for field in FIELDS:
        for grade in ACCURACY[:-1]:
            pair(f"{field} · {grade}", [by_field[g][field]["accuracy"][grade] for g in GROUPS])

    lines.extend(["", "## Per 100 papers", "",
                  "| Group | Input tokens | Output tokens | Total tokens | Extraction seconds | Off-peak CNY ceiling | Peak CNY ceiling |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for group, data in run["groups"].items():
        per = data["per_100"]
        if per is None:
            lines.append(f"| {group} | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        tokens = [f'{per[k]:,.0f}' if data["usage_complete"] else "N/A" for k in (
            "prompt_tokens", "completion_tokens", "total_tokens"
        )]
        lines.append(f'| {group} | {" | ".join(tokens)} | {per["elapsed_seconds"]:.2f} | '
                     f'¥{per["offpeak_cny"]:.4f} | ¥{per["peak_cny"]:.4f} |')
    lines.extend([
        "", "Costs treat every input token as a cache miss. Unsettled requests keep their reserved "
        "ceilings. Timing includes extraction retries and excludes shared PDF conversion.", "",
        "## Version 2 decision rules", "",
    ])
    for name, result in {**passed, **operational}.items():
        lines.append(f'- {"PASS" if result else "NOT MET"} · {name}')
    for name, result in redlines.items():
        lines.append(f'- {"TRIGGERED" if result else "NOT TRIGGERED"} · {name}')
    if verdict == "PROVISIONAL":
        lines.append("\nFormal verdict requires all 20 paired papers.")
    return "\n".join(lines) + "\n"


def score(output, workbook):
    run, predictions = load_run(output)
    l1 = json.loads((output / "l1_summary.json").read_text(encoding="utf-8"))
    if l1["run_id"] != run["run_id"] or l1["selected_ids"] != run["selected_ids"]:
        raise ValueError("L1 summary belongs to another run")
    rows = decode(output, workbook, run, predictions)
    summaries = {group: ratings_summary(items) for group, items in rows.items()}
    for group, data in summaries.items():
        count = data["evidence"]["Supported"]["denominator"]
        if l1["groups"][group]["quote_coverage"]["count"] != count:
            raise ValueError("L1 and human review disagree on quote coverage")
    by_field = {
        group: {field: ratings_summary([row for row in items if row["field"] == field])
                for field in FIELDS}
        for group, items in rows.items()
    }
    outcome = evaluate_gates(run, l1, summaries)
    atomic_write_text(output / "medical_eval_report.md", report_text(run, l1, summaries, by_field, outcome))
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    paths(parser)
    parser.add_argument("--workbook", type=Path)
    args = parser.parse_args()
    output = args.output or args.dataset / "evaluation_v1"
    print(score(output, args.workbook or output / "blind_review_sheet.xlsx")[0])


if __name__ == "__main__":
    main()

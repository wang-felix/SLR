"""Make the offline medical grading workbook and its separate decoding key."""

import argparse
import random
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from atomic_files import atomic_output
from evaluation.medical.check_medical_quotes import has_quote, load_run
from evaluation.medical.run_medical_eval import (
    FIELDS, GOLD, GROUPS, indexed, load_gold, paths, read_csv, sha256, write_csv,
)
from spreadsheet_safety import literal_workbook

SYSTEMS = ("SYS_A", "SYS_B")
ACCURACY = ("Exact", "Partial", "Missed", "Wrong", "not_in_pdf")
EVIDENCE = ("Supported", "Weak", "Unsupported")
ACCURACY_TAB = "1_Accuracy_Blind"
EVIDENCE_TAB = "2_Evidence_Support"
ACCURACY_HEADERS = ("paper_id", "system", "field", "expert_value", "prediction",
                    "accuracy_grade", "reviewer_note")
EVIDENCE_HEADERS = ("paper_id", "system", "field", "prediction", "quote", "matched_pages",
                    "evidence_grade", "reviewer_note")
KEY_COLUMNS = ("run_id", "paper_id", "system", "group")
GUIDE = [
    ("用途", "先完成 1_Accuracy_Blind 的全部准确度评分。随后右键单击工作表标签，选择「取消隐藏」，打开 2_Evidence_Support 评引句。"),
    ("页码", "matched_pages 是 PDF 查看器从 1 开始的物理页码，可能与论文印刷页码不同。"),
    ("Exact", "专家表中可由该 PDF 核实的核心要素均准确，允许同义表达及明确口径。"),
    ("Partial", "方向正确，但遗漏部分可核实要素或有轻微细节误差。"),
    ("Missed", "PDF 报告了该字段，输出为空、Not reported 或抽取失败。"),
    ("Wrong", "严重错误，包括试验组别张冠李戴、错误数值、错误结局或无原文依据的事实。"),
    ("not_in_pdf", "专家表对应信息在这份 PDF 不可核实。仅在预测没有虚构事实时使用，并备注原因。该行排除出准确度分母。"),
    ("Supported", "引句直接支持所填字段的核心事实。"),
    ("Weak", "引句只支持部分要素，或上下文不足以肯定字段值。"),
    ("Unsupported", "引句无法定位、与字段无关，或在语义上与字段值矛盾。"),
    ("No_Quote", "解盲时自动标记没有实质引句的字段。"),
    ("结局字段", "只评结局定义、主次和测量时间。效应方向、效应值和 P 值不进入金标准。"),
    ("P003", "存在 240 人随机化与排除 17 人后 223 人纳入的双口径，按原文标识对应阶段。"),
    ("P009", "专家表干预字段括号引用误写 Bermon 2021。以当前 Bae 2021 的原始 PDF 为准。"),
    ("P017", "专家表使用 Gladys 原写法，遇到疑义请核查原始 PDF 并备注。"),
    ("盲评", "每篇独立分配 SYS_A 和 SYS_B。请在完成评分前保持 blind_key.csv 分离。"),
]


def create(dataset, output, seed):
    run, groups = load_run(output)
    if sha256(dataset / "gold_standard.csv") != run["gold_sha256"]:
        raise ValueError("Gold table changed since the extraction run")
    gold = indexed(load_gold(dataset), lambda row: row["paper_id"])
    l1 = indexed(read_csv(output / "l1_quotes.csv"), lambda row: (row["paper_id"], row["group"]))
    expected = {(pid, group) for pid in run["selected_ids"] for group in GROUPS}
    if set(l1) != expected or any(row["run_id"] != run["run_id"] for row in l1.values()):
        raise ValueError("L1 results do not match this paired run")
    target = output / "blind_review_sheet.xlsx"
    key_path = output / "blind_key.csv"
    if target.exists() or key_path.exists():
        raise FileExistsError("Blind workbook or key already exists; preserve the existing review before regenerating")

    wb = Workbook()
    accuracy = wb.active
    accuracy.title = ACCURACY_TAB
    evidence = wb.create_sheet(EVIDENCE_TAB)
    evidence.sheet_state = "hidden"
    guide = wb.create_sheet("评分说明")
    guide.append(["run_id", run["run_id"]])
    for line in GUIDE:
        guide.append(line)

    key = []
    accuracy.append(ACCURACY_HEADERS)
    evidence.append(EVIDENCE_HEADERS)
    for pid in run["selected_ids"]:
        order = list(GROUPS)
        random.Random(f"{seed}/{pid}").shuffle(order)
        for system, group in zip(SYSTEMS, order):
            key.append(dict(zip(KEY_COLUMNS, (run["run_id"], pid, system, group))))
            prediction = groups[group][pid]
            for field in FIELDS:
                value = prediction[field]
                quote = prediction[field + "_quote"]
                accuracy.append([pid, system, field, gold[pid][GOLD[field]], value, None, None])
                if has_quote(quote):
                    evidence.append([pid, system, field, value, quote,
                                     l1[pid, group][field + "_matched_pages"], None, None])
    for tab, grades, column in ((accuracy, ACCURACY, "F"), (evidence, EVIDENCE, "G")):
        if tab.max_row == 1:
            continue
        validation = DataValidation(type="list", formula1='"' + ",".join(grades) + '"',
                                    allow_blank=True, showErrorMessage=True)
        validation.error = "Choose a valid grade from the dropdown."
        tab.add_data_validation(validation)
        validation.add(f"{column}2:{column}{tab.max_row}")
    for tab, widths, grade_column in (
        (accuracy, (12, 12, 30, 58, 52, 19, 35), 6),
        (evidence, (12, 12, 30, 52, 52, 15, 19, 35), 7),
        (guide, (20, 98), None),
    ):
        for index, width in enumerate(widths, 1):
            tab.column_dimensions[get_column_letter(index)].width = width
        for row in tab.iter_rows():
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                if cell.row == 1:
                    cell.fill = PatternFill("solid", fgColor="1F3864")
                    cell.font = Font(bold=True, color="FFFFFF")
                elif grade_column and cell.column >= grade_column:
                    cell.fill = PatternFill("solid", fgColor="E8F2FE")
        tab.freeze_panes = "D2" if grade_column else "A2"
        if grade_column:
            tab.auto_filter.ref = tab.dimensions
        for row in range(2, tab.max_row + 1):
            tab.row_dimensions[row].height = 88 if grade_column else 34
    literal_workbook(wb)
    try:
        with atomic_output(target) as temporary:
            wb.save(temporary)
        write_csv(key_path, KEY_COLUMNS, key)
    finally:
        wb.close()
    return accuracy.max_row - 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    paths(parser)
    parser.add_argument("--seed", type=int, default=20261126)
    args = parser.parse_args()
    print("Review rows:", create(args.dataset, args.output or args.dataset / "evaluation_v1", args.seed))


if __name__ == "__main__":
    main()

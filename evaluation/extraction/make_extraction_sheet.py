"""Create a blind human-labeling sheet for extraction evaluation.

Usage: python make_extraction_sheet.py extraction_results.xlsx [--n 20] [--seed 20261116]
Outputs: extraction_label_sheet.xlsx (for the human) and extraction_key.csv (AI answers; keep, needed by evaluate_extraction.py).
The AI's extracted values are NOT shown on the sheet: the human reads the paper and fills in their own answer.
"""
import argparse
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation
from extraction_utils import read_table, pick_col, METHOD_CHOICES

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extraction"); ap.add_argument("--n", type=int, default=20); ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--col-material", default="coating_material"); ap.add_argument("--col-method", default="preparation_method")
    ap.add_argument("--col-excerpt", default="source_section_table_or_text_excerpt"); ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    df = read_table(a.extraction)
    f = pick_col(df, "filename", ["filename"], "filename")
    m = pick_col(df, a.col_material, ["coating_material"], "coating material")
    p = pick_col(df, a.col_method, ["preparation_method"], "preparation method")
    e = pick_col(df, a.col_excerpt, ["excerpt", "source"], "source excerpt")
    df = df.drop_duplicates(f).reset_index(drop=True)
    s = df.sample(n=min(a.n, len(df)), random_state=a.seed).reset_index(drop=True)
    s.insert(0, "id", range(1, len(s) + 1))
    s[["id", f, m, p, e]].rename(columns={f: "filename", m: "ai_material", p: "ai_method", e: "ai_excerpt"}).to_csv(
        f"{a.prefix}extraction_key.csv", index=False, encoding="utf-8-sig")

    wb = Workbook(); ws = wb.active; ws.title = "标注表"
    heads = ["序号", "文件名", "该论文应纳入吗", "人工_涂层材料", "人工_制备方法类别", "备注"]
    ws.append(heads)
    for _, r in s.iterrows():
        ws.append([int(r["id"]), r[f], None, None, None, None])
    for c in ws[1]:
        c.font = Font(name="Arial", bold=True); c.fill = PatternFill("solid", fgColor="D9E1F2")
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for row in ws.iter_rows(min_row=2):
        for c in row: c.font = Font(name="Arial")
    for col, w in zip("ABCDEF", [7, 40, 16, 34, 30, 44]): ws.column_dimensions[col].width = w
    ws.row_dimensions[1].height = 32; ws.freeze_panes = "C2"
    last = len(s) + 1
    lists = wb.create_sheet("Lists")
    for i, c in enumerate(METHOD_CHOICES, 1): lists.cell(row=i, column=1, value=c)
    dv1 = DataValidation(type="list", formula1='"是,否"', allow_blank=True); ws.add_data_validation(dv1); dv1.add(f"C2:C{last}")
    dv2 = DataValidation(type="list", formula1=f"=Lists!$A$1:$A${len(METHOD_CHOICES)}", allow_blank=True); ws.add_data_validation(dv2); dv2.add(f"E2:E{last}")
    fill = PatternFill("solid", fgColor="FFF2CC")
    for row in ws.iter_rows(min_row=2, min_col=3, max_col=6):
        for c in row: c.fill = fill

    lg = wb.create_sheet("说明", 1)
    rows = [["怎么填(这张表不显示 AI 的答案,请读论文后独立填写)"],
            ["1. 该论文应纳入吗:按你的纳入标准判断,选「否」的论文不参与抽取评测。"],
            ["2. 人工_涂层材料:写论文自己制备的涂层材料,用化学式或常用简称,多个组分用 - 或 / 分隔,含量可省略。例:Al2O3-TiO2、TiSiN、MgO。论文没写填「未说明」。"],
            ["3. 人工_制备方法类别:从下拉列表选;用了两种及以上主要工艺选「复合工艺」;列表里没有的选「其他」并在备注写明;没描述选「未说明」。"],
            ["4. 备注:写依据所在位置(如 §2.1)或你的疑问。"], [],
            ["示例行(仅作格式参考)", "文件名", "是", "Al2O3-TiO2", "大气等离子喷涂(APS)", "§2.2 喷涂参数表"]]
    for r in rows: lg.append(r)
    lg["A1"].font = Font(name="Arial", bold=True)
    for row in lg.iter_rows(min_row=2):
        for c in row: c.font = Font(name="Arial")
    lg.column_dimensions["A"].width = 100
    wb.save(f"{a.prefix}extraction_label_sheet.xlsx")
    print(f"wrote {a.prefix}extraction_label_sheet.xlsx ({len(s)} papers) and {a.prefix}extraction_key.csv")

if __name__ == "__main__":
    main()

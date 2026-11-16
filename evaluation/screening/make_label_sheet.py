"""Build a blind, stratified labeling sheet from SLR Assistant screening results.

Baseline sample (what you already labeled):
  python make_label_sheet.py screening_results.xlsx
Independent validation sample (papers NOT in the baseline sample):
  python make_label_sheet.py screening_results.xlsx --exclude-key sample_key.csv --prefix val_

Outputs: <prefix>label_sheet.xlsx (give to the human labeler) and <prefix>sample_key.csv (keep; needed by evaluate.py).
The AI decision is deliberately NOT shown in the label sheet, to avoid anchoring the labeler.
Weights are stratum size / number sampled, computed on the pool that was sampled from. For the validation sample
that pool is the papers left after removing the baseline sample, so its estimates describe those remaining papers.
"""
import argparse
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation

BORDER_PAT = r"金属陶瓷|高熵|并非纯|而非纯|不是纯|建议人工|建议复核|虽题为|cermet"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--n-excl", type=int, default=None)
    ap.add_argument("--n-border", type=int, default=None)
    ap.add_argument("--n-clear", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--exclude-key", default=None, help="sample_key.csv of an earlier sample; those papers are skipped")
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    validation = a.exclude_key is not None
    n_excl = a.n_excl if a.n_excl is not None else (6 if validation else 15)
    n_border = a.n_border if a.n_border is not None else (4 if validation else 20)
    n_clear = a.n_clear if a.n_clear is not None else (12 if validation else 10)

    df = pd.read_excel(a.results)
    n_all = len(df)
    # Duplicate PDFs: identical extracted-text length AND identical reasoning (cache hit on same text)
    df = df.drop_duplicates(subset=["Text Length", "Reasoning"], keep="first").reset_index(drop=True)
    print(f"rows: {n_all} -> {len(df)} unique after removing duplicate PDFs")

    text = df["Reasoning"].fillna("") + " " + df["Notes"].fillna("")
    is_excl = df["Decision"].eq("Likely Exclude")
    is_border = ~is_excl & text.str.contains(BORDER_PAT, regex=True)
    df["stratum"] = "C_include_clear"
    df.loc[is_border, "stratum"] = "B_include_borderline"
    df.loc[is_excl, "stratum"] = "A_exclude"

    if validation:
        used = set(pd.read_csv(a.exclude_key)["filename"])
        before = len(df)
        df = df[~df["Filename"].isin(used)].reset_index(drop=True)
        print(f"validation pool: {before} -> {len(df)} papers after removing {before - len(df)} already sampled")

    want = {"A_exclude": n_excl, "B_include_borderline": n_border, "C_include_clear": n_clear}
    parts = []
    for s, k in want.items():
        g = df[df.stratum == s]
        take = g.sample(n=min(k, len(g)), random_state=a.seed)
        take = take.assign(N_stratum=len(g), n_sampled=len(take), weight=len(g) / max(len(take), 1))
        parts.append(take)
        print(f"{s}: population {len(g)}, sampled {len(take)}")
    key = pd.concat(parts).sample(frac=1, random_state=a.seed).reset_index(drop=True)  # shuffle: hide strata
    key.insert(0, "id", range(1, len(key) + 1))
    key[["id", "Filename", "Decision", "stratum", "N_stratum", "n_sampled", "weight", "Text Length"]].rename(
        columns={"Filename": "filename", "Decision": "ai_decision", "Text Length": "text_length"}).to_csv(
        f"{a.prefix}sample_key.csv", index=False, encoding="utf-8-sig")

    wb = Workbook()
    ws = wb.active; ws.title = "标注表"
    heads = ["序号", "文件名", "Q1 是综述吗", "Q2 作者自己制备了涂层并写了制备方法",
             "Q3 涂层连续相为陶瓷(非金属基)", "Q4 用于耐磨或耐腐蚀研究", "最终人工判断", "备注"]
    ws.append(heads)
    for _, r in key.iterrows():
        ws.append([int(r["id"]), r["Filename"], None, None, None, None, None, None])
    hdr_fill = PatternFill("solid", fgColor="D9E1F2")
    for c in ws[1]:
        c.font = Font(name="Arial", bold=True); c.fill = hdr_fill
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for row in ws.iter_rows(min_row=2):
        for c in row: c.font = Font(name="Arial")
    for col, w in zip("ABCDEFGH", [7, 40, 14, 24, 24, 20, 16, 44]): ws.column_dimensions[col].width = w
    ws.row_dimensions[1].height = 48
    ws.freeze_panes = "C2"
    last = len(key) + 1
    for col, opts in {"C": '"是,否"', "D": '"是,否"', "E": '"是,否,拿不准"', "F": '"是,否,拿不准"',
                      "G": '"纳入,排除,拿不准"'}.items():
        dv = DataValidation(type="list", formula1=opts, allow_blank=True); ws.add_data_validation(dv); dv.add(f"{col}2:{col}{last}")
    fill_in = PatternFill("solid", fgColor="FFF2CC")
    for row in ws.iter_rows(min_row=2, min_col=3, max_col=8):
        for c in row: c.fill = fill_in

    lg = wb.create_sheet("说明")
    rows = [
        ["怎么填"],
        ["1. 只填「标注表」里黄色的格子(C~H 列),按文件名找到对应 PDF,看标题、摘要和实验部分即可,不必通读全文。"],
        ["2. 这张表故意没有显示 AI 的判断,请独立判断,不要事先去看 AI 的结果。"],
        ["3. 最终人工判断:同时满足 Q1=否、Q2=是、Q3=是、Q4=是 时填「纳入」;否则填「排除」;确实无法判断填「拿不准」(不要把难判的论文都填拿不准)。"],
        ["4. Q3 看「连续相」,即把整个涂层连在一起的那一相(基体或粘结相):是陶瓷(氧化物、氮化物、碳化物膜层)填「是」;"
         "是金属或合金(WC-Co、Cr3C2-NiCr 等金属陶瓷、熔覆层、金属基陶瓷增强复合涂层)填「否」,即使陶瓷颗粒含量很高,备注写「金属基」。"],
        ["5. 陶瓷涂层中少量掺入 Ag、石墨烯等添加相,仍算陶瓷(Q3=是)。"],
        ["6. 喷涂/制备使用公司或外单位设备,只要论文描述了本研究样品的制备参数,Q2 仍填「是」,备注写「外协」。"],
        ["7. 备注请写依据所在位置(如 §2.1)和一句理由;用途不符写「用途不符」。"],
        [],
        ["示例行(仅作格式参考,不要抄)", "文件名", "Q1", "Q2", "Q3", "Q4", "最终人工判断", "备注"],
        ["示例", "xxxx.pdf", "否", "是", "是", "是", "纳入", "§2.1 微弧氧化 Al2O3 膜,方法部分有电解液参数"],
    ]
    for r in rows: lg.append(r)
    lg["A1"].font = Font(name="Arial", bold=True)
    for row in lg.iter_rows(min_row=2):
        for c in row:
            if c.row != 10: c.font = Font(name="Arial")
    for c in lg[10]: c.font = Font(name="Arial", bold=True)
    lg.column_dimensions["A"].width = 30; lg.column_dimensions["B"].width = 16; lg.column_dimensions["H"].width = 44
    wb.save(f"{a.prefix}label_sheet.xlsx")
    print(f"wrote {a.prefix}label_sheet.xlsx ({len(key)} papers to label) and {a.prefix}sample_key.csv")

if __name__ == "__main__":
    main()

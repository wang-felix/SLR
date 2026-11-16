"""Evaluate extraction (coating material, preparation method, source excerpt) against human answers.

Usage: python evaluate_extraction.py extraction_label_sheet.xlsx extraction_key.csv [--excerpt-check excerpt_check.csv] [--out report_extraction.md]
- coating material: both sides are reduced to a set of normalised components; exact = identical sets, partial = overlap, none = no overlap.
- preparation method: AI free text is mapped to fixed categories by keyword rules (extraction_utils.METHOD_RULES) and compared with the human's category.
- source excerpt: summarised from the automatic check (check_excerpts.py), which needs no human labels.
All proportions carry 95% Wilson intervals (valid even with 0 errors). Only papers the human marked 「应纳入=是」 are scored.
"""
import argparse
import pandas as pd
from extraction_utils import is_missing, wilson, score_material, classify_method

def pct(k, n):
    if n == 0: return "n/a"
    lo, hi = wilson(k, n); return f"{k}/{n} = {k/n:.1%} (95% CI {lo:.1%} ~ {hi:.1%})"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels"); ap.add_argument("key")
    ap.add_argument("--excerpt-check", default=None); ap.add_argument("--out", default="report_extraction.md")
    a = ap.parse_args()
    lab = pd.read_excel(a.labels, sheet_name="标注表")
    key = pd.read_csv(a.key)
    d = key.merge(lab[["序号", "该论文应纳入吗", "人工_涂层材料", "人工_制备方法类别", "备注"]], left_on="id", right_on="序号", how="left")
    n_all, n_unfilled = len(d), int(d["该论文应纳入吗"].isna().sum())
    elig = d[d["该论文应纳入吗"] == "是"].copy()
    L = ["# 抽取评测报告", "", f"- 抽样 {n_all} 篇;未填「应纳入吗」{n_unfilled} 篇;人工认为不应纳入 {int((d['该论文应纳入吗'] == '否').sum())} 篇(不计分);参与评分 {len(elig)} 篇", ""]

    # --- material ---
    mrows = []
    for _, r in elig.iterrows():
        if pd.isna(r["人工_涂层材料"]): continue
        sc, jac, A, H = score_material(r["ai_material"], r["人工_涂层材料"])
        mrows.append(dict(id=r["id"], ai_raw=r["ai_material"], human_raw=r["人工_涂层材料"], ai_set=" | ".join(sorted(A)),
                          human_set=" | ".join(sorted(H)), score=sc, jaccard=round(jac, 2), note=r["备注"]))
    m = pd.DataFrame(mrows)
    L += ["## 1. 涂层材料", ""]
    if len(m):
        n = len(m); ex = int((m.score == "exact").sum()); pa = int((m.score == "partial").sum())
        L += [f"- 完全一致:{pct(ex, n)}", f"- 完全或部分一致(有重叠组分):{pct(ex + pa, n)}", f"- 平均重叠度(Jaccard):{m.jaccard.mean():.2f}",
              "- 逐篇对照见 material_comparison.csv。归一化只处理写法差异(下标、中文名、含量),合金写法(NiCr 与 Ni-Cr)等仍可能误判,请抽看 partial/none 行,必要时修改人工答案的写法后重跑。", ""]
        m.to_csv("material_comparison.csv", index=False, encoding="utf-8-sig")
    else:
        L += ["没有可评分的行。", ""]

    # --- method ---
    prow = []
    for _, r in elig.iterrows():
        if pd.isna(r["人工_制备方法类别"]): continue
        cat, hits = classify_method(r["ai_method"])
        unrec = (cat == "其他" and not is_missing(r["ai_method"]))
        prow.append(dict(id=r["id"], ai_raw=str(r["ai_method"])[:120], ai_category=cat, human_category=r["人工_制备方法类别"],
                         correct=int(cat == r["人工_制备方法类别"]), ai_text_unrecognized_by_rules=int(unrec), note=r["备注"]))
    p = pd.DataFrame(prow)
    L += ["## 2. 制备方法(归类后比较)", ""]
    if len(p):
        n = len(p); k = int(p.correct.sum()); u = int(p.ai_text_unrecognized_by_rules.sum())
        L += [f"- 类别准确率:{pct(k, n)}", f"- 其中 AI 文本未被关键词规则识别、被归为「其他」的:{u} 篇(这类先检查是规则不全还是 AI 答错,见 method_comparison.csv)", ""]
        wrong = p[p.correct == 0]
        if len(wrong):
            L += ["不一致的情况:", "", "| 序号 | AI 归类 | 人工类别 | AI 原文(截取) |", "|---|---|---|---|"]
            for _, r in wrong.iterrows():
                L.append(f"| {int(r.id)} | {r.ai_category} | {r.human_category} | {r.ai_raw} |")
            L.append("")
        p.to_csv("method_comparison.csv", index=False, encoding="utf-8-sig")
    else:
        L += ["没有可评分的行。", ""]

    # --- excerpt ---
    L += ["## 3. 原文出处(自动校验,无需人工)", ""]
    if a.excerpt_check:
        e = pd.read_csv(a.excerpt_check)
        vc = e.status.value_counts().to_dict()
        chk = e[e.status.isin(["pass", "partial", "fail"])]; n = len(chk)
        L += [f"- 状态分布:{vc}"]
        if n:
            L += [f"- 完全核实:{pct(int((chk.status == 'pass').sum()), n)}",
                  f"- 完全找不到(疑似编造):{pct(int((chk.status == 'fail').sum()), n)}",
                  f"- 未完全核实(部分或找不到):{pct(int((chk.status != 'pass').sum()), n)}"]
        L += ["- language_mismatch 表示引句语言与论文不同,模型多半是翻译而非逐字引用;pdf_missing 表示没找到对应 PDF,不计入。",
              "- 「找不到」也可能来自 PDF 转文字的差异(表格、公式、连字符),请先看 excerpt_check.csv 的 first_unverified_segment 再下结论。", ""]
    else:
        L += ["未提供 --excerpt-check,先运行 check_excerpts.py 生成 excerpt_check.csv。", ""]
    L += ["## 解读提示", "", "样本小,区间会很宽;结论应连同区间一起汇报。人工答案由一人完成,没有标注者之间的一致性检验。"]
    open(a.out, "w", encoding="utf-8").write("\n".join(L)); print("\n".join(L))

if __name__ == "__main__":
    main()

"""Evaluate extraction (coating material, preparation method, source excerpt) against human answers.

Usage: python evaluate_extraction.py extraction_label_sheet.xlsx extraction_key.csv [--excerpt-check excerpt_check.csv] [--out report_extraction.md]
- coating material: both sides are reduced to a set of normalised components; exact = identical sets, partial = overlap, none = no overlap. Ambiguous rows are reported as review.
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
        if pd.isna(r["人工_涂层材料"]):
            continue
        sc, jac, ai_components, reference = score_material(r["ai_material"], r["人工_涂层材料"])
        review_reason = ""
        if sc == "review":
            review_reason = (
                "参考标注缺少可核对的具体组分" if not reference
                else "AI 答案的组分或层归属需要人工确认"
            )
        mrows.append({
            "id": r["id"],
            "ai_raw": r["ai_material"],
            "human_raw": r["人工_涂层材料"],
            "ai_set": " | ".join(sorted(ai_components)),
            "human_set": " | ".join(sorted(reference)),
            "matched": " | ".join(sorted(ai_components & reference)) if not review_reason else "",
            "missing": " | ".join(sorted(reference - ai_components)) if not review_reason else "",
            "extra": " | ".join(sorted(ai_components - reference)) if not review_reason else "",
            "score": sc,
            "jaccard": jac,
            "review_reason": review_reason,
            "note": r["备注"],
        })
    m = pd.DataFrame(mrows)
    L += ["## 1. 涂层材料", ""]
    if not m.empty:
        scored = m[m.score != "review"]
        n = len(scored)
        L.append(f"- 有材料标注 {len(m)} 篇，自动评分 {n} 篇，待复核 {len(m) - n} 篇。")
        if n:
            exact = int((scored.score == "exact").sum())
            partial = int((scored.score == "partial").sum())
            L += [
                f"- 完全一致：{pct(exact, n)}",
                f"- 完全或部分一致（有重叠组分）：{pct(exact + partial, n)}",
                f"- 平均组分重叠度（Jaccard）：{scored.jaccard.mean():.2f}",
            ]
        L += ["- 逐篇匹配、漏报、多报及待复核原因见 material_comparison.csv。", ""]
        m.round({"jaccard": 2}).to_csv("material_comparison.csv", index=False, encoding="utf-8-sig")
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
                  f"- 未在 PDF 文字中找到:{pct(int((chk.status == 'fail').sum()), n)}",
                  f"- 未完全核实(部分或找不到):{pct(int((chk.status != 'pass').sum()), n)}"]
        else:
            L += ["- 本次没有可计分的引句，匹配比例不适用。"]
        L += ["- 匹配比例仅统计 pass、partial、fail。pass_fragmented 表示引句分段按序匹配，单独计数。",
              "- not_found 表示字段声明找不到可引用句子，location_only 表示字段只有位置说明，两者单独计数。",
              "- language_mismatch 表示引句语言与论文不同，pdf_missing 表示没找到对应 PDF。",
              "- 未匹配可能来自 PDF 转文字的差异（表格、公式、连字符），也可能是字段内容并非引句。请结合 excerpt_check.csv 的 first_unverified_segment 与 PDF 原文复核。", ""]
    else:
        L += ["未提供 --excerpt-check,先运行 check_excerpts.py 生成 excerpt_check.csv。", ""]
    L += ["## 解读提示", "", "样本小,区间会很宽;结论应连同区间一起汇报。人工答案由一人完成,没有标注者之间的一致性检验。"]
    open(a.out, "w", encoding="utf-8").write("\n".join(L)); print("\n".join(L))

if __name__ == "__main__":
    main()

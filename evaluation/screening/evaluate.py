"""Evaluate AI screening against human labels collected on a stratified sample.

Usage:
  python evaluate.py label_sheet.xlsx sample_key.csv --out report_baseline.md
  python evaluate.py label_sheet.xlsx sample_key.csv --results new_screening_results.xlsx --out report_v2.md

Positive class = 纳入 (include). "Flag for Human Review" counts as positive (the paper goes to a human, it is not lost).
Rows the human marked 拿不准 are reported separately and excluded from the metrics.
Because borderline papers were deliberately oversampled, every metric is re-weighted by stratum
(weight = stratum size / number sampled) so it estimates performance on the WHOLE corpus.
Strata and weights are always taken from sample_key.csv (the baseline run), even when --results is given.
95% CIs come from a stratified bootstrap. The bootstrap collapses to a zero-width interval when a metric has
no errors in the sample; in that case a one-sided upper bound (see section 'miss-rate bound') is reported instead.
--results: a new screening_results.xlsx/csv (columns Filename, Decision; Text Length is used if file names changed). It replaces the AI decisions, and the report
adds a before/after comparison against the baseline decisions stored in sample_key.csv.
"""
import argparse
import numpy as np, pandas as pd

def ai_to_num(x):
    s = str(x).lower()
    if "exclude" in s: return 0
    if "include" in s or "flag" in s: return 1
    return np.nan

def load(label_path, key_path, results_path=None):
    lab = pd.read_excel(label_path, sheet_name="标注表")
    key = pd.read_csv(key_path)
    d = key.merge(lab[["序号", "最终人工判断", "备注"]], left_on="id", right_on="序号", how="left")
    d["ai_baseline"] = d["ai_decision"]
    if results_path:
        r = pd.read_csv(results_path) if results_path.lower().endswith(".csv") else pd.read_excel(results_path)
        if "Reasoning" not in r.columns: r["Reasoning"] = None
        by_name = r.drop_duplicates("Filename").set_index("Filename")
        missing = [f for f in d["filename"] if f not in by_name.index]
        if not missing:
            hit = by_name.loc[d["filename"]]
            how = "文件名"
        elif "text_length" in d.columns and "Text Length" in r.columns:
            # SLR assigns new random file names on every upload, so fall back to the extracted-text length
            # (identical for the same PDF). Only accepted when every sampled length is unique on both sides.
            by_len = r.drop_duplicates("Text Length", keep=False).set_index("Text Length")
            miss_len = [t for t in d["text_length"] if t not in by_len.index]
            if miss_len or not d["text_length"].is_unique:
                raise SystemExit(f"Cannot match results to the sample: {len(missing)} file names missing and "
                                 f"{len(miss_len)} text lengths missing/ambiguous.")
            hit = by_len.loc[d["text_length"]]
            how = "Text Length(文件名已变化)"
        else:
            raise SystemExit(f"{len(missing)} sampled files are missing from {results_path}, e.g. {missing[:3]}")
        print(f"matched new results to the sample by: {how}")
        d["ai_decision"] = hit["Decision"].to_numpy()
        d["ai_reasoning_new"] = hit["Reasoning"].to_numpy()
    d["human"] = d["最终人工判断"].map({"纳入": 1, "排除": 0})
    d["ai"] = d["ai_decision"].map(ai_to_num)
    d["ai_base"] = d["ai_baseline"].map(ai_to_num)
    bad = d[d["ai"].isna() & ~d["ai_decision"].astype(str).str.lower().eq("error")]
    if len(bad):  # "Error" rows are processing failures and are handled in main(); anything else is a typo/new label
        raise SystemExit("Unrecognised AI decision values: " + str(bad["ai_decision"].unique()))
    return d

def metrics(d):
    w = d["weight"].to_numpy(float); h = d["human"].to_numpy(int); a = d["ai"].to_numpy(int)
    tp = w[(a == 1) & (h == 1)].sum(); fp = w[(a == 1) & (h == 0)].sum()
    fn = w[(a == 0) & (h == 1)].sum(); tn = w[(a == 0) & (h == 0)].sum()
    tot = tp + fp + fn + tn
    div = lambda x, y: x / y if y > 0 else float("nan")
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    f1 = div(2 * prec * rec, prec + rec) if not (np.isnan(prec) or np.isnan(rec)) else float("nan")
    po = (tp + tn) / tot
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / tot**2
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, accuracy=po, precision=prec, recall=rec, f1=f1,
                false_exclusion_rate=1 - rec if not np.isnan(rec) else float("nan"), kappa=div(po - pe, 1 - pe))

def bootstrap(d, n=2000, seed=0):
    rng = np.random.default_rng(seed); out = []
    groups = [g for _, g in d.groupby("stratum")]
    for _ in range(n):
        s = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups])
        out.append(metrics(s))
    return pd.DataFrame(out)

def fmt_ci(bs, k):
    lo, hi = bs[k].quantile(0.025), bs[k].quantile(0.975)
    if np.isnan(lo) or abs(hi - lo) < 1e-12:
        return "区间不适用(无误差,见漏收上限)"
    return f"{lo:.3f} ~ {hi:.3f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels"); ap.add_argument("key")
    ap.add_argument("--results", default=None); ap.add_argument("--out", default="report.md")
    ap.add_argument("--errors", choices=["drop", "flag"], default="drop",
                    help="how to treat AI rows with Decision=Error: drop (not scored, strata re-weighted) or flag (counted as sent to human review = positive)")
    a = ap.parse_args()
    d = load(a.labels, a.key, a.results)
    unlabeled = d["最终人工判断"].isna().sum(); unsure = (d["最终人工判断"] == "拿不准").sum()
    n_err = int(d["ai"].isna().sum())  # AI failed on these papers (Decision=Error): not a decision
    if n_err and a.errors == "flag":
        d.loc[d["ai"].isna(), "ai"] = 1  # a failed paper would be sent to a human, so it counts like a Flag (positive)
    use = d.dropna(subset=["human", "ai"]).copy()
    if n_err and a.errors == "drop":
        # complete-case adjustment: keep each stratum's total weight after dropping failed papers
        use["weight"] = use["N_stratum"] / use.groupby("stratum")["id"].transform("count")
    if len(use) == 0: raise SystemExit("No labeled rows found. Fill column F of the label sheet first.")
    m = metrics(use); bs = bootstrap(use)
    L = ["# AI 筛选评测报告", "",
         f"- AI 判断来源:{a.results or 'sample_key.csv(基线)'}",
         f"- 抽样 {len(d)} 篇;已标注 {int(d['human'].notna().sum())};参与指标 {len(use)};未填 {unlabeled};人工标「拿不准」{unsure}(不计入指标)",
         (f"- AI 处理失败(Decision=Error):{n_err} 篇;处理方式 --errors {a.errors}:" + ("不计入指标,并按层重新调整权重" if a.errors == "drop" else "按转人工复核计为正类")) if n_err else "- AI 处理失败:0 篇",
         "- 正类 = 纳入(Flag 视为转人工,计为正类)。指标已按分层权重还原到全体语料;区间为分层自助法 95% CI。", "",
         "## 加权混淆矩阵(估计的论文篇数)", "",
         "| | 人工:纳入 | 人工:排除 |", "|---|---|---|",
         f"| AI:纳入 | {m['TP']:.1f} | {m['FP']:.1f} |", f"| AI:排除 | {m['FN']:.1f} | {m['TN']:.1f} |", "",
         "## 指标", "", "| 指标 | 估计值 | 95% CI |", "|---|---|---|"]
    for k, name in [("accuracy", "准确率"), ("precision", "精确率"), ("recall", "召回率"), ("f1", "F1"),
                    ("false_exclusion_rate", "误排除率"), ("kappa", "Cohen's κ")]:
        L.append(f"| {name} | {m[k]:.3f} | {fmt_ci(bs, k)} |")
    # miss-rate bound: AI-excluded papers that a human would include
    ex = use[use.ai == 0]
    L += ["", "## 漏收上限(AI 排除的论文里,人工认为该纳入的比例)", ""]
    if len(ex) == 0:
        L.append("AI 没有排除任何抽样论文,无法评估。")
    else:
        k_miss = int((ex.human == 1).sum()); n_ex = len(ex)
        Npop = float(ex["weight"].sum())
        if k_miss == 0:
            ub = 1 - 0.05 ** (1 / n_ex)
            L.append(f"抽查 {n_ex} 篇被排除论文,未发现漏收。单侧 95% 上限约 {ub:.1%}(约 {ub*Npop:.0f} 篇/估计 {Npop:.0f} 篇被排除论文)。"
                     "写简历或汇报时请写「抽查 N 篇,未发现漏收」,不要写误排除率 0%。")
        else:
            L.append(f"抽查 {n_ex} 篇被排除论文,发现 {k_miss} 篇漏收(原始比例 {k_miss/n_ex:.1%}),详见 disagreements.csv。")
    if a.results:
        flip = use[use.ai != use.ai_base]
        fixed = flip[flip.ai == flip.human]; broke = flip[flip.ai != flip.human]
        L += ["", "## 前后对比(相对基线 sample_key.csv)", "",
              f"- 改判总数:{len(flip)} 篇;其中改对 {len(fixed)} 篇,改错 {len(broke)} 篇",
              f"- 未改判:{len(use) - len(flip)} 篇"]
        flip[["id", "filename", "ai_baseline", "ai_decision", "最终人工判断", "stratum", "备注", "ai_reasoning_new"]].to_csv(
            "flips.csv", index=False, encoding="utf-8-sig")
        L.append("- 逐篇改判见 flips.csv")
    L += ["", "## 分层原始计数(未加权,便于核对)", "", "| 层 | 抽样数 | AI 与人工一致 | 不一致 |", "|---|---|---|---|"]
    for s, g in use.groupby("stratum"):
        agree = int((g.human == g.ai).sum()); L.append(f"| {s} | {len(g)} | {agree} | {len(g) - agree} |")
    dis = use[use.human != use.ai][["id", "filename", "ai_decision", "最终人工判断", "stratum", "备注"]]
    dis.to_csv("disagreements.csv", index=False, encoding="utf-8-sig")
    L += ["", f"## 不一致样本:{len(dis)} 篇(见 disagreements.csv)", "",
          "解读提示:样本量较小时区间会很宽,这是正常的;请把区间一并写进结论,而不是只报单个数字。"]
    open(a.out, "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))

if __name__ == "__main__":
    main()

# SLR 评测

本目录包含两套本地评测。筛选评测检查 AI 是否正确纳入或排除论文。抽取评测检查 AI 填写的涂层材料、制备方法和原文引句。两套评测都读取已导出的结果。

以下命令从仓库根目录运行。输入文件请替换为相应的本地路径。

## 筛选评测

```bat
python evaluation/screening/make_label_sheet.py screening_results.xlsx
python evaluation/screening/evaluate.py label_sheet.xlsx sample_key.csv --out report_baseline.md
python evaluation/screening/evaluate.py label_sheet.xlsx sample_key.csv --results new_screening_results.xlsx --errors drop --out report_updated.md
```

抽样脚本从 AI 排除、边界纳入和其余纳入三组分别抽样。填写 `label_sheet.xlsx` 时先独立判断，再用 `sample_key.csv` 对照 AI 结果。把这份样本对照表留在本地，前后评测使用同一批论文、同一套人工判断和同一份基线权重。

若文件名随重新上传发生变化，脚本可用基线对照表的 `text_length` 与新版结果的 `Text Length` 匹配。每个样本的文字长度在两边都必须能唯一对应。`--errors drop` 排除 AI 处理失败的记录，并重新计算各组权重。`--errors flag` 把失败记录计为转人工复核。

如果需要另抽一批尚未标注的验证样本，可运行 `python evaluation/screening/make_label_sheet.py screening_results.xlsx --exclude-key sample_key.csv --prefix val_`。这个结果只代表排除基线样本后剩下的论文。

## 抽取评测

抽取评测由 `check_excerpts.py`、`make_extraction_sheet.py` 和 `evaluate_extraction.py` 同目录运行。完整命令、参数和输出文件见[抽取评测说明](extraction/README.md)。

前后对比时，按文件对应同一批论文，保留抽取对照表里原来的序号和人工标注。将新的 AI 抽取字段填入配套对照表，再运行评分和引句校验。重新上传导致文件名变化时，先核对原始文件的对应关系。

## 记录结果

按「抽样 → 盲标 → 评测 → 前后对比」进行。盲标指人工判断时先隐藏 AI 的答案。比较报告里的指标和逐篇分歧，记录模型和筛选规则中实际改变的部分。运行下一次评测前，保存需要保留的逐篇明细，因为脚本会写入同名文件。

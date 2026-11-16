# 抽取评测:涂层材料、制备方法、原文出处

流程(每一步的输入输出见各脚本开头的说明):

1. 自动校验原文出处(不需要人工,对全部抽取结果都能跑):
   `python check_excerpts.py extraction_results.xlsx --pdf-dir <存放 PDF 的文件夹> --out excerpt_check.csv`
   PDF 要能按抽取表里的 Filename 找到(SLR 上传后保存的就是这些哈希文件名)。
2. 生成人工标注表(盲标,不显示 AI 的答案):
   `python make_extraction_sheet.py extraction_results.xlsx --n 20`
   得到 `extraction_label_sheet.xlsx` 和 `extraction_key.csv`。
3. 人工填写标注表后评测:
   `python evaluate_extraction.py extraction_label_sheet.xlsx extraction_key.csv --excerpt-check excerpt_check.csv`
   得到 `report_extraction.md`、`material_comparison.csv`、`method_comparison.csv`。

依赖:pandas、openpyxl;读 PDF 优先用 pymupdf4llm(与 SLR 一致),没有则退回 pdfplumber。
制备方法的归类规则在 `extraction_utils.py` 的 `METHOD_RULES`,可按你的领域增删关键词。

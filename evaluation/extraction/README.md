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

### 涂层材料评分

每篇先识别两边的材料组分，统一化学式下标及已收录的同义词。脚本保留含小数或有括号的化学式，并排除能由句式明确识别的基材和电解液配方。材料字段明确声明的添加组分仍参与比较。

`material_comparison.csv` 提供原始答案、识别出的组分、匹配（`matched`）、漏报（`missing`）、多报（`extra`）和待复核原因。完全一致要求两边组分集合相同，部分一致要求至少有一个组分重叠。脚本把只有大类的标注、举例式答案和归属不清的多层写法列为待复核。材料比例以可自动评分的篇数为分母，报告同时给出待复核篇数。

自由文本的识别依赖规则。材料明细中的「匹配」表示两边答案对应，原文证据由引句校验与人工复核提供。

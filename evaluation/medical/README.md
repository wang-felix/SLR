# Issue #4 医学评测数据集

在 D:\SLR 下使用现有 Anaconda 环境运行。

```powershell
& 'D:\Anaconda\envs\slr\python.exe' -X utf8 evaluation/medical/fetch_medical_benchmark.py
& 'D:\Anaconda\envs\slr\python.exe' -X utf8 -m pytest tests/test_evaluation_extraction.py
```

脚本通过 OpenAlex 的 Cochrane ISSN、PMC 编号和 CC 授权过滤，依照代码内的干预综述与专题清单检查 PMC 开放数据桶。它按 S3 对象发现文章版本，核查 PMC 授权与文章身份，并用 PMC 元数据提供的 MD5 校验下载内容。HTTPS 请求超时为 25 秒，瞬时故障最多重试 3 次。

从综述 JATS XML 的 Characteristics of included studies 特征表保留 Methods、Participants、Interventions 和 Outcomes 四项专家文本。优先使用表标题中的引用 ID 关联纳入研究的参考文献组。只有在已排除标题明显属于方案、二次分析等文献后，可唯一识别原始试验报告时才进入 PDF 环节。OpenAlex 批量按 PMID 或 DOI 查询，PMC 元数据还须匹配原始引用中的 PMID 或 DOI。每项研究只收录一份报告。专家表描述的是研究层面，可能综合多份报告。Outcomes 原文可能只列结局指标，主结果是否由单篇 PDF 支持需在后续评分前核对。

PDF 需要有足够的原生文本，使用 pymupdf4llm 关闭 OCR（图像文字识别）提取。按可识别的正文标题起算，遇参考文献标题停止，正文归一化空白后须至少 5,000 字符。字符数与页数写入 CSV。

输出位于 Git 忽略的 `test_output/medical_eval_v1/`。

- `gold_standard.csv` 保存已通过的论文及四项外部标注。
- `manifest.json` 保存来源、授权、文件校验值及每篇综述的试验匹配与排除记录。
- `pdfs/` 和 `cache/` 保存论文 PDF 与来源 XML。以本次 CSV 和 manifest 的条目为准，目录可能包含上次运行留下的文件。

收满 20 篇即停止。同一综述最多接受 8 篇，每个候选综述对应不同专题，因此 20 篇至少覆盖 3 个专题。最多扫描 8 篇含专家特征表的综述。达到上限或候选耗尽时保留已验证的部分结果，退出码为 2。流程只访问公开文献服务。

## 第 3 步医学对照评测

在 `D:\SLR` 下运行。离线预检只运行 PyMuPDF4LLM，不读取 API Key。两组使用同一份去除参考文献的全文，单篇输入上限 100,000 字符。出现文件解析失败或需截断时退出。

```powershell
& 'D:\Anaconda\envs\slr\python.exe' -X utf8 -m evaluation.medical.run_medical_eval --dry-run
```

默认读 `test_output/medical_eval_v1/`，默认写 `test_output/medical_eval_v1/evaluation_v1/`。四个脚本均可用 `--dataset` 和 `--output` 指定数据与产物目录。预检写入 `dry_run_summary.json`，与在线统计分开。

### 抽取与预算

`run_medical_eval.py` 跳过筛选。G1 用默认提示词和普通 JSON 抽取。G2 用医学字段提示词、先引句后作答和现有 instructor 结构化路径。两组模型、全文、思考模式、温度 0.05、最大输出 4,000 token 相同。第四字段 `outcomes_measured` 使用旧金标准 CSV 的 `gold_outcomes_and_main_findings` 列，只评结局指标和时间点。

`deepseek-chat` 已停用，默认请求 `deepseek-flash`。模型文档和价格见 [DeepSeek 官方说明](https://api-docs.deepseek.com/quick_start/pricing/)。在线阶段由环境变量 `DEEPSEEK_API_KEY` 提供密钥。命令行需提供以下四个人民币每百万 token 单价。输入价使用缓存未命中价。

- `--offpeak-input-cny-per-million`
- `--offpeak-output-cny-per-million`
- `--peak-input-cny-per-million`
- `--peak-output-cny-per-million`

默认预算为 `--budget-cny 8`。每次发送前，输入预留按已序列化请求体的 UTF-8 字节数加 4,096 计算，以模型上下文上限为封顶。使用这个输入预留和请求的最大输出按高峰价检查剩余预算，额度不足时在发送前停止。账本记录请求体字节数和预留输入 token。收到有效用量后按回报结算，若回报超过预留则停止。请求中断或用量缺失时保留整笔预留。4,096 用于覆盖服务端格式开销，实际账单以平台为准。

在线试跑可加 `--limit 5`。继续全部样本时移除 `--limit`，沿用相同的模型、单价、数据和输出目录。已记录的论文/组别直接复用，两次运行共享预算账本。`resume_state.json` 每次请求前后及每组每篇完成后原子写入，进程中断后未结算的请求仍计入预算。更换模型、单价或数据需要新输出目录。原始组别 CSV 分别为 `G1_extraction.csv`、`G2_extraction.csv`，`run_summary.json` 保存每组解析率、输入/输出/总 token、抽取耗时与每百篇折算。未结算用量标为未知。

### 原文核验、盲评和报告

完整双组输出就绪后，按顺序运行前两条。人工填完工作簿的两项评分后运行第三条。

```powershell
& 'D:\Anaconda\envs\slr\python.exe' -X utf8 -m evaluation.medical.check_medical_quotes
& 'D:\Anaconda\envs\slr\python.exe' -X utf8 -m evaluation.medical.make_blind_review_sheet
& 'D:\Anaconda\envs\slr\python.exe' -X utf8 -m evaluation.medical.score_medical_eval
```

- L1 校验生成 `l1_quotes.csv` 和 `l1_summary.json`。分别报告严格匹配 `strict_pass`、分段匹配 `fragmented_pass` 和两者合计 `verbatim_match`。冻结门槛使用合计值，两类匹配有物理页码时均计入定位率。物理页码从 1 开始，分母为实质引句。引句与明确的 `Not reported` 事实值另计一致性。
- `blind_review_sheet.xlsx` 的 `1_Accuracy_Blind` 在全量 20 篇时有 160 行，仅提供准确度评分所需列。完成后按「评分说明」取消隐藏 `2_Evidence_Support`，为有实质引句的字段评分。解盲时自动将其余字段计为 `No_Quote`。映射表 `blind_key.csv` 单独保管。证据页的默认隐藏用于分阶段评分，工作簿可自行取消隐藏。生成器会保护已有工作簿或密钥。
- 解盲输出 `medical_eval_report.md`，按第 2 版通过门槛和止损红线判断。小于 20 篇的双组样本只给出预评估。字段准确度排除 `not_in_pdf`，引句逻辑支撑度排除 `No_Quote`。95% Wilson 区间按字段观测计算，同篇试验字段间的相关性未纳入区间。

离线自动化用合成两页 PDF 与模拟 HTTP 返回验证这条流程，真实试验的抽取质量和线上账单需要另行实测。

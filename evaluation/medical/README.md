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

# 评测提示词与抽取字段

本目录存放耐磨与耐腐蚀陶瓷涂层文献筛选及抽取评测使用的公开提示词和字段模板，不含论文数据、标注或运行导出。

| 文件 | 用途 | 文件 SHA-256 | Full-text `prompt_hash` |
| --- | --- | --- | --- |
| `screening_prompt_v2.txt` | 全文筛选提示词 v2 | `01be66145181bd986836b89e5308f2625bed23e8fd606044335c5d16a9f4b768` | `13c584cbd6b2a31b21bdfc150073c84594cef49e610682566f5b3d88caadb529` |
| `screening_prompt_v3.txt` | 全文筛选提示词 v3 | `6409b5a0f87c4f63431d9136c2a9ed83a4ac496857f12e45c2825e60d5eb5590` | `fb6209f964265a4d2ce31b12a1c9c0881f95727d9452b2b7788fe1c2545809fd` |
| `screening_prompt_v4.txt` | 全文筛选提示词 v4 | `5d7b22fe00ed213fc68d4f241c24c257ca0aab6ed7354f4bb7de2b20a3874c74` | `3db7a6b166ffbb45e020ea884ff40604fbed1f2b7d4c363fe145ff7f78ca23d3` |
| `extraction_fields_v2.txt` | 抽取字段名及顺序 | `959897f098b976130974bdf084282b20272f65e9fe2d033622dabc137c7542bd` | 不适用 |

- `screening_prompt_v2.txt` 在初版基础上按涂层连续相明确陶瓷涂层与金属基涂层的界定。
- `screening_prompt_v3.txt` 在 v2 基础上规定，含作者自有涂层实验的综述按研究论文处理。连续相或实验归属无法确定时判为 `Flag for Human Review`。
- `screening_prompt_v4.txt` 沿用 v3 的筛选规则，新增 `confidence` 字符串字段，取值为 `High`、`Medium` 或 `Low`。v4 尚未在真实论文语料上实测。
- `extraction_fields_v2.txt` 固定 5 列字段的顺序，将原文逐字引句与章节位置拆为独立字段。出处校验时可用 `--excerpt-column` 指定引句列，见 [check_excerpts.py](../extraction/check_excerpts.py)。

这些文件仅用于评测复现与审计核对。评测结果以评测报告为准。

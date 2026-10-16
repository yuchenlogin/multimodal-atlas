# 证据审阅的精确替换建议

以下是建议，不是已应用的修改。行号对应首稿；主审需同步重建图、HTML、PDF。

## H1 · MMMU（推荐保留整体validation目标）

将 `reports/10-benchmarks-eval/report.md:61` 整行替换为：

```markdown
![图 3：MMMU v4 官方全文 Appendix Table 4 的 Validation Overall 列（900 题）：BLIP-2 FLAN-T5-XL 34.4、LLaVA-1.5-13B 36.4、GPT-4V Playground 56.8。固定使用该版本，未混入 test 或学科子集数字。](../../assets/technical/10-evidence.png)
```

引用 `[1]` 改为锁定 `https://arxiv.org/html/2311.16502v4`。`evidence.json` topic10 用以下字段；其他身份字段保留：

```json
{
  "arxiv": "2311.16502v4",
  "url": "https://arxiv.org/html/2311.16502v4",
  "locator": "Appendix B.1, Table 4, Validation Overall column (900 questions)",
  "metric": "MMMU validation overall accuracy (%)",
  "labels": ["BLIP-2 FLAN-T5-XL", "LLaVA-1.5-13B", "GPT-4V Playground"],
  "values": [[34.4, 36.4, 56.8]],
  "series": ["v4, validation overall, n=900"],
  "caveat": "Historical paper table, not test or a live leaderboard. Table 5 Art & Design validation has different values and n=120. Model training/access differ, precluding causal architecture attribution."
}
```

## H2 · Chameleon（推荐统一1048题Overall）

将 `reports/12-frontier/report.md:79–83` 的图注和随后两段替换为：

```markdown
![图 3：Chameleon v2 Appendix Tables 11–12 的同一 Overall 行（1048 个提示，均要求图文混合回答）：对 Gemini+ 为 58.8%，对 GPT-4V+ 为 51.6%。两种增强基线均由原模型先生成 caption，再用 DALL-E 3 生成图像；分数为 (wins + 0.5×ties)/N，不是严格胜出比例。](../../assets/technical/12-evidence.png)

这组证据描述特定长篇图文生成协议下的含平局折算偏好分数，不能把 51.6% 无条件称为统计显著优势，也不能外推成所有视觉问答更强。原论文引言/正文给出的 60.4% 与附录 Overall 58.8% 不一致；60.4% 对应 Table 11 的 441 个 mixed-modal-input 提示子集。本图使用完整表中的同一 Overall 行，避免混合分母。

因此阅读前沿报告时，应先核对候选系统的输出链、展示方式、评判规则和样本集合。这里的 Gemini+、GPT-4V+ 已经补上图像生成器，不能当作没有图像输出的裸基线；它们与端到端 Chameleon 的差异仍包含模型、训练和系统链，不是隔离网络结构的因果实验。
```

`evidence.json` topic12相关字段替换为：

```json
{
  "locator": "Appendix Tables 11–12, Overall rows; Section 4.2 baselines and 4.2.2 pairwise scoring",
  "metric": "Tie-adjusted pairwise preference score (%)",
  "labels": ["vs Gemini+ (DALL-E 3)", "vs GPT-4V+ (DALL-E 3)"],
  "values": [[58.8, 51.6]],
  "series": ["Overall, n=1048; win + 0.5 tie"],
  "caveat": "Both baselines generate captions followed by DALL-E 3 images. All prompts request mixed-modal responses; 441 have mixed-modal input and 607 text-only input. Score=(wins+0.5*ties)/n. Intro/body's 60.4 conflicts with Overall 58.8; 60.4 is the mixed-input subset row. Not visual reasoning accuracy or evidence of significant superiority."
}
```

若主审改用441题mixed-input子集，则两个值应同时改为60.4/47.3，不能只换其中一点。

## M1 · CLIP

将 `reports/01-vision-language/report.md:79` 替换为：

```markdown
图中三者的输入分辨率均为 224×224，但模型规模、patch 划分与训练计算不同，不能从整体趋势隔离某个因素在固定计算下的效应。它们说明自然语言监督得到的表示可以用于零样本类别匹配；并不证明模型能够理解遮挡关系、阅读任意密集文字，或在问答时保持视觉事实性。
```

ledger topic01 caveat去掉“resolution differs”，加“all three use224×224; 336px model is a separate row”。

## M2 · RT-2

ledger topic06：

```json
{
  "metric": "Unseen-condition generalization success (%)",
  "caveat": "Author-reported Average over unseen objects, backgrounds and environments, each with easy/hard conditions. Not an unseen-task or cross-embodiment score. Raw per-trial outcomes are unavailable here, so uncertainty cannot be reconstructed."
}
```

正文60行首句改为：“这张表可以区分两个问题：在论文设置下，5B 的预训练初始化将报告的平均成功率从 9 提高到 42；共同微调在不同模型规模上的收益并不相同。”其余段落保留。

## M3 · OSWorld

64行图注改为：

```markdown
![图 3：OSWorld v2 原文 Table 5 的 GPT-4 + a11y tree 原始基线为 12.24%，人类任务成功率为 72.36%。12.24% 不是 screenshot-only VLM 结果；模型、时间预算与交互条件不同，不是隔离单个架构部件的因果对照。](../../assets/technical/07-evidence.png)
```

ledger labels首项改为 `GPT-4 + a11y tree`，locator改为 `Table 5 and Section 4.2; original benchmark setup`。

## M4 · DataComp

64行替换为：

```markdown
DataComp [1] 固定训练代码和预算，把数据选择作为比较对象。图中 LAION-2B 的最终训练集比 WIT、LAION-400M 和 DataComp-1B 更大，但在相同累计训练曝光下并未得到最高 ImageNet 准确率。这里比较的是最终训练集，不是过滤前候选池；DataComp 的 CommonPool 候选池另有 12.8B 对。它支持“数据来源和选择值得独立研究”，不支持“一个相似度阈值在所有任务上最优”。
```

## M5 · 版本与记录

版本对应表见 `evidence.md`。使用实际已核验版本更新结果的 arxiv/url 和report引用。retrieval-log补充真实纠错记录：MMMU误读Art & Design Table5为Table4已修；Chameleon正文/摘要与完整表冲突，采用相同Overall行与含tie定义。不要把DataCite的论文级DOI强行加v后缀。

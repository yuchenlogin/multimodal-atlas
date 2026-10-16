# 研究方法与证据审阅

审阅日期：2026-10-16。审阅对象：`content/technical-report-rewrite` 的首稿工作树，基准提交 `fbb3820`；下列行号以发现时的文件为准。本审阅不修改正文、图表或代码。

结论：**需要修改后复核**。13 组定量证据中，11 组数值和表列匹配；MMMU 与 Chameleon 存在影响含义的取数/协议错误。另发现 5 项 medium。旧 `paper/` 的实证陈述是本轮之前遗留的真实性问题，另列 H3，不能归为本轮新引入。

## 核验过程与范围

- 完整审读 13 篇 `reports/*/report.md`，以及 `sources/evidence.json`、`core-reading.json`、`primary/datacite-verified.json`、`retrieval-log.md`。
- 逐组打开原论文官方全文，核对模型行、结果列、条件与版本；不使用搜索摘要补数。主表解析可能混淆时，另下载官方 PDF，使用 Poppler 文本与页面图像复核。
- 本审阅重新调用全部 23 个真实 DataCite DOI endpoint 的 `creators` 字段，均成功返回。它确认作者身份与 DOI，不替代实验核验。endpoint 模式为 `https://api.datacite.org/dois/10.48550/arXiv.{id}`，日期为本审阅日。
- 原流程的 arXiv Atom / Semantic Scholar 429 不构成跳过全文的理由。本审阅成功取得所有核心全文；SigLIP 读官方 PDF。web 工具读取 MMMU、Chameleon PDF 因 16.9 MB / 25.2 MB 限额失败后，改用 `curl` 成功下载，再用 `pdftotext` 和 `pdftoppm` 核对，未把失败记作成功。
- 审阅的教学计算不是大型模型复现。本文没有自行运行大型模型、GPU 加速基准或机器人实验，也不据此确认真实性能。

## 必须修正的发现

### H1 · high · MMMU 把 Art & Design 子集误标为整体 validation

位置：`reports/10-benchmarks-eval/report.md:61`、`:85`；`sources/evidence.json:278–299`；相关 `10-evidence.png`、HTML、PDF。

官方 [MMMU v4 Appendix B.1/Table 4](https://arxiv.org/html/2311.16502v4) 的整体 validation 为 900 题，三个模型的值依次为 **34.4、36.4、56.8**。首稿的 **44.2、51.7、65.8** 来自随后 B.2 / **Table 5** 的 Art & Design validation（120 题），不是 Table 4。已经用 [官方 v4 PDF](https://arxiv.org/pdf/2311.16502v4) 第 15、16 页交叉确认，不能用 HTML 表格位置模糊来保留原数字。

建议：若保留整体评测教学目标，换成 34.4/36.4/56.8，锁定 v4、Table 4、900 validation 题；同步 ledger、图、caption 和输出产物。若选择保留原数，则必须将题目/指标完整改为 Art & Design 子集并引用 Table 5。首选前者。

### H2 · high · Chameleon 混用了子集与全体，并漏掉 baseline 的图像生成器和 tie 定义

位置：`reports/12-frontier/report.md:79–83`；`sources/evidence.json:335–351`；相关 `12-evidence.png`、HTML、PDF。

[Chameleon v2 §4 与 Appendix Tables 11–12](https://arxiv.org/html/2405.09818v2) 的完整表显示：对 Gemini+ 的 **Overall 是 58.8%**，对 GPT-4V+ 是 **51.6%**，两者均 N=1048；mixed-modal-input 子集 N=441 的值则是 **60.4% / 47.3%**。原文引言及结果叙述本身把 60.4 与 51.6 并列，但 Table 11 明确将 60.4 放在 mixed-modal prompts 行，不能将其当成同一全体协议的两点。

此外，“+”代表基线生成 caption 后调用 **DALL-E 3** 生成图像；结果不是裸 Gemini-Pro / GPT-4V 的比较。win rate 的定义是 **(wins + 0.5 × ties) / N**，不是严格偏好占比。Overall 对 Gemini+ 的计数为 435/362/251，对 GPT-4V+ 为 375/331/342（win/tie/loss）。已视觉核对 [官方 PDF](https://arxiv.org/pdf/2405.09818v2) 第 26 页。

建议：统一使用 Tables 11–12 的 Overall **58.8/51.6**，完整写 baseline 与 tie 定义；保留“不代表视觉推理准确率、不证明显著优势”的边界。原 83 行“没有图像输出接口的基线”在解释这两点时尤其误导，因为此处被比较的增强基线已经有 DALL-E 3 输出。不能只在 caveat 增加“协议不同”后保留错误标签。

### H3 · high · 历史 paper/ 将未执行实验写成已执行结果（本轮前遗留）

位置：发现时的 `paper/index.html:14、20`；`assets/paper-verifier-gated-visual-decoding.pdf` 第 1 页摘要与贡献、第 4–6 页结果表/消融/延迟叙述。

旧 PDF 明确声称内部重跑多模型、保存匿名仓库日志，并给出 +2.0 POPE F1、−5.9 CHAIR-S、约 1.3× 延迟等实证结论。主审已确认没有执行这些实验，故“设计稿”或 limitations 声明不能使其成为真实结果。网页还把未经核验的“思考越多、看见越少”和“已有工作只做单一维度”作为确定选题依据。

建议：移除所有结果、消融、延迟、已提供日志/实现和新颖性优越声明，重写为明确未执行的研究方案；新 PDF 应由方案源文件生成，并复查旧 PDF 与公开页面都已被替换。主审正在处理；待 `paper/proposal.md` 落盘后再复核。此项不是对本轮 13 章首稿的新增指控。

### M1 · medium · CLIP 的三个选定模型分辨率并未改变

位置：`reports/01-vision-language/report.md:79`；`sources/evidence.json:48`。

[CLIP Appendix A.2 与 Table 20](https://arxiv.org/html/2103.00020v1) 明确列出所选 B/32、B/16、L/14 都使用 **224×224**。336 分辨率属于未绘出的 L/14-336px 行，其 Table 11 ImageNet 值为 76.2。63.2/68.6/**75.3** 的取数正确，DataComp 的75.5也确实是另一张表；错误仅在把分辨率列为这三个点共同变化的因素。

建议：改成“图中三者均为224×224；模型规模、patch划分与训练计算不同，不能从整体趋势隔离某个因素在固定计算下的效应”。

### M2 · medium · RT-2 metric 将未见条件误称为未见任务

位置：`sources/evidence.json:180`（会进入图的纵轴）。正文 `reports/06-embodied/report.md:58` 的“未见条件”表述是正确的。

[RT-2 v1 Appendix H.3/Table 8](https://arxiv.org/html/2307.15818v1) 的六列是 unseen **objects / backgrounds / environments** 各 easy/hard，Average 为 **9/42/44/52/63**。它不是统一的 unseen-task benchmark，也不等于跨机器人泛化。

建议：metric 改为 “Generalization success under unseen conditions (%)”，caveat 明列三类条件与 easy/hard，保持作者汇总 Average，不自行对已四舍五入的六列重算。`:60` 的“显著改变”也宜改为“报告值大幅改变”，避免无试验级数据却暗示显著性检验。

### M3 · medium · OSWorld 12.24 的输入条件应写到结果标签

位置：`reports/07-agent-tooluse/report.md:64`；`sources/evidence.json:209、212`。

[OSWorld v2 Table 5](https://arxiv.org/html/2404.07972v2) 中 **12.24 是 GPT-4 + accessibility tree**，不是 screenshot-only VLM 的最佳分数。两数 12.24/72.36 与摘要及 §4.2 一致，已有“非实时排行榜”限定也正确，但泛称最佳 Agent 隐去了本章恰应解释的观察接口。

建议：图标签、caption 和 ledger 直接写 “GPT-4 + a11y tree (original baseline)” 与 Human；locator 包含 Table 5。无需删除这组历史结果。

### M4 · medium · DataComp 将训练集大小称为候选数据

位置：`reports/11-data-training/report.md:64`。

[DataComp v5 Table 1 与引言](https://arxiv.org/html/2304.14108v5) 的 2.3B LAION-2B 和 1.4B DataComp-1B 是最终 dataset size；DataComp 候选池 CommonPool 是 12.8B。75.5/72.8/73.1/79.2、13B exposure 和 1.1e21 MACs 的取数正确，但“LAION-2B 的候选数据更多”混淆了本章核心区分。

建议：改成“LAION-2B 的最终训练集比 WIT、LAION-400M 和 DataComp-1B 更大，在相同累计训练曝光下并未得到最高 ImageNet 准确率”；不从该表推断候选池大小效应。

### M5 · medium · 结果来源需锁定实际读到的 arXiv 版本

位置：`sources/evidence.json:4–6、28–30` 等未带版本的11组结果，`sources/retrieval-log.md:3`。

数值快照加访问日期仍不能让读者重取当时的表；这也弱于 plan.md 对“每个引用结果记录论文版本”的验收要求。本次通过实际 HTML 的 abstract/PDF 入口读得版本如下，不是按发表年份猜测：

| topic | 实际核验全文 | 位置 |
|---|---|---|
| 00 | 2203.15556v1 | §4.2.2 / Table 6 |
| 01 | 2103.00020v1 | Table 11；分辨率 Table 20 |
| 02 | 2212.04356v1 | Appendix D.1.2 / Table 9 |
| 03 | 2405.21075v3 | Main Table 4 / GPT-4o |
| 04 | 2503.20215v1 | Table 7 |
| 05 | 2212.09748v2 | Table 3 |
| 06 | 2307.15818v1 | Appendix H.3 / Table 8 |
| 07 | 2404.07972v2 | Table 5 / §4.2 |
| 08 | 2205.14135v2 | Table 1 |
| 09 | 2305.10355v3 | Table 1 |
| 10 | 2311.16502v4 | Appendix Table 4（修正后） |
| 11 | 2304.14108v5 | Table 1 |
| 12 | 2405.09818v2 | Appendix Tables 11–12（修正后） |

建议：将 ledger 的 arxiv/url 锁定上述版本；report references 同步；retrieval-log 记录此次表列纠错及来源冲突，不只保留泛称“手工核验过”。DataCite DOI 可继续不带版本，因为它承担论文身份核验，而非表格快照身份。

## 逐篇独立结论

| 报告 | 数值与机制核对 | 本篇结论 |
|---|---|---|
| 00 基础 | [Chinchilla](https://arxiv.org/html/2203.15556v1) Table 6 为60.0/67.6；摘要67.5差异确实存在。attention维度、FLOPs、2048/512MiB KV核算与6ND适用边界成立。 | 没有发现本篇实质数值错误；保留摘要/表格冲突说明，按M5补版本与Table 6定位。未把scaling经验外推为多模态定律。 |
| 01 视觉语言 | [CLIP](https://arxiv.org/html/2103.00020v1) Table11三个值正确；[SigLIP官方PDF](https://arxiv.org/pdf/2303.15343) Algorithm1确为除N，Table1的4 TPUv4/两天/84.5属于冻结图像塔SigLiT。[BLIP-2](https://arxiv.org/html/2301.12597) 两阶段与32-query瓶颈叙述匹配。 | 需修M1；其余梯度与冻结接口说明没有发现事实性错误。 |
| 02 音频 | [Whisper](https://arxiv.org/html/2212.04356v1) Table9为多语言checkpoints，clean 3.3/2.7/2.8/2.5，other 7.2/5.6/5.7/4.9，beam+temperature fallback正确；[EnCodec](https://arxiv.org/html/2210.13438) RVQ/重建/对抗目标匹配。 | 证据适配本章，未混入.en或greedy结果；50Hz、4kbps、96倍明确是假设核算。需补M5版本，未发现其他实质错误。 |
| 03 视频 | [Video-MME v3](https://arxiv.org/html/2405.21075v3) GPT-4o短中长两行数值与有无字幕列一致。抽帧命中率推导明确随机相位与单帧检测条件。 | 通过本次实质事实核对；字幕信息增益未被冒称视觉时间推理因果增益。把主表locator细化为Table4即可。 |
| 04 Omni | [Qwen2.5-Omni](https://arxiv.org/html/2503.20215v1) Table7的65.1/71.6与64.3/72.4正确。§2.3文字token传入Talker、§2.4两秒audio块、2×2视觉合并与DiT lookback/lookahead均有原文依据。 | 需补M5版本；没有将支持流式误称已验证全双工，队列实验明确非模型实测，未发现其他实质错误。 |
| 05 生成 | [DiT v2](https://arxiv.org/html/2212.09748v2) 512² Table3为12.03/4.64/3.04，recall 0.64/0.57/0.54；未混入256²。[DDPM](https://arxiv.org/html/2006.11239) 后验与simple loss区分成立；[Transfusion](https://arxiv.org/html/2408.11039) 混合目标与图内双向/跨序列约束叙述匹配。 | 需补M5版本；不把FID当提示遵循或把simple objective等同完整ELBO，未发现其他实质错误。 |
| 06 具身 | [RT-2](https://arxiv.org/html/2307.15818v1) 五个Average数值匹配，具体类别由M2列明；[OpenVLA](https://arxiv.org/html/2406.09246) 离散动作接口引用匹配。 | 需修M2与M5。正文的坐标契约、闭环频率、试验级统计边界合理；不要用“显著”暗示已作统计检验。 |
| 07 Agent | [OSWorld](https://arxiv.org/html/2404.07972v2) 历史数值正确，12.24具体来自a11y-tree GPT-4。p^n、一次独立重试公式在写明的假设下成立。 | 需修M3与M5。未将独立失败教学模型当真实Agent预测，也未以合法工具返回替代终局正确。 |
| 08 效率 | [FlashAttention](https://arxiv.org/html/2205.14135v2) Table1为8×A100、10次、20.0±1.5/17.4±1.4到72%MLM，正文一致。online-softmax更新、受限roofline与Amdahl核算成立。 | 需补M5版本；正确保留±含义未明的边界，没有外推任意GPU的decode加速，未发现其他实质错误。 |
| 09 对齐 | [POPE](https://arxiv.org/html/2305.10355v3) Table1 CHAIR-S32.7/62.7与长度64.3/90.7正确；[DPO](https://arxiv.org/html/2305.18290) §4/AppendixA的KL解、BT差分和导数一致；[R1](https://arxiv.org/html/2501.12948) 为文本RL案例，正文未冒充多模态验证。 | 需补M5版本；因果归因与事实性/拒答/执行安全拆分合理。low建议：97行明确其“安全”主要聚焦任务/权限，避免被误读为完整危害内容安全定义。 |
| 10 评测 | H1为严重表列错误。其余200题配对示例的45%/50%及5个百分点正确；cluster与采样层次说明合理。 | **本篇当前不能通过**：恰在讲split/版本边界时误用了学科子集。修H1及重绘后再核。 |
| 11 数据 | [DataComp](https://arxiv.org/html/2304.14108v5) 四个数与模型/曝光/MACs吻合；重复族交叉概率成立且分母解释正确。 | 需修M4/M5；没有把池大小与exposure等同，除64行用词外来源/分布/过滤器的因果限制合适。 |
| 12 前沿 | H2为严重协议错误。[Chameleon §2.1](https://arxiv.org/html/2405.09818v2) 512²→1024码及重OCR弱点正确；[V-JEPA2 §3.1](https://arxiv.org/html/2506.09985) 约62小时仍含末端状态、无任务/奖励/成功标签的解释准确。 | **本篇当前不能通过**：修H2并重绘。三个待执行实验明确无预期提升百分比；文本RL不推出视觉grounding、观测不推出动作干预，证据边界处理成立。 |

## 作者、论文身份与机制来源

本次23个真实DataCite `creators`返回的首署名：Hoffmann（Chinchilla）；Radford（CLIP、Whisper）；Fu（Video-MME）；Xu（Qwen2.5-Omni）；Peebles（DiT）；Brohan（RT-2）；Xie（OSWorld）；Dao（FlashAttention）；Yifan Li（POPE）；Yue（MMMU）；Gadre（DataComp）；Chameleon Team；Vaswani（Attention）；Zhai（SigLIP）；Junnan Li（BLIP-2）；Défossez（EnCodec）；Ho（DDPM）；Rafailov（DPO）；Kim（OpenVLA）；Zhou（Transfusion）；DeepSeek-AI；Assran（V-JEPA2）。没有发现首稿把上述论文署给错误研究者的情况。Qwen官方HTML首页使用团队署名Qwen Team，DataCite以Jin Xu等个人作者登记，两者可并存，现有团队引用可接受。

机制来源与正文对应关系已逐篇检查：Attention为scaled dot product；SigLIP为pairwise sigmoid与/N；BLIP-2为冻结塔和Q-Former的两阶段；EnCodec为RVQ与多项训练目标；DDPM为前向/后验/简化损失；DPO为KL最优策略与偏好概率；OpenVLA为视觉语言到离散动作；Transfusion为文本CE与图像diffusion共享Transformer；R1为文本推理RL；V-JEPA2为观测预训练与动作条件后训练。没有用元数据替代这些方法内容的核验。

## 修订后的复核要求

H1、H2必须同步修正文、ledger、图、网页和PDF；H3必须替换旧公开PDF与旧页面陈述。M1–M5应全部解决后才能把“13组结果已核验”作为发布结论。`evidence-patches.md` 提供精确替换建议；复核以实际落盘产物为准，不以“已计划修改”销项。

## 修订复核记录 1

主审已按原表修正 MMMU 与 Chameleon 的正文/ledger 数值；CLIP 分辨率、DataComp 最终训练集措辞、OSWorld 输入标签和 RT-2 metric 已改。本审阅按新授权更新了13组结果的版本、URL和详细locator，新增 `sources/primary/source-versions.json`；每条的官方 `/abs/idvN` 和 `/pdf/idvN` 两个入口均已实际确认。Video-MME v3 的性能主表定位为 **Table 4**。

新 `paper/proposal.md` 已完整审读：明确未运行模型实验、撤回旧数字、不主张新颖性或固定收益；具体算法、预算对照、验证器独立性与风险—覆盖的验证标准具有可证伪性。正文真实性问题已消除。尚待两处轻微用词修正：MMMU caption 的 Appendix B.2 应为 **B.1**；proposal50行“会产生高度相关的错误”宜改为“可能产生相关错误”，因为相关性正是后面计划检验的量。RT-2 的“显著改变”建议改为“报告值大幅改变”。

本轮复核时图表和PDF仍在重建，因此暂不销项最终产物核验；旧PDF尚不能以新proposal文本替代检查。

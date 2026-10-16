# 科学真实性终轮复核

日期：2026-10-16。角色：方法与来源核验；不承担代码审查。范围为新增第08/09/11章技术补文、13组主要定量证据修订，以及研究方案的事实边界。仅修改本审阅与 `sources/core-reading.json` 的新条目，不改正文、图表或构建代码。

## 当前结论

新增第6–8组技术补文未发现需要修改的high/medium事实错误。13组定量ledger继承首轮逐表核验，并已核查纠错后的数值、标签、split、baseline与版本落盘。H1/H2、M1–M5在源文与ledger中已得到实质修正；图和最终HTML/PDF仍须以实际重建产物销项。H3的研究方案源文已符合“未运行实验”的边界；旧公开页面和PDF替换待主编通知后复核，不能提前标为完成。

## 08：量化、并行与服务

真实取得以下官方全文，版本来自实际abstract入口/全文标识，而非按年份猜测：

- [GPTQ v2](https://arxiv.org/html/2210.17323v2)，§3 Eq.1、§4 Algorithm1。逐层输出重建目标与正文一致。将Frobenius误差展开成含XXᵀ的trace成立；正文没有把未中心化二阶矩误称为协方差。构造输入(1,1)/(10,1)下输出误差0/0.9、平方0/0.81正确。GPTQ利用二阶信息补偿权重误差的归因有依据，教学搜索伪代码也明确不是GPTQ完整实现。
- [AWQ v6](https://arxiv.org/html/2306.00978v6)，§3.2 Eqs.4–5。等价变换WD与D⁻¹X、量化后输出误差搜索与通道统计的解释正确。scale改变组量化步长、保护某通道可能损害其他通道的限定符合原文。未将小例子冒称AWQ实测。
- [Megatron-LM v4](https://arxiv.org/html/1909.08053v4)，§3 Fig.3(a)。第一投影按输出列切，局部激活，第二投影按输入行切，最后对部分输出求和，矩阵shape与非线性位置正确。8192bytes、16MiB、四卡理想ring每卡发送1.5倍即12KiB/24MiB均正确。这里是MLP前向通信量，正文没有把它当整层训练的全部通信；前/反向与attention还有额外交换。
- [PagedAttention v1官方PDF](https://arxiv.org/pdf/2309.06180v1)，§2.3、§4。HTML入口本次失败，已改走PDF成功读取，未跳过。iteration-level调度与逻辑块/物理块映射是不同机制。块16、长度33需3块、末块余15位置正确；分页不消除有效KV元素，只降低分配浪费并支持共享。正文没有把这些逻辑推导当真实吞吐测量。

4-bit加每128权重一个FP16 scale的有效4.125bits/weight、7B理想3.609375GB正确。理想前向流水线p=4、m=8的bubble比3/11≈27.3%正确；已清楚限定不等于真实forward/backward调度。纯DP、TP、PP、EP和优化器分片的资源账边界合理。

## 09：GRPO与R1训练链

已真实读取[DeepSeekMath v3 §4.1、Eqs.3–5、Algorithm1](https://arxiv.org/html/2402.03300v3)和[DeepSeek-R1 v1 §2.2–2.4](https://arxiv.org/html/2501.12948v1)。

补文明确old rollout策略、当前策略与reference的不同角色。GRPO逐token比率、每回答先除有效长度再对组平均、KL直接进入目标的结构符合原始目标；prompt/图像/padding不计回答目标是清晰实现约定。总体标准差被明确选为教学约定，原文未指定ddof，因此未冒充唯一规范。

奖励[0,0,1,1]的总体标准差0.5、优势约±1，样本标准差时约±0.866；三种clip例子的min方向正确。全等reward的相对信号为0、KL仍可产生梯度的区别正确。RM分数差1.2对应概率0.7685、loss0.2633，手算正确。

k=exp(u)−u−1在当前策略分布、同一条件且支持集充分时的期望是forward KL；正文已限定old rollout多轮复用后的样本均值不能无条件叫精确KL。此处讨论数值估计的期望，不应进一步推成“固定样本自动微分就是精确KL梯度”。补文没有作这个额外保证。

R1四阶段与仅用生成样本做SFT的蒸馏小模型陈述有v1支持；没有把文本推理训练成绩当视觉grounding证据。参考策略“在已声明区间冻结”的表述也保留了原DeepSeekMath Algorithm1外层可刷新reference的可能。

## 11：过滤、采样与归一化

构造的1000条开发记录与OCR分层计数逐项相容。三阈值precision/recall、分层选择TP730/FP70/保留800、precision/recall91.25%及OCR93.75%均正确。它们明确为教学数据，不是DataComp实测。

qₖ∝nₖ^τ、单条qₖ/nₖ的概率与期望exposure正确：两来源1M/10K时，τ=1单条约9.901×10⁻⁷，τ=0时5×10⁻⁷/5×10⁻⁵；“来源间100倍”和“B相对自身原配方50.5倍”的分母区分正确。1,010,000次有放回抽取的期望、至少出现一次概率1−(1−p)^D正确。

全局token平均与先样本平均在给定100/1000有效token长度下，对B分别约10/11和1/2的记账权重正确；已限定不同于实际梯度范数。按q抽样再乘r/q恢复来源加权样本loss期望，要求q覆盖r支持集；给出的0.5/0.5与0.9/0.1满足该条件。实际读取[CS336 lecture14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py)的importance_sampling函数，p/q思想对应，未声称其具体实现与本伪代码完全相同。

## 13组结果与旧问题的修订核对

本轮不把元数据API的成功冒充结果复现。原始逐表/PDF视觉核验记录见 `evidence.md`；此处对其修订落盘作复核：

- MMMU ledger和第10章现为v4、B.1/Table4整体validation900题，34.4/36.4/56.8；120题Art & Design单独标明，不再误当overall。H1源文已解决。
- Chameleon现为v2、Tables11–12全部1048prompt，58.8/51.6，明确Gemini+/GPT-4V+均接DALL-E3、tie算半次。mixed-input441题60.4/47.3单独说明。H2源文已解决。
- CLIP三模型均224px；RT-2为unseen objects/backgrounds/environments条件；OSWorld12.24明确GPT-4+a11y tree；DataComp最终dataset size与候选池分开；13组均带实际vN与locator。M1–M5源文/ledger已解决。
- 其余11组已核查修订未改变原先通过的数值。Chinchilla摘要67.5/表67.6、Flash ±含义不能自称95%CI、POPE两prompt不能解释成纯长度因果等边界均保留。

## proposal与H3

完整读过 `paper/proposal.md`。现有算法是明确标注的未实现方案，不宣称训练过verifier、运行过模型、建立固定收益或新颖性。旧数字被明确撤回；没有用“设计稿”小字为虚构结果辩护。CLIP相似度不等于局部真假概率、POPE对象存在性不等于全面事实性、错误相关性须测量而非假定的边界成立。预算匹配、打乱证据、长度/拒答控制与图像cluster设计合理。

最终产物复核尚待执行：检查 `paper/index.html` 与原下载URL对应PDF不再出现已撤回的结果/消融/日志宣称，并确认13章网页和PDF与纠错源文一致。此项完成后将追加H3销项记录。

## 来源登记

已向 `core-reading.json` 仅追加5个新key：gptq(v2)、awq(v6)、megatron(v4)、paged_attention(v1)、deepseekmath(v3)，含真实标题、完整版本URL、用途与章节定位。R1原有key未改；本轮训练配方明确核对v1，应由主编将正文引用也锁到v1。DataCite身份交叉核验由主编统一执行，本审阅没有将尚未调用的新endpoint记为成功。

## 产物复核记录 1

已复读09/11整合后的正文，公式与上述核验稿一致。已实际查看 `10-evidence.png` 与 `12-evidence.png`：条形数值、baseline标签、MMMU900题与Chameleon1048题/tie定义均已正确重绘。

已对 `paper/index.html` 与 `assets/paper-verifier-gated-visual-decoding.pdf` 作实际内容复核。完整PDF文本中的虚构实验/消融/收益宣称已移除，+2.0、−5.9、1.3×、内部重跑、匿名仓库、we show、NeurIPS等旧宣称不存在；“没有训练”“没有运行大模型”“这些数字已撤回”在两产物均存在。PDF标题明确为“待验证的实验方案”。H3在本地待发布文件中已解决；尚未据此声称远程线上已经更新。

本次PDF SHA256：`4debfff240fdb6ccb540ae8c6df18b38c150f9eb2fb35039d39b8e6a1cc8cb5e`；HTML SHA256：`3b97eb1ca0c05c0454c1f77b2836f190f2c5b91c0e4f611b0a543e5fa03b1c55`。后续重建可能改变hash，应以最终内容复核为准。

10/12章PDF此时仍待主编重建，不能用已正确的PNG和HTML替代PDF核验；H1/H2仅剩该发布产物同步检查。

## 最终产物复核与销项

主编完成全部14份PDF重建后，本审阅重新实际读取下列文件的完整提取文本与HTML，并用Poppler渲染检查第10章PDF第7页、第12章第8页和proposal封面。没有用旧hash或源码替代最终文件。第10/12章PDF分别为8/9页，各有4张嵌入图；第10章此前含方括号的cluster图注现已保留，MMMU图作为图4出现，未再因图注解析而漏图。

**H1关闭。** 第10章最终PDF图4清楚显示34.4/36.4/56.8，图内来源为v4、B.1/Table4、900题；图注说明120题Art & Design及44.2/51.7/65.8属于另一子集。HTML同样正确，没有把子集重写为overall。900题在PNG来源脚注中可见，故仅检索PDF可提取文字未命中900不构成缺失；已视觉确认。

**H2关闭。** 第12章最终PDF图4与HTML均为全部1048prompt的58.8/51.6，Gemini+/GPT-4V+、DALL-E3和半次tie定义完整；441题mixed-input的60.4/47.3单独交代，与整体没有混用。PNG、图注、正文和ledger相容。

**H3关闭（本地最终发布文件）。** 重新完整审读proposal PDF及HTML后，旧虚构结果/消融/延迟/已存在日志宣称均未残留。“没有训练”“没有运行大模型”“这些数字已撤回”在两产物明确出现。PDF标题明确待验证；研究设计和实际已运行CPU教学探针没有被混为模型实验。尚未提交/部署的远端状态不在本次本地文件验收结论内。

最终文件SHA256如下，可用于确认本次审阅对象：

| 文件 | SHA256 |
|---|---|
| `assets/10-benchmarks-eval.pdf` | `379366e7ca6d626e821a6a8dab82c8560c5177794a0bba569729558a7fae51d9` |
| `reports/10-benchmarks-eval/index.html` | `72a2d03911cae6512fd3419e0c1368597dc033aa0d0176455ab4e7bc629deb44` |
| `assets/12-frontier.pdf` | `a9aa7943473774daf34b6cc666bc8d44e00fd1717035950949cc6e1b068a3619` |
| `reports/12-frontier/index.html` | `9e6c4a970c92876b7822410d4137094be13cbd3d60d0c065004c3859e939a500` |
| `assets/paper-verifier-gated-visual-decoding.pdf` | `c1e8da39485eea143b28424cac15536278148a093288bec8dab75854067dc0f0` |
| `paper/index.html` | `7398b32b1e187582c69284277f83aa9d79e716dc981ae0c4ed2cfd855bfb9b92` |

另检查主编提供的最终DataCite核验JSON：35个unique arXiv论文记录均有实际endpoint、返回标题与非空creators。它承担身份交叉核验，原表结果和机制由此前官方全文审阅承担，二者没有互相替代。

最终结论：本审阅发现的high/medium科学真实性问题已解决；本轮最终文件复核没有发现新的high/medium。13组主数字及其口径的纠错结论保持有效。读者看到的真实模型结果仍明确归作者报告，CPU教学计算与未执行研究方案分别标明。

## 最终产物补充说明

最后一轮专门复读 PDF 的 reviewer 因账户使用额度停止，不能视作完成。主线程随后实际复核全部 PDF：10章为34.4/36.4/56.8（整体validation900题）；12章为58.8/51.6（1048 prompts，tie半计，DALL-E 3增强基线）；研究方案明确撤回旧数字并声明尚未执行。最终PDF哈希和自动检查另保存在本轮qa.json；此处不将失败的子任务记作独立通过。

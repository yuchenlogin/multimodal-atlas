# 新增技术机制的第二轮证据核验

日期：2026-10-16。范围：`depth-expansions.md` 的新增技术补文及其整合正文；不是代码审查。只追加来源元数据与本审阅，不修改正文。DataCite新增身份交叉核验由主审统一执行；下列判定来自真实官方全文。

## 已核验部分

### 00 · RoPE与MoE

[RoFormer 2104.09864v5](https://arxiv.org/html/2104.09864v5) §3.2的成对旋转与相对内积构造支持补文。`R(m)^T R(n)=R(n-m)`、二维例子和共享位移不改变相位差的解释成立。cache的“仅旋转新K，不重旋旧K”是所选实现约定下的数学检查，正文没有冒充原论文推理基准。

[Switch Transformers 2101.03961v3](https://arxiv.org/html/2101.03961v3) §2.2 / Figure3明示容量公式、溢出token跳过专家分支经残差传递，以及auxiliary balancing。16token/4expert/capacity4的例子成立；FFN50.3M、总402.7M、激活约100.7M及16MiB逻辑双程流量均是正确手算。不同实现可以重路由或使用不同容量策略的限定必要且已经保留。

结论：未发现high/medium事实错误。需要保持“generic top-k例子”和“Switch top-1容量实例”的区分：Switch选中门值并非必须在top-1后重新归一为1；本补文将归一化作为通用示例，而没有把它署给Switch，故不作为错误。

### 01 · Q-Former三目标与冻结梯度

[BLIP-2 2301.12597v3](https://arxiv.org/html/2301.12597v3) §3.1–3.3 / Figures2–3支持32×768 queries、共享self-attention参数、ITC/ITG/ITM三种mask及两阶段训练。ITC query–text互不读取、ITG文字读queries与过去文字而query不读文字、ITM双向的四块mask正确。ITC max-query相似度、ITM平均query logits、阶段二视觉前缀接冻结LLM的描述均匹配。

0.731的两候选softmax、冻结权重仍需对LLM输入保留autograd、未来文字经query旁路泄漏的推理成立。未发现high/medium事实错误。该论文已有core key，不重复新增。

### 02 · CTC / RNN-T / EnCodec

[CTC作者原版ICML2006 PDF](https://www.cs.toronto.edu/~graves/icml_2006.pdf) §3.1、§4.1支持blank/重复collapse、路径求和与跳两格约束。目标aa/T3的唯一a-blank-a路径及概率0.144正确；初始化与末两格求和描述正确。元数据作者为Alex Graves、Santiago Fernández、Faustino Gomez、Jürgen Schmidhuber；年与会议由PDF版权/出版行确认，未用arXiv编号代替这篇来源。

[RNN-T 1211.3711v1](https://arxiv.org/html/1211.3711v1) §2.3–2.5，Equations16–17，明确使用`alpha(1,0)=1`及`P(y|x)=alpha(T,U)*P(blank|T,U)`。补文转移式和末端blank与该索引一致。label推进输出轴、blank推进声学轴，输出可长于输入的区别准确。原论文encoder是双向；补文也明确loss本身不保证encoder流式。

[EnCodec 2210.13438v1](https://arxiv.org/html/2210.13438v1) §3.2明确STE、码本EMA（原文decay0.99）与失效码替换；§3.4 Equation3是各残差层commitment求和，梯度只到量化器输入。补文的`z+sg(q-z)`、残差commitment、EMA计数/向量和与optimizer路径分开均成立。0.005重建平方误差、1.9新计数及(1.0105,−0.0158)EMA示例正确；其中0.9明确为教学取值，不是原文配置。未发现high/medium事实错误。

### 03/04 · 物理时间与流式可用性

[Qwen2.5-VL 2502.13923v1](https://arxiv.org/html/2502.13923v1) §2.1.2–2.1.3支持动态FPS及绝对时间MRoPE。[Qwen2.5-Omni 2503.20215v1](https://arxiv.org/html/2503.20215v1) §2.2明确每时间ID为40ms、text/audio三轴相同ID、图像空间ID与实际视频时间间隔；2秒片内视觉先于音频。§2.4明确2秒audio块及输出DiT的2lookback/1lookahead。补文没有将其等同完整产品双工策略。

round映射、三轴分组和最大ID后接文字被明确标为教学接口；不能删去这项限定后当成checkpoint processor实现。FN²+NF²与(FN)²的配对数量、80ms lookahead例及6400/24000秒播放前缀均计算正确。位置时间、数据到达时间和输出提交的分离具有清晰条件。未发现high/medium事实错误。

### 05/06 · Flow、采样与动作分布

[Flow Matching 2210.02747v2](https://arxiv.org/html/2210.02747v2) §3 / Theorems1–2支持由条件向量场边际化及CFM训练；§4 Equations20–23给线性高斯条件路径。补文用η/z/τ避免DDPM的x0方向混淆，条件平方损失的最优平均速度、条件直线不等于边际ODE直线的区分正确。

[DDIM 2010.02502v4](https://arxiv.org/html/2010.02502v4) §4的随机系数为0的跳步更新，与补文一致。DDIM手算1.1340/1.2072以及Euler1.5、Heun1.625、精确1.6487均正确。NFE与额外CFG分支的开销没有被混为零成本。

[Diffusion Policy 2303.04137v5](https://arxiv.org/html/2303.04137v5) §2.2–2.3、§4.1、§4.3支持动作序列去噪、多峰动作分布及滚动时域执行。flow动作头被作为由前面数学构造的教学方案，未冒称Diffusion Policy原文使用flow。H16/k4/20Hz/80ms推理的同步280ms与3.57Hz核算正确；异步响应280ms也是在写明调度假设下的上界例。

结论：没有high/medium错误；有下列一项low补充建议。

## 可疑点与建议

### L1 · low · FM的零终点噪声路径是教学极限

位置：`depth-expansions.md:296–307`，以及整合后05的CFM定义。

原FM论文线性条件路径含`σ_min>0`以保留非退化终点；补文`xτ=(1−τ)η+τz`是`σ_min→0`的直线路径特例。在τ<1上公式与平方回归解释成立，不能据此说错，但在讲“论文核心构造”时宜注明“这里采用零终点噪声的教学极限；实际实现可保留σ_min>0”。已在新增core用途字段保留此边界。无需改变本章手算或宣称新增模型实验。

## 来源登记与proposal

已只追加新key到`core-reading.json`：rope(v5)、switch(v3)、rnnt(v1)、qwen25vl(v1)、qwen25omni(v1)、flow_matching(v2)、ddim(v4)、diffusion_policy(v5)。版本从实际官方HTML的abstract/PDF入口读取，两个入口一致；没有猜版本。CTC已登记到`primary/other-sources.json`，含官方作者托管URL、完整标题、作者、年与位置。

`paper/proposal.md:50`已改为“可能产生相关错误”，相关性不再被写成未经测量的确定事实；本项通过。

AWQ / Megatron / GRPO对应补文尚在写入，已取得官方全文，待其落盘后继续逐式复核并追加来源。

# 从 Transformer 到多模态基座：现代大模型的架构基础

> 副标题：拆解 2017–2026 年间构成现代多模态基座模型的八个核心基件——Transformer 骨架、位置编码、注意力效率变体、MoE、视觉骨干、预训练范式、扩展定律与对齐技术
> 版本日期：2026-09-29 | 面向读者：有深度学习基础的多模态初学者

## 摘要

本篇是整个多模态大模型报告系列的第 0 篇，也是地基篇。今天的多模态基座模型——无论是 LLaVA 式的视觉-语言模型、支持音频视频的全模态模型，还是视觉-语言-动作（VLA）具身模型——其内部并没有发明什么全新的骨架，而是复用了一整套在纯文本大模型时代被反复锤炼过的架构组件。本篇系统梳理这套"基件库"：（1）Transformer 骨架，包括自注意力、多头机制、前馈网络、残差连接与归一化的设计动机与数学形式；（2）位置编码的演进谱系，从正弦位置编码、可学习位置嵌入到 RoPE 与 ALiBi；（3）注意力效率变体，包括 MQA/GQA/MLA 的 KV cache 压缩思想与 FlashAttention 的 IO 感知计算范式；（4）混合专家（MoE）如何用稀疏激活解耦参数量与计算量；（5）Vision Transformer（ViT）如何确立"视觉即 token 序列"的编码范式；（6）BERT 式掩码语言建模与 GPT 式因果语言建模两大预训练范式的分野；（7）从 Kaplan 幂律到 Chinchilla 计算最优定律再到多模态扩展的 scaling law 方法论；（8）RLHF、DPO 与 GRPO 构成的对齐技术栈。每个基件我们都回答三个问题：它解决了前人的什么痛点、它的数学本质是什么、它在多模态模型中被如何复用。本篇结尾将把这些基件拼回完整图景，为后续各专题篇（视觉-语言、音频、视频、全模态、生成、具身等）提供统一的概念坐标系。

## 1. 引言：本方向在多模态版图中的位置

### 1.1 为什么"第 0 篇"要回到纯文本时代

一个常见误解是：多模态大模型是一类与语言模型平行发展的新物种。事实恰恰相反——2017 年 Transformer 诞生以来的几乎全部架构进步，首先发生在纯文本语言模型上，随后被原封不动或稍作修改地搬运到多模态领域。LLaVA 用大语言模型当"大脑"；Qwen-VL、InternVL 等模型直接把 ViT 编码器接到 LLM 的词表接口上；视频模型沿用时序 token + 因果注意力；具身模型把动作离散化成 token 走同一条自回归管线。**理解现代多模态模型的最低成本路径，就是先彻底吃透语言模型的架构基件，再学习"接入层"如何把这些基件扩展到新模态。**

本篇对应整个系列报告的学习地图最底层。所谓"最基础、最扎实"，体现在两个标准上：一是本篇的每个概念都给出完整数学形式而非名词解释——attention、RoPE 旋转、MoE 门控、PPO/DPO 损失，这些公式在后续每一篇都会不加推导地复用；二是本篇的每个方法都追问"它解决了前人的什么痛"——只有理解了痛点，才能在后续遇到新模态变体时判断它是真创新还是旧酒新瓶：

```
先修概念（读者已具备）
  ├─ 反向传播、梯度下降、交叉熵损失
  ├─ CNN / RNN / LSTM 基本结构
  └─ 词嵌入（word embedding）思想
        │
        ▼
本篇（第 0 篇）：架构基础
  Transformer 骨架 → 位置编码 → 注意力效率 → MoE
  → ViT 视觉骨干 → 预训练范式 → Scaling Laws → 对齐技术
        │
        ▼
后续专题篇
  01 视觉-语言（connector、投影、指令微调）
  02 音频语音 / 03 视频 / 04 全模态
  05 生成（扩散 vs 自回归）/ 06 具身（VLA）
  07 Agent 与工具 / 08 效率 / 09 对齐安全
  10 评测 / 11 数据与训练 / 12 前沿展望
```

### 1.2 本方向的核心问题

本篇贯穿三条核心问题线索：

1. **表达与效率的张力**：自注意力赋予任意两位置 $O(1)$ 的交互路径，代价是 $O(n^2)$ 的计算与存储。这条线索驱动了 MQA/GQA/MLA（压缩推理期 KV cache）、FlashAttention（压缩访存）、滑窗与稀疏注意力（压缩计算图）、乃至 linear attention / 状态空间模型（换掉注意力本身）等一系列工作。
2. **规模与数据的张力**：模型变大几乎总是变好，但算力与数据都是稀缺资源。给定预算，参数与数据如何配比？规模之外何时出现质变？这条线索驱动了 Kaplan 幂律、Chinchilla 计算最优、涌现能力研究与多模态 scaling 扩展。
3. **能力与人意的张力**：next-token prediction 训练出的是"互联网文本续写机"，不是"助手"。如何让强大但野生的模型变得有用、诚实、无害？这条线索驱动了 SFT → RLHF → DPO → RLVR 的整段后训练史。

三条线索共享一个隐含主题：**大模型研究的本质是在多重约束（算力、数据、显存、带宽、人类偏好）下做受约束优化**，每一个经典基件都是对某个具体约束的响应。

这三条线索在 2024–2026 年汇聚于多模态：多模态模型把序列变得更长（图像几千 token、视频几万 token），效率问题更尖锐；多模态数据远少于文本，scaling 的数据约束更紧；多模态输出的安全性与有用性，对齐技术同样直接复用。

### 1.3 时间线总览

![2017–2026 多模态模型时间线](../../assets/fig_timeline_vlm.png)

上图给出了 2017–2026 年间的代表性模型时间线。可以看到一个清晰模式：文本侧的架构创新（Transformer、RoPE、GQA、MoE、FlashAttention）通常领先多模态应用 1–3 年，随后被多模态模型批量吸收。本篇讲述的正是时间线左侧这些"被吸收的基件"。

## 2. 基础知识

### 2.1 从 RNN 到注意力：为什么需要 Self-Attention

在 Transformer 之前，序列建模的主力是 RNN/LSTM。RNN 的根本缺陷有二：其一，计算必须沿时间步串行进行，第 $t$ 步的隐状态依赖第 $t-1$ 步，无法并行——这意味着序列长度直接乘在训练时间上，GPU 的并行算力被结构性浪费；其二，信息必须逐步传递，长距离依赖在反复的非线性压缩中衰减（梯度消失问题的另一种表现），LSTM 的门控机制缓解了但并未根除这个问题。CNN 系的方案（如 ConvS2S）可以并行，但感受野随层数线性增长，建模全局依赖需要堆很多层。注意力机制（attention）最初是作为 RNN 编码器-解码器之间的辅助对齐工具出现的——解码每个词时"回头看"源句的哪些词最相关。而 Vaswani 等人在 2017 年提出了一个激进的观点：**既然注意力已经足够强，为什么不抛弃循环与卷积，只用注意力？** 这就是《Attention Is All You Need》[1]。回看历史，这个"激进"判断的赌注在于：它赌硬件的并行红利会补偿注意力 $O(n^2)$ 的复杂度劣势——在当时序列长度只有几十上百的机器翻译场景，这个赌注稳赢。

### 2.2 自注意力（Self-Attention）：数学形式

给定输入序列的 token 表示矩阵 $X \in \mathbb{R}^{n \times d}$（$n$ 为序列长度，$d$ 为模型维度），通过三组可学习投影矩阵 $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$ 得到查询（query）、键（key）、值（value）：

$$Q = XW^Q,\quad K = XW^K,\quad V = XW^V$$

缩放点积注意力（scaled dot-product attention）定义为：

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

逐行看：第 $i$ 个 token 的输出是所有 token 的 value 向量的加权和，权重由它的 query 与每个 key 的点积相似度经 softmax 归一化得到。这里有两个关键设计决策：

- **为什么用点积**：点积可整块用矩阵乘法实现，高度并行，充分利用 GPU 的矩阵乘单元；
- **为什么除以 $\sqrt{d_k}$**：当 $d_k$ 较大时，点积的方差随维度线性增长，softmax 会进入梯度极小的饱和区。除以 $\sqrt{d_k}$ 使注意力分数的方差维持在 $O(1)$，这是原作者基于"query/key 分量独立同分布、均值 0 方差 1"假设推出的缩放因子 [1]。

自注意力一次性建立了任意两个位置之间的直接通路——路径长度为 $O(1)$，彻底解决了 RNN 的长距离依赖衰减问题；代价是 $QK^\top$ 矩阵带来 $O(n^2 d)$ 的计算与 $O(n^2)$ 的注意力矩阵存储，这是后面所有效率变体要对付的"原罪"。

一个有助直觉的类比：把每个 token 想成图书馆里的读者，query 是"我想找什么资料"，key 是"我能提供什么资料"的索引卡，value 是资料本身。每个读者拿自己的 query 去匹配所有人的索引卡，按匹配度加权取回资料——一轮操作后，每个位置都聚合了全序列的相关信息。RNN 里信息要"口口相传"，距离 $k$ 的两个位置需要 $k$ 步传递且每步都有损耗；注意力则相当于一次全员大会，代价是会议规模随人数的平方增长。

**因果掩码（causal masking）**：用于自回归语言建模时，第 $i$ 个位置不允许"看到"未来，实现上在 softmax 之前把 $QK^\top$ 的上三角（$j > i$ 的部分）置为 $-\infty$。这一改动把 Transformer 从双向编码器变成了从左到右的生成器，是 GPT 系列的起点（见 2.8 节）。

### 2.3 多头注意力（Multi-Head Attention）

单一注意力分布只能在一个表示子空间里计算相似度。多头注意力把 $Q, K, V$ 分别切成 $h$ 份（每份维度 $d_k = d/h$），并行计算 $h$ 组注意力再拼接投影：

$$\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\, W^O,\qquad \mathrm{head}_i = \mathrm{Attention}(XW_i^Q, XW_i^K, XW_i^V)$$

直觉上，不同 head 可以学习不同性质的关系（语法依存、指代、位置邻近等）——后续的可解释性研究确实在特定 head 中观察到了可辨识的句法与语义角色。多头的计算量与单头（同总维度）几乎相同，却显著增强了表达能力。原始 Transformer 的 base/big 配置为 $d=512/1024$、$h=8/16$ [1]；当代大模型如 7B 量级常用 $d=4096, h=32$。

一个实用的比例经验：head 维度 $d_k$ 通常固定在 64–128，扩大模型时优先加 head 数而非加大 $d_k$。过小的 $d_k$ 让每个 head 的"相似度判断"欠参数化，过大的 $d_k$ 则浪费并行的多样性。这个看似琐碎的工程约定，在 2.7 节的 KV cache 讨论中会突然变得重要——head 数与 $d_k$ 的乘积直接决定缓存体积。

### 2.4 前馈网络（FFN）与激活函数

每个 Transformer 块里，注意力之后是一个逐位置（position-wise）两层 MLP：

$$\mathrm{FFN}(x) = W_2\, \sigma(W_1 x + b_1) + b_2$$

隐层维度通常为 $4d$（原论文 $d_{ff}=2048$ 对 $d=512$）[1]。FFN 占模型参数量的约 2/3，被认为是模型的"知识存储器"，而注意力负责"信息搬运"。

激活函数经历了 ReLU → GELU → SwiGLU 的演进。当代开源模型的标准配置是 **SwiGLU**：

$$\mathrm{SwiGLU}(x) = \bigl(\mathrm{Swish}(xW_1) \odot xW_3\bigr) W_2,\quad \mathrm{Swish}(x) = x \cdot \mathrm{sigmoid}(x)$$

门控结构（element-wise 乘一个线性分支）带来更平滑的优化景观；为保持参数量不变，隐层维度通常从 $4d$ 调整到约 $8d/3$。SwiGLU 是 LLaMA、Qwen、Mistral、DeepSeek 等现代模型的共同选择 [13, 19, 22]。

### 2.5 残差连接与归一化：Post-Norm 到 Pre-Norm

深网络训练的两个敌人是梯度消失与表示漂移。Transformer 用残差连接 $x + \mathrm{SubLayer}(x)$ 保证梯度有直通路径，用层归一化（LayerNorm）把每层输入拉回稳定分布。二者的组合方式有重要分野：

- **Post-Norm**（原始 Transformer）：$x \leftarrow \mathrm{LayerNorm}(x + \mathrm{SubLayer}(x))$。归一化施加在残差相加之后，意味着"主干"上的表示每过一层就被重新归一一次；浅层时这有利于表示规整，但层数加深后，反向传播的梯度要穿过每一层的 LayerNorm 雅可比，深层模型在训练初期极易出现梯度爆炸/消失，必须配合精心设计的学习率 warmup 才能启动。
- **Pre-Norm**：$x \leftarrow x + \mathrm{SubLayer}(\mathrm{LayerNorm}(x))$。归一化只作用在分支内部，残差主干从第一层到最后一层是纯粹的恒等累加——反向传播时梯度沿主干无损直达任意深度。这就是为什么 GPT-3 时代的 96 层、PaLM 的 118 层能够稳定训练：Pre-Norm 把"深度"从优化难题变成了简单的超参数。代价是理论上表示能力略弱（主干缺乏正则），实践中无人介意。

归一化层本身也在简化：**RMSNorm** 去掉了 LayerNorm 的均值中心化，只做均方根缩放：

$$\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot g$$

实验表明中心化贡献有限而计算可省约一半归一化开销。RMSNorm + Pre-Norm + SwiGLU + RoPE 构成 2023 年后开源大模型的"现代配方"四件套 [13, 22]。

### 2.6 位置编码：让模型知道"谁在哪"

自注意力本身是**置换等变**（permutation-equivariant）的：打乱输入顺序，输出也仅被打乱。但语言里顺序即意义，必须显式注入位置信息。

**（1）正弦绝对位置编码（Sinusoidal）** [1]：为每个位置 $pos$ 构造一个确定性的向量，维度 $2i$ 与 $2i+1$ 分别为

$$\mathrm{PE}(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d}}\right),\qquad \mathrm{PE}(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

不同频率的正弦/余弦构成一组"位置基"，任意偏移 $k$ 后 $\mathrm{PE}(pos+k)$ 是 $\mathrm{PE}(pos)$ 的线性函数，理论上便于学习相对位置。优点是零参数、可外推到任意长度；实践中直接加到 token 嵌入上。

**（2）可学习位置嵌入（Learned PE）**：BERT、GPT-2、ViT 给每个位置一个可训练向量。效果略优于正弦编码，但训练时见过最长 $n_{\max}$ 个位置，推理时无法超出——外推性差是其致命伤。

**（3）相对位置编码与 ALiBi** [10]：与其编码绝对位置，不如直接在注意力分数里加入与相对距离有关的偏置。**ALiBi（Attention with Linear Biases）** 把它推到极致：

$$\mathrm{score}(q_i, k_j) = \frac{q_i \cdot k_j}{\sqrt{d_k}} - m_h \cdot |i - j|$$

每个 head 用一个固定的斜率 $m_h$（按 $2^{-8h/H}$ 几何数列设置），距离越远惩罚越大。ALiBi 完全不加位置向量，却能实现出色的长度外推：论文报告 1.3B 模型在 1024 长度上训练，可外推到 2048 长度且困惑度与训练到 2048 的正弦位置模型相当 [10]。缺点是线性偏置表达能力有限，长距离信息被硬性衰减。

**（4）旋转位置编码 RoPE（Rotary Position Embedding）** [8]：当代大模型的事实标准。核心思想是：把 $d$ 维向量两两分组，对位置 $m$ 的 query/key 向量施加一个与 $m$ 成正比的二维旋转。形式化地，将嵌入维度两两配对为 $(x_{2i}, x_{2i+1})$，定义旋转矩阵

$$R_{\Theta, m} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix},\qquad \theta_i = 10000^{-2i/d}$$

对位置 $m$ 的 query 与位置 $n$ 的 key 分别旋转后做点积，利用旋转矩阵的性质 $R_{\Theta,m}^\top R_{\Theta,n} = R_{\Theta, n-m}$，注意力分数只依赖**相对位置** $n-m$：

$$\langle R_{\Theta,m}\, q,\; R_{\Theta,n}\, k \rangle = \langle q,\; R_{\Theta,\,n-m}\, k \rangle$$

这实现了"以绝对方式实现、以相对方式生效"：实现上只是对 $Q, K$ 各做一次逐位置的旋转（可以用复数乘法高效计算——把每对维度看作一个复数，位置 $m$ 的旋转就是乘以 $e^{im\theta_i}$），效果上注意力天然感知相对距离，且远程依赖随距离自然衰减（RoPE 论文证明的长程衰减性质）[8]。

值得玩味的是各维度的频率分工：低维（$i$ 小）对应的 $\theta_i$ 大、旋转快，负责编码近距离的精细位置差；高维旋转慢、周期长达上万，负责粗粒度的远程结构。这种"多尺度时钟"结构与原始正弦编码一脉相承——RoPE 可以看作把正弦编码从"加在嵌入上"改为"乘在 Q/K 上"，从而让相对性成为内禀性质而非需要学习的巧合。RoPE 的主要弱点是训练长度外的外推能力差，由此催生了位置插值（Position Interpolation）、NTK-aware 缩放与 **YaRN**（按频率分组做差异化插值，少量微调即可数倍扩展上下文窗口）[18] 等一系列长上下文技术。2026 年的位置编码综述把这条谱系总结为：absolute → relative → RoPE → 长上下文 scaling [27]。

### 2.7 注意力效率变体：MQA / GQA / MLA 与 FlashAttention

自回归推理时，每生成一个 token 都要用它与**所有历史 token** 做注意力。为避免重复计算，历史 token 的 $K, V$ 被缓存下来（KV cache）。KV cache 的大小为 $O(n \cdot h \cdot d_k \cdot L)$（$L$ 为层数），长序列 + 大 batch 下显存带宽成为推理瓶颈——这正是 MQA/GQA/MLA 的靶心。

先建立一个关键的数量级直觉：解码阶段每个新 token 都要从 HBM 读入**全部**历史 KV cache，而矩阵-向量乘的算术强度极低——解码是纯粹的访存受限（memory-bound）操作。也就是说，推理速度几乎与 FLOPs 无关，正比于 KV cache 的字节数。于是"压缩 KV cache"与"加速推理"变成了同一件事，这就是下面三种方案的统一逻辑，差别只在压缩手法。

- **MQA（Multi-Query Attention，2019）** [4]：所有 query head 共享同一组 $K, V$。KV cache 缩小 $h$ 倍，解码显著加速；代价是质量下降。
- **GQA（Grouped-Query Attention，2023）** [16]：把 $h$ 个 query head 分成 $g$ 组，每组共享一组 $K, V$。$g=1$ 退化为 MQA，$g=h$ 退化为 MHA。论文表明 GQA-8 的质量几乎与 MHA 持平、推理速度接近 MQA，并可通过对已有 MHA 检查点用 5% 预训练算力"uptraining"转换而来 [16]。GQA 如今是 Llama 2/3、Mistral、Qwen 等开源模型的标配。
- **MLA（Multi-head Latent Attention，DeepSeek-V2/V3，2024）** [21, 23]：不再共享 K/V head，而是把每个 token 的 $K, V$ 联合低秩压缩到一个潜在向量 $c_t \in \mathbb{R}^{d_c}$（$d_c \ll h \cdot d_k$），推理时只缓存 $c_t$（及 RoPE 所需的少量解耦 key），使用时再升维重构：

$$c_t = W^{DKV} x_t,\qquad k_t = W^{UK} c_t,\quad v_t = W^{UV} c_t$$

由于 $W^{UK}$ 可与 $W^Q$ 在推理时合并（矩阵乘结合律），升维几乎不产生额外计算。DeepSeek-V2 报告 KV cache 减少 93.3%、最大生成吞吐提升至 5.76 倍 [21]。MLA 代表了从"删 head"到"压表示"的思路升级：缓存的是信息的低秩瓶颈而非删减后的原表示。

- **FlashAttention（2022）** [12]：与前三个改架构的方法不同，FlashAttention 是**不改变数学结果的系统层优化**。标准实现把 $n\times n$ 注意力矩阵整体物化到 GPU 高带宽显存（HBM），读写开销 $O(n^2)$；FlashAttention 利用 GPU 片上 SRAM 分块（tiling）计算注意力，配合在线 softmax（online softmax，逐块维护归一化常数）避免物化整个矩阵，HBM 访问量从 $O(n^2)$ 降到 $O(n^2 d^2 / M)$（$M$ 为 SRAM 大小），显存复杂度降为线性，并带来端到端加速（论文报告 BERT-large 训练 15% 墙钟时间加速）[12]。FlashAttention-2 进一步优化并行划分（沿序列维度并行而非仅 batch×head）与 warp 分工（减少共享内存读写与同步），速度再翻倍 [17]。

一个常见误解需要澄清：FlashAttention 并没有降低注意力的 FLOPs——$QK^\top V$ 的乘加次数一次不少。它优化的是**内存墙**：现代 GPU 的算力增长远快于显存带宽，许多 kernel 的实际瓶颈是把数据在 HBM 与 SRAM 之间来回搬运。FlashAttention 把 $O(n^2)$ 的中间矩阵读写从 HBM 挪进片上 SRAM，于是同样的数学、同样的精度，换来数倍的墙钟加速和线性的显存占用。它的深远影响在于确立了一个方法论：**架构创新必须与硬件内存层级协同设计**，这一视角在 2025 年 DeepSeek 团队的硬件复盘文中被系统化 [25]。今天，"是否 FlashAttention 友好"事实上成为新注意力变体能否被社区采纳的隐形门槛。

### 2.7.1 混合专家（MoE）：用稀疏激活解耦参数与计算

稠密（dense）模型的每个 token 都流经全部参数，参数量与计算量死死绑定——想把模型做大以容纳更多知识，就必须等比例付出训练和推理 FLOPs。**混合专家（Mixture-of-Experts, MoE）** 打破这个绑定：把每层 FFN 复制成 $N$ 份"专家"，配一个轻量路由网络（router/gating network），每个 token 只被分派给 $k \ll N$ 个专家。参数量扩大 $N$ 倍，每 token 的计算量只扩大 $k$ 倍。

形式上，给定 token 表示 $x$，路由器产生专家打分并经 softmax 得到门控权重：

$$p_i(x) = \mathrm{softmax}\bigl(W_r \, x\bigr)_i,\qquad \mathrm{MoE}(x) = \sum_{i \in \mathrm{TopK}(x)} g_i(x)\, \mathrm{FFN}_i(x)$$

其中 $g_i(x)$ 为 top-$k$ 选中专家的（重归一化）门控权重。Shazeer 等人 2017 年的稀疏门控 MoE [2] 首次在 LSTM 堆叠中验证了这一思想，并在打分中加入可学习噪声以促进负载均衡。但朴素 top-k 路由有一个工程灾难：如果路由坍缩到少数专家（"赢者通吃"），其余专家得不到训练、被选中专家又超出单卡容量，整个系统崩溃。围绕**负载均衡**的技术演进构成 MoE 史的主线：

- **辅助均衡损失**：Switch Transformer（top-1 路由，把路由简化到极致，证明万亿参数可训练）[9] 与 GLaM（1.2T 参数、top-2 路由、等算力下优于 GPT-3 级 dense 模型）[11] 都使用辅助损失惩罚专家负载的不均匀：$\mathcal{L}_{aux} = \alpha \cdot N \sum_i f_i \cdot P_i$，其中 $f_i$ 是分派给专家 $i$ 的 token 比例、$P_i$ 是路由分配给它的平均概率。代价是辅助损失与主损失存在目标冲突，系数 $\alpha$ 需要小心调节。
- **容量因子（capacity factor）**：每个专家每步最多处理 $C = c \cdot (T/N)$ 个 token（$T$ 为 batch 内 token 总数），超出的 token 被丢弃（dropped）由残差连接旁路——这提醒我们 MoE 本质上是"有损"的稀疏化。
- **DeepSeek 路线的两次升级**：DeepSeekMoE [31] 提出**细粒度专家切分**（把专家切得更小、选得更多，如 64 选 8 而非 8 选 2）加**共享专家**（一个始终激活的通用专家捕获公共知识，让路由专家专注差异化），逼近专家专业化的上界；DeepSeek-V3 [23] 进一步提出**无辅助损失的负载均衡**——不给主损失加干扰项，而是给路由打分加逐专家的动态偏置项 $b_i$，过载专家的偏置自动下调、欠载的上调（只影响路由选择、不影响加权权重），从机制上消除了辅助损失的目标冲突 [23]。
- **开源侧的代表**：Mixtral 8×7B [20] 每层 8 个专家、top-2 路由，每 token 可访问 47B 参数但推理只用 13B 激活参数，质量对标 GPT-3.5 [20]——它向社区证明了稀疏 MoE 在开源预算下的可行性。

系统层面，MoE 的代价藏在通信里：专家通常按**专家并行**（expert parallelism）分布在不同 GPU 上，每个 token 被路由到哪个专家，就要把它的激活值通过 all-to-all 通信发到对应卡上，算完再发回。负载不均在这里是双重打击——既浪费训练信号，又让最快的卡等最慢的卡。DeepSeek-V3 能把 671B MoE 的训练控制在 2.788M H800 小时 [23]，一半功劳在架构，另一半在与 all-to-all 通信深度协同的流水线设计。

MoE 对多模态的意义超出效率本身：不同模态、不同任务天然形成"专家分工"的想象空间，多模态 MoE（按模态路由、模态内细粒度路由）是 2025 年后活跃的研究方向。

### 2.8 预训练范式：MLM 与 CLM 的分野

Transformer 块是通用的，差别在于训练目标：

- **掩码语言建模（MLM，BERT，2018）**：双向编码器，随机遮盖 15% token 并预测之。表征质量高，适合理解类任务；但预训练-微调不一致（推理时没有 [MASK]），且天然不适合生成。
- **因果语言建模（CLM，GPT 系列）**：单向解码器，标准 next-token prediction：给定 $x_{<t}$ 最小化 $-\log P(x_t \mid x_{<t})$。目标与生成完全一致，配合 scaling 展现出惊人通用性。

2019–2021 年间曾有 encoder-decoder（T5）、prefix-LM 等多种变体之争，争论的焦点是：双向上下文带来的表征质量，值不值得牺牲与生成的统一性？历史给出了裁决——随着 GPT-3 证明 CLM + 大规模可以"通吃"理解与生成（in-context learning 让理解任务也能被表述为条件生成），纯解码器 CLM 成为大模型唯一主流范式。这个裁决的逻辑值得记取：**不是双向编码输了，而是"统一接口"赢了**——当一切任务都能写成 next-token prediction，scaling 的红利就能无差别地灌溉所有任务。BERT 式双向编码并未消亡——它活在 ViT 的视觉预训练（如 MAE 的掩码图像建模）与 CLIP 类视觉编码器里，这正是多模态时代两条范式重新汇合的地方。现代预训练在这个骨架上发展出一系列工程实践：**填充目标**（FIM，fill-in-the-middle——把一段文本中间挖空、让模型在已知前后缀的条件下补全，训练时通过特殊 token 重排语序，使模型获得"补全"而非只会"续写"的能力，对代码模型尤其关键）；**多阶段训练与退火**（annealing——预训练末期混入高质量数据并快速衰减学习率，以低成本换取显著的知识注入）；**数据配比课程**（先通识后专业、先短后长的 curriculum）。这些技巧的细节属于第 11 篇（数据与训练）的范畴，但它们共享一个前提：CLM 目标的简单性使一切数据操作都可以安全地叠加在其上。

### 2.9 Scaling Laws：规模的可预测性

大模型时代最深刻的方法论发现是：**损失随规模的变化是平滑、可预测的幂律**。

**Kaplan 定律（2020）** [5]：OpenAI 团队发现交叉熵损失 $L$ 随参数量 $N$、数据量 $D$、算力 $C$ 呈幂律，跨越七个数量级依然成立；且大模型的样本效率显著更高，因此给定算力应优先把预算投给参数（"大模型优先"），在远未收敛时停止训练。这一结论直接指导了 GPT-3（175B 参数、仅 300B token）的设计。

**Chinchilla 定律（2022）** [6]：DeepMind 训练了 400 余个 70M–16B 参数的模型、覆盖 5B–500B token，用三种独立方法拟合，得出相反结论：**此前的大模型普遍训练不足（undertrained）；计算最优要求参数量与 token 数同比扩展——大约每参数 20 token**。据此训练的 70B 模型 Chinchilla（1.4T token）在与 280B 的 Gopher 相同算力下全面反超，MMLU 达 67.5%（较 Gopher 提升超过 7 个百分点）[6]。Chinchilla 改写了整个行业的训练配方，LLaMA 系列"小模型 + 超量数据"的策略正是其直接后裔 [13]。

**数学形式**：把两条定律写成可操作的式子。Kaplan 发现损失对参数量 $N$、数据量 $D$、算力 $C$ 分别满足 $L \propto N^{-\alpha_N}$、$L \propto D^{-\alpha_D}$（指数在 $0.05$–$0.1$ 量级，跨越七个数量级仍成立）[5]；Chinchilla 则用三方拟合给出联合形式

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

其中 $E$ 是不可约损失（数据的内在熵），后两项分别是"模型不够大"与"数据不够多"的惩罚项。给定算力预算 $C \approx 6ND$（Transformer 训练 FLOPs 的经典近似），对 $N, D$ 做约束优化即可解出计算最优分配 $N^*(C) \propto C^{a}$、$D^*(C) \propto C^{b}$，且拟合结果显示 $a \approx b \approx 0.5$——这就是"同比扩展"的定量含义 [6]。理解这个框架比记住"20 token/参数"更重要：它是**用可承受的小规模实验外推不可承受的大规模决策**的方法论范本。

**涌现现象的另一面**：与平滑幂律并存的是"涌现"（emergent abilities）——某些能力（如多步算术、思维链推理）在模型规模跨越阈值前接近随机水平、之后突然可用 [30]。后续研究表明部分"涌现"是评测指标的伪影（不连续指标把平滑进步显示成跳变），但它准确刻画了一个实践事实：scale 不只是量变。这场争论至今未完全平息，也是 scaling 文献中少有的概念性争议。

**多模态扩展**：把 scaling law 推广到多模态时，数据侧出现新问题——高质量图文对、视频-文本对的规模远小于纯文本，**数据受限（data-constrained）** 条件下的 scaling（重复数据的边际收益递减）、混合模态的配比最优解成为前沿问题。加上 2024 年后 test-time compute（推理时扩展，如长思维链）的兴起，"scale"一词已从单一的预训练算力扩展为"预训练 + 数据 + 推理"三维资源分配问题。

### 2.10 对齐技术栈：RLHF / DPO / GRPO

预训练让模型"什么都会"，对齐让它"按人的意图做"。

**RLHF 三段式（InstructGPT，2022）** [7]：

1. **SFT（监督微调）**：用标注员撰写的示范回答微调预训练模型；
2. **RM（奖励模型）**：对同一 prompt 的多个模型回答做人工排序，训练一个标量奖励模型 $r_\theta(x, y)$，常用 Bradley-Terry 偏好模型：

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\bigl(r_\theta(x, y_w) - r_\theta(x, y_l)\bigr)\right]$$

3. **PPO 强化学习**：以 SFT 模型为初始策略 $\pi_\phi$，最大化期望奖励同时用 KL 散度防止跑偏：

$$\max_{\phi}\; \mathbb{E}_{y \sim \pi_\phi(\cdot|x)}\bigl[r_\theta(x, y)\bigr] - \beta\, \mathrm{KL}\bigl(\pi_\phi \,\|\, \pi_{ref}\bigr)$$

PPO（Proximal Policy Optimization）[3] 用 clipped surrogate objective 约束每步策略更新幅度：

$$\mathcal{L}^{CLIP}(\phi) = \mathbb{E}_t\left[\min\bigl(\rho_t \hat{A}_t,\; \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\bigr)\right],\qquad \rho_t = \frac{\pi_\phi(a_t|s_t)}{\pi_{\phi_{old}}(a_t|s_t)}$$

InstructGPT 的标志性结果是：1.3B 参数的 InstructGPT 在人类偏好评估中优于 175B 的 GPT-3——**对齐带来的"可用性"增益超过 100 倍参数规模的差距** [7]。

这套管线也暴露了三个后来反复被研究的结构性问题：其一，**奖励模型是真实偏好的有损代理**，策略对它优化过度时会出现 reward hacking——回答越来越长、越来越迎合格式偏置而未必更好；KL 正则项 $\beta\,\mathrm{KL}(\pi_\phi \| \pi_{ref})$ 正是为拉住策略、不让它跑离参考分布太远。其二，**PPO 训练的系统性成本**——actor、critic、reward model、reference model 四个模型同时在内存中，加上在线采样的 rollout 开销，使 RLHF 阶段的工程复杂度远超预训练。其三，**alignment tax**——对齐后模型在部分原始能力基准上退化，需要混入预训练梯度等手段缓解。这三个问题分别定义了后续研究的三个出口：更好的奖励（RLVR）、更简单的优化（DPO）、更温和的目标（各种正则化变体）。

**DPO（Direct Preference Optimization，2023）** [14]：RLHF 需要训练奖励模型 + 跑不稳定的在线 RL，工程负担重。DPO 的洞察是：KL 约束下的最优策略与奖励函数存在闭式关系 $r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$，代入 Bradley-Terry 偏好模型，配分函数 $Z(x)$ 在成对比较中消去，直接得到对策略的分类式损失：

$$\mathcal{L}_{DPO}(\pi_\phi; \pi_{ref}) = -\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\!\left(\beta \log \frac{\pi_\phi(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\phi(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

无需奖励模型、无需在线采样、无需 PPO 超参调优，一次普通的梯度训练即完成偏好对齐。DPO 在情感控制、摘要、对话任务上匹配或超过 PPO-based RLHF [14]，迅速成为开源社区后训练默认选项，并衍生出 IPO、KTO、ORPO 等方法族；2026 年的理论综述已将这一族方法统一在偏好学习框架下 [26]。

**GRPO（Group Relative Policy Optimization）与 RLVR（2025）** [24]：DeepSeek-R1 把强化学习推向新方向。GRPO 去掉了 PPO 的 value network（critic），改为对同一 prompt 采样一组 $G$ 个回答，用组内奖励的相对排名构造优势：

$$\hat{A}_i = \frac{r_i - \mathrm{mean}(\{r_1,\dots,r_G\})}{\mathrm{std}(\{r_1,\dots,r_G\})}$$

配合与 PPO 类似的 clipped 目标与 KL 正则更新策略。去掉 critic 的收益是双重的：省掉一个与 policy 同规模的价值网络的训练（显存与不稳定性的主要来源），且组内基线天然自适应——同一 prompt 的难样本获得负优势、易样本获得正优势，无需学习绝对的价值标度。

关键转折在于奖励来源：**RLVR（Reinforcement Learning with Verifiable Rewards）** 用可程序化验证的奖励（数学答案是否正确、代码是否通过测试）替代 learned reward model，从根上规避了 reward hacking——你无法欺骗一个编译器或一个等号判定。DeepSeek-R1 证明纯 RL 即可激发长思维链、自我反思与验证等推理行为，不依赖人工标注的推理轨迹 [24]：论文记录了训练过程中模型的平均回答长度自发增长、出现"等等，让我重新检查"式的自我修正，这些行为没有出现在任何训练数据里，是 RL 压力下的自发产物。这直接重塑了 2025–2026 年后训练的技术主线，也把"推理"从 prompting 技巧升格为可训练的能力维度。

## 3. 经典方法谱系

### 3.1 关键论文一览表

| 论文 | 年份 | 核心思想 | 架构要点 | 备注 |
|---|---|---|---|---|
| Attention Is All You Need [1] | 2017 | 纯注意力架构取代循环/卷积 | MHA + FFN + Post-Norm，encoder-decoder | 一切基件之源 |
| Sparse MoE（Shazeer 等）[2] | 2017 | 稀疏门控解耦参数与计算 | top-k 路由 + 噪声均衡 | MoE 开山 |
| PPO [3] | 2017 | clipped 目标稳定 on-policy RL | 策略比值截断 | 后成 RLHF 默认算法 |
| MQA [4] | 2019 | 共享单组 K/V | KV cache 缩 h 倍 | 推理加速先驱 |
| Kaplan Scaling Laws [5] | 2020 | 损失随规模幂律变化 | 主张大模型优先 | GPT-3 的理论依据 |
| ViT [6b→见 29] | 2020 | 图像切块直过 Transformer | patch embedding + CLS | 视觉骨干范式确立 |
| RoFormer（RoPE）[8] | 2021 | 旋转注入相对位置 | $R_{\Theta,n-m}$ 相对性 | 现代 LLM 标配 |
| Switch Transformer [9] | 2021 | top-1 路由简化 MoE | 容量因子 + 辅助均衡损失 | 万亿参数可训练 |
| ALiBi [10] | 2021 | 线性距离偏置 | 无位置向量、可外推 | 短训长推 |
| GLaM [11] | 2021 | 1.2T 稀疏 MoE | top-2 路由 | 等算力超 GPT-3 级 dense |
| InstructGPT [7] | 2022 | SFT→RM→PPO 三段式 | RLHF 流程定型 | 对齐开山 |
| Chinchilla [6] | 2022 | 参数/数据同比扩展（~20 tok/param） | 400+ 模型拟合 | 改写训练配方 |
| PaLM [28] | 2022 | 540B dense + Pathways | 大规模系统协同 | 涌现能力展示 |
| FlashAttention [12] | 2022 | IO-aware 精确注意力 | tiling + online softmax | 算法-系统协同标杆 |
| Emergent Abilities [30] | 2022 | 规模阈值后的能力涌现 | 评测现象学 | scaling 讨论核心 |
| LLaMA [13] | 2023 | 公开数据训开源基座 | Pre-Norm/RMSNorm/SwiGLU/RoPE | 开源生态引爆点 |
| DPO [14] | 2023 | 闭式偏好优化 | 无 RM、无在线 RL | 后训练简化 |
| GQA [15/16] | 2023 | 分组共享 K/V | 质量≈MHA、速度≈MQA | 现代标配 |
| Llama 2 [15] | 2023 | 商用许可开源 + RLHF 配方 | GQA（70B） | 开源基座标准 |
| FlashAttention-2 [17] | 2023 | 并行度优化 | 再提速约 2× | 训练基础设施 |
| YaRN [18] | 2023 | RoPE 分频插值 | 少数据扩窗 | 长上下文主流 |
| Mistral 7B [19] | 2023 | GQA + 滑窗注意力 | 小模型架构优化 | 效率示范 |
| Mixtral 8×7B [20] | 2024 | 开源稀疏 MoE | top-2 路由、47B/13B 激活 | 对标 GPT-3.5 |
| DeepSeekMoE [31] | 2024 | 细粒度专家 + 共享专家 | 专业化上界 | DeepSeek 系谱基础 |
| DeepSeek-V2（MLA）[21] | 2024 | 低秩压缩 KV cache | 236B/21B 激活、cache -93.3% | MLA 首次落地 |
| Qwen2 [22] | 2024 | 现代配方代表 | GQA/RoPE/SwiGLU 全系 | 开源系列标杆 |
| DeepSeek-V3 [23] | 2024 | 671B MoE 全栈工程 | MLA + 无辅助损失均衡 + MTP | 2.788M H800 小时 |
| DeepSeek-R1 [24] | 2025 | 纯 RL 激发推理 | GRPO + 可验证奖励 | reasoning RL 转折点 |
| Hardware-AI 反思 [25] | 2025 | 架构-硬件协同复盘 | 设计决策的硬件约束 | co-design 时代宣言 |

### 3.2 谱系的内在逻辑

这张表看似繁杂，内在结构其实是四幕剧。读这张表的正确姿势不是记住每篇的名字，而是识别**每篇论文在解决前一篇的哪个遗留问题**——这正是本系列报告反复强调的"问题接力"视角：

**第一幕（2017–2020）确立骨架**：Transformer 定义了 MHA+FFN 的基本块；MoE 提供了稀疏扩展路线；ViT 证明同一骨架可以消化图像；Kaplan 定律给出了规模扩张的信心。

**第二幕（2021–2022）训练配方定型**：RoPE/ALiBi 解决位置编码；Chinchilla 修正 scaling 实践；FlashAttention 打通长序列训练的系统瓶颈；InstructGPT 建立对齐管线。到 2022 年底，"如何训练一个大模型"的全套方法论已经闭环。

**第三幕（2023–2024）开源与效率**：LLaMA 把配方开放给社区，GQA/FlashAttention-2/MQA-系 KV 压缩让推理成本骤降；Mixtral/DeepSeekMoE/MLA 把 MoE 从实验室推向生产；YaRN 等把上下文窗口推向 128K 甚至更长。

**第四幕（2025–2026）推理与协同**：DeepSeek-R1 开启 RLVR 推理范式；架构设计进入硬件感知阶段（V3 的 2.788M H800 GPU 小时训练 671B 模型是标志性数字 [23]）。多模态模型在这一幕中成为最大受益者——它直接站在所有基件的成熟态之上。

## 4. 代表作深度剖析

### 4.1 Attention Is All You Need（Transformer，2017）[1]

**研究问题**：序列转导模型（如机器翻译）能否彻底摆脱循环与卷积结构？

**关键洞察**：注意力机制本身就足以建模序列内任意位置间的依赖；循环结构是并行计算的障碍而非表达能力的必需品。

**技术方法**：encoder-decoder 各 6 层，$d_{model}=512$（base）/1024（big），$h=8/16$ 头，$d_{ff}=2048/4096$；缩放点积注意力 + 正弦位置编码 + Post-Norm 残差；训练用 Adam、带 warmup 的学习率调度、label smoothing 与 dropout 正则。

**实验结果**：WMT 2014 英德翻译 28.4 BLEU，超越此前最佳（含集成模型）超过 2 BLEU；英法翻译 41.8 BLEU 创单模型纪录，且仅用 8 张 GPU 训 3.5 天——训练成本只是同期最佳模型的一小部分 [1]。

**批判性分析**：优点是结构简洁、并行度极高、可扩展性被后续九年反复验证；还有一个事后才看清的优点——它足够"空"。Transformer 几乎不内置任何关于语言的归纳偏置（不像 CNN 的局部性、RNN 的时间性），这意味着它既不限制模型学什么，也不限制模型从哪种模态学。这个"空"正是它能从文本迁移到图像、音频、视频、动作的根本原因：通用架构 + 数据 + 算力的组合击败了各种领域特化设计，即所谓 "bitter lesson" 的架构注脚。局限在于论文本身未预见 $O(n^2)$ 注意力在长序列下的代价，也未解决位置外推、训练稳定性（Post-Norm 需要 warmup）等问题——本篇 2.5–2.7 节的全部内容，都可以读作对这篇论文遗留问题的九年修复史。可复现性极佳——代码与超参全部公开，衍生出 Tensor2Tensor 与无数复现。

**一句话评价**：现代深度学习史上被引用最多的架构论文，整个报告系列的一切都从这里生长。评级：★★★★★（必读原文）。

### 4.2 An Image is Worth 16×16 Words（ViT，2020）[29]

**研究问题**：图像识别能否像 NLP 一样，不依赖卷积归纳偏置，直接用纯 Transformer？

**关键洞察**：把图像切成固定大小的 patch（如 16×16 像素）拉平成向量序列，图像就变成了"一句话"；Transformer 在足够大的预训练数据下能自己学会 CNN 赖以成名的局部性与平移等变性。

**技术方法**：图像 $x \in \mathbb{R}^{H\times W\times C}$ 切成 $N = HW/P^2$ 个 $P\times P$ patch，每个 patch 经线性投影成 $d$ 维向量，加上可学习位置嵌入，序列前拼接可学习 [CLS] token，送入标准 Transformer encoder；用 [CLS] 输出做分类。模型族 ViT-B/L/H 分别为 86M/307M/632M 参数。

**实验结果**：在 JFT-300M 等大规模数据上预训练后，ViT 在 ImageNet、CIFAR-100、VTAB 等基准上达到或超越当时最优 CNN，且预训练所需计算资源更少 [29]。但论文同样诚实地指出：在中等规模数据（ImageNet 级别，约 1.3M 图）上，ViT 不如同级别 ResNet——大数据是 ViT 发挥作用的前提。

**批判性分析**：优点是彻底统一了视觉与语言的建模范式，使"视觉编码器输出 token 序列 → 喂给 LLM"的多模态管线成为可能（后续 CLIP、SigLIP、Qwen-VL 的视觉塔全部是 ViT 变体）；从工程角度看，ViT 还带来一个隐性红利——视觉与文本共享同一套训练基础设施（kernel、并行策略、混合精度），跨模态团队的系统栈得以统一。局限同样清晰：二次复杂度对高分辨率图像代价高（分辨率翻倍、序列长度翻四倍、注意力开销翻十六倍），后续 Swin 等用滑窗层级结构缓解，但多模态大模型仍以纯 ViT 为主流视觉塔（因为预训练好的纯 ViT 权重生态最丰富）；patch 化丢失像素级细节，影响 OCR、细粒度定位等任务——多模态模型后来用动态分辨率、多尺度裁剪等技巧打补丁，都是对这个原始缺陷的回应。可复现性好，官方权重公开。

**一个常被忽视的观察**：ViT 论文标题说"一张图值 16×16 个词"，但它的真正遗产是把图像变成了**可被 LLM 直接消费的离散序列**。没有这一步，后续"视觉 token 拼进文本 token 一起做自回归"的整个 MLLM 范式无从谈起。从这个意义上说，ViT 与其说是视觉论文，不如说是多模态论文——它提前五年为多模态基座模型准备好了接口协议。

**一句话评价**：把视觉带入 Transformer 纪元的论文，多模态大模型视觉塔的直系祖先。评级：★★★★★。

### 4.3 Training Compute-Optimal Large Language Models（Chinchilla，2022）[6]

**研究问题**：给定固定训练算力预算 $C$，模型参数量 $N$ 与训练 token 数 $D$ 应如何分配？

**关键洞察**：Kaplan 定律的"大模型优先"结论是在数据不受限的假设下得出的；当把 $N$ 与 $D$ 作为联合变量拟合 $L(N, D)$ 时，最优解要求二者**同比扩展**。

**技术方法**：训练超过 400 个模型（70M–16B 参数、5B–500B token），用三种独立方法估计计算最优前沿：（1）固定模型族规模、扫描 token 数取各算力下损失最低点；（2）固定 FLOPs 预算、扫描模型规模取 isoFLOP 曲线谷底；（3）直接拟合参数化损失函数 $L(N,D) = E + A/N^\alpha + B/D^\beta$（不可约损失 + 参数项 + 数据项）。三种方法一致指向 $D \approx 20N$。

**实验结果**：按此结论训练的 Chinchilla（70B 参数、1.4T token）在与 Gopher（280B、300B token）相同算力下，在绝大多数下游任务上显著反超 Gopher、GPT-3（175B）、Jurassic-1（178B）、Megatron-Turing NLG（530B）；MMLU 平均 67.5%，较 Gopher 提升超 7 个百分点 [6]。

**批判性分析**：方法论是经验科学的典范——三路径交叉验证、幂律外推有明确的置信区间；其研究设计本身就值得每位入门者抄笔记：不是训练一个超大模型然后写故事，而是训练数百个中小模型、把"预测大模型行为"当作一个可以被证伪的回归问题来处理。局限在于：（1）"20 token/参数"是**训练计算最优**而非**推理成本最优**，推理时代小模型超量训练（LLaMA 式）反而成为主流；（2）数据质量差异、数据重复（data-constrained 场景）未纳入模型；（3）未预见 test-time compute 范式——长思维链模型的最优配比可能系统性地偏向更小模型 + 更多推理算力。结论部分可复现（LLaMA 等公开模型间接验证了其预测）。

**一句话评价**：大模型时代最重要的一篇"如何花钱"的论文，其框架至今是所有训练预算讨论的起点。评级：★★★★★。

### 4.4 Training Language Models to Follow Instructions with Human Feedback（InstructGPT，2022）[7]

**研究问题**：大模型能力强大，但输出经常不符合用户意图（编造、有害、跑题）——如何让模型"对齐"？

**关键洞察**：把"符合人类意图"从抽象目标变成可优化的奖励信号：让人类对模型输出排序，用排序训练奖励模型，再用 RL 最大化该奖励。

**技术方法**：即 2.10 节的 SFT→RM→PPO 三段式。数据来自标注员撰写的 prompt 与 OpenAI API 真实用户 prompt；标注团队约 40 人，经过严格培训与标注者间一致性筛选——论文用了相当大的篇幅讨论标注协议，因为整个方法的成立前提就是"排序信号足够干净"。模型上以 GPT-3 各尺寸（1.3B/6B/175B）为起点，逐级走完三段流程。PPO 训练中混入预训练梯度（PPO-ptx）以缓解公共 NLP 基准上的能力退化（alignment tax），这个细节预示了后来所有对齐工作都要面对的"对齐-能力"跷跷板。

**实验结果**：在 API prompt 分布的人工评估中，1.3B InstructGPT 的输出偏好率超过 175B GPT-3——百倍参数差距被对齐抹平；真实性提升、有毒输出减少，公共 NLP 基准退化轻微 [7]。

**批判性分析**：历史性贡献在于确立了"预训练→SFT→RLHF"这一沿用至今的范式（本系列后续对齐篇的 fig_training_pipeline 三阶段图即源于此）；局限包括：标注分布狭窄（40 名标注者的偏好能否代表全人类？）、奖励模型可被策略模型过优化（reward hacking 的早期体现）、PPO 训练不稳定且昂贵。这些局限恰好定义了后续 DPO、RLVR 要解决的目标。部分可复现（代码未公开，但配方细节充分，开源社区已有大量复现）。

**一句话评价**：把"对齐"从哲学讨论变成工程管线的论文，ChatGPT 时代的技术起点。评级：★★★★★。

### 4.5 Direct Preference Optimization（DPO，2023）[14]

**研究问题**：RLHF 的三段式管线（RM + PPO）复杂且不稳定，能否绕开显式奖励模型与在线强化学习，直接优化偏好？

**关键洞察**：在 KL 正则的 RLHF 目标下，最优策略对任意奖励函数有闭式解；反过来，奖励可以被最优策略与参考策略的对数比**解析地参数化**——代入 Bradley-Terry 模型后，配分函数消去，偏好学习退化为一个二分类问题。

**技术方法**：推导骨架值得再走一遍。KL 正则 RLHF 目标 $\max_\pi \mathbb{E}[r(x,y)] - \beta\,\mathrm{KL}(\pi \| \pi_{ref})$ 对每个 prompt $x$ 是独立的优化问题，其闭式最优解为 $\pi^*(y|x) \propto \pi_{ref}(y|x) \exp(r(x,y)/\beta)$——软最大化（softmax）形式的策略。反解出 $r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$ 代入 Bradley-Terry 模型，未知的配分函数 $Z(x)$ 在 $r(x,y_w) - r(x,y_l)$ 的差中精确消去，剩下的就是 2.10 节的 $\mathcal{L}_{DPO}$：一个只含策略对数概率的 logistic 回归损失。实现上只需 SFT 模型（兼作 $\pi_{ref}$ 与初始化 $\pi_\phi$）与偏好数据集，一次监督式梯度训练即完成对齐；超参数只有 $\beta$（KL 强度）与学习率。

**实验结果**：在情感控制、单轮对话、摘要等任务上，DPO 匹配或超过 PPO-based RLHF，且训练稳定性与计算成本显著更优 [14]。

**批判性分析**：优点是优雅（理论推导干净）+ 实用（工程简化一个数量级），迅速成为开源后训练事实标准；局限包括：偏好数据是离线的（off-policy），随着策略漂移，数据分布与当前策略脱节——PPO 的在线采样能持续"问自己现在会答成什么样"，DPO 只能问"当初那个模型答成什么样"；对偏好数据质量敏感，一个已知的系统性问题是长度偏置——标注者普遍偏好更长的回答，DPO 会忠实地把这个偏置放大为冗长输出；$\beta$ 控制隐式 KL 强度，极端偏好对下可能出现"赢家概率也被压低"的退化现象（只追求拉开差距而非抬高赢家），后续 length-normalized 与约束变体都在修补这一点。这些局限催生了在线 DPO、IPO、KTO 等改进族。2026 年的统一理论框架 [26] 证明 DPO 族与 RLHF 在数学上是同一目标的不同参数化——"去 RL 化"并非绕开 RL，而是把 RL 的解折叠进了监督目标。

**一句话评价**：一行损失函数掀翻整条 RLHF 流水线的论文，理论与实践结合的最佳范例。评级：★★★★★。

### 4.6 DeepSeek-V3 Technical Report（2024）[23]

**研究问题**：能否在受限硬件（出口管制下的 H800，互联带宽受限）上，以可承受的成本训练一个与顶级闭源模型同档的开源 MoE 模型？

**关键洞察**：把此前分散验证的架构创新（MLA、细粒度 MoE）与系统创新（FP8 训练、通信-计算重叠、无辅助损失均衡）组装成一个**协同设计**的整体——每个决策都同时回应"质量"与"硬件约束"两个目标，而非先后优化。

**技术方法**：671B 总参数、每 token 激活 37B；注意力采用 MLA 低秩压缩 KV cache；FFN 采用 DeepSeekMoE 细粒度专家 + 共享专家；负载均衡不用辅助损失，改用可动态调整的路由偏置项（见 2.7.1）；训练目标加入多 token 预测（MTP，同时预测未来多个 token 以 densify 监督信号）；预训练 14.8T token；全程 FP8 混合精度。

**实验结果**：训练全程仅消耗 2.788M H800 GPU 小时，且没有发生任何不可恢复的损失尖峰、没有回滚 [23]；综合评测超越全部开源模型、与顶级闭源模型相当。作为参照，同档 dense 模型的训练成本通常是其数倍。

**批判性分析**：这是"架构-系统-硬件"三位一体的工程杰作，其意义不在于某个单点创新（每个组件此前都有论文），而在于**证明了中国开源生态可以在硬件受限条件下重构前沿配方**——2025 年的硬件复盘 [25] 把这种协同设计上升为方法论。局限同样明显：报告是工程文档而非科学论文，大量消融实验（每个组件的独立贡献）未公开；2.788M GPU 小时的数字不含前期研究与失败的消融，读者不应把它当作"可复现成本"而应当作"下界参考"。可复现性中等——权重开源、推理可验证，但训练栈（框架、数据配比）未完整公开。

**一句话评价**：单篇浓缩 MLA、MoE 均衡、FP8、流水线四大工程主题的"现代大模型训练百科全书"，理解 2025 年后所有旗舰模型的必读文献。评级：★★★★★。

## 5. 研究前沿（2025–2026）

**（1）推理范式转向 RL**：DeepSeek-R1 [24] 之后，"RLVR + 长思维链"成为后训练主线，2025 年几乎所有旗舰模型的发布都包含一个"reasoning 版本"。GRPO 类 value-free 算法被开源社区广泛采用，并衍生出去 KL 项、去方差归一化、token 级优势等一系列变体——细节之争背后是一个共同的观察：当奖励是稀疏的 0/1 验证信号时，PPO 时代精心设计的许多组件不再必要。前沿问题是：可验证奖励如何推广到不可验证领域（写作、开放式问答——rubric 评分与 LLM-as-judge 是目前两条探路）、推理长度与正确率的 trade-off（"overthinking"——模型在简单题上浪费数千 token——已成为新的优化目标）、以及多模态推理（视觉 CoT）的 RLVR 化。

**（2）MoE 架构定型并下探**：DeepSeek-V3 [23] 确立了"MLA + 细粒度 MoE + 无辅助损失负载均衡 + 多 token 预测（MTP）"的组合配方，671B 总参、37B 激活、14.8T token 预训练、全程仅 2.788M H800 GPU 小时且无不可恢复的损失尖峰 [23]。Qwen3、GLM、Kimi 等系列跟进类似设计。前沿问题包括：专家专业化是否真的发生、细粒度与共享专家的最优比例、MoE 的 scaling law 形式化。

**（3）硬件感知架构设计**：DeepSeek 团队的 V3 复盘 [25] 标志着架构创新进入"硬件 co-design"阶段——MLA 的低秩结构、FP8 混合精度训练（用 8 位浮点做大部分矩阵乘、关键累加留在高精度）、流水线并行的气泡消除（DualPipe 式双向流水调度），都是为具体硬件（H800 受限的互联带宽）量身定制的决策。这篇复盘最启发性的一点是：**许多被社区当作"算法创新"的设计，其实是对硬件约束的最优响应**——如果互联带宽不同，同样的模型可能会长出不同的路由与压缩方案。这意味着未来架构论文将越来越多地包含系统层的约束分析，脱离硬件语境讨论"哪个架构更好"正在失去意义。

**（4）高效架构的百家争鸣**：2025–2026 年的高效架构综述 [32] 反映了社区焦点的扩散：linear attention、sparse attention、状态空间模型（Mamba 系）、混合架构（hybrid：少量全注意力层 + 大量线性层）都在争夺"后 Transformer"的位置。位置编码综述 [27] 则显示长上下文技术（NTK、YaRN、插值）已收敛为成熟工具箱。对多模态而言，这些高效架构尤为关键——视频理解动辄数万 token，纯二次注意力难以为继。

**（5）对齐理论收敛**：RLHF→DPO→RLVR 的接力已清晰，理论综述 [26] 把偏好学习方法族统一到同一框架；2026 年的新动向包括 rubric（评分细则）引导的 RL、多目标对齐（有用/无害/诚实的帕累托前沿）、以及多模态场景的安全对齐（图像输入的越狱攻击面）。

## 5.5 点题：这些基件如何被多模态模型复用

让我们把视角拉回整个系列的开篇问题：一个 2026 年的多模态基座模型，内部到底装了什么？答案是——几乎全是本篇讲过的基件，只是换了一种拼装方式：

1. **语言主干 = 现代配方全家桶**。主流多模态大模型（MLLM）的语言侧就是一个标准 decoder-only LLM：Pre-Norm + RMSNorm 残差、SwiGLU FFN、RoPE 位置编码、GQA 或 MLA 注意力、FlashAttention 内核、dense 或 MoE 前馈。Qwen-VL、InternVL、Kimi、GPT-4o 级系统的语言塔无一不在这条谱系上。理解本篇第 2 节，就等于理解了任何 MLLM 一半以上的参数量在做什么。

2. **视觉骨干 = ViT 的直接后裔**。视觉塔几乎清一色是 ViT 变体（CLIP/SigLIP 对比学习预训练的 ViT），输出 patch token 序列——ViT 把"图像即 token 序列"这件事确立之后，视觉接入语言模型就变成了纯粹的 token 对齐问题（connector/投影层），这是第 01 篇（视觉-语言）的全部出发点。

3. **位置编码与长上下文技术 = 多模态的刚需放大器**。一张高清图可以是数千 token，一分钟视频可以是数万 token；RoPE + YaRN 式插值 [18] 与 ALiBi 式外推思想被直接搬到图像二维位置、视频时空位置编码上（2D-RoPE、时空分解旋转等变体），长上下文技术从"文本奢侈品"变成"多模态基础设施"。

4. **注意力效率变体 = 多模态推理成本的生命线**。多模态序列长、图像 token 冗余度高，KV cache 压力成倍放大；GQA/MLA [21] 与滑窗注意力 [19] 的每一项都直接决定视觉-语言模型能否实时交互。视觉 token 的剪枝与压缩，则可以看作 MQA/MLA"压缩信息而非删减结构"思想在模态维度上的延伸。

5. **MoE = 多模态容量扩展的主通道**。当模型要同时掌握文本、图像、音频、视频、动作时，"总参大、激活小"的 MoE 结构 [20, 23] 成为容纳多模态知识而不爆推理成本的自然选择；按模态/任务分化专家，是 MoE 语义在多模态下的再诠释。

6. **预训练范式 = 多模态目标的模板**。CLM（next-token prediction）不仅统治文本生成，也被推广为"next-patch/next-image-token prediction"（自回归图像生成，见第 05 篇）；BERT 式掩码建模则活在 MAE 式视觉自监督里。两大范式的分野与汇合贯穿整个系列。

7. **Scaling laws = 多模态预算分配的方法论**。Chinchilla 的 $L(N, D)$ 拟合框架 [6] 被逐字搬到多模态 scaling 研究：图文配比对、视觉 token 数与语言参数的最优配比、数据受限下的重复训练，都是 Kaplan→Chinchilla 方法论的直接后代（第 11 篇数据与训练将展开）。

8. **对齐技术栈 = 原封不动的移植**。SFT → DPO → RLVR 的三段式后训练对多模态模型几乎不加修改地复用：多模态指令微调就是 SFT、多模态偏好数据上的 DPO 就是 DPO、视觉数学/图表题上的可验证奖励就是 RLVR [24]。多模态对齐的特殊性（幻觉、跨模态安全）是第 09 篇的主题，但工具箱就是本篇 2.10 节的那一套。

一句话总结本篇的立场：**多模态大模型不是新架构，而是旧基件的新编排**。吃透这八个基件的动机、数学与权衡，后续每一篇专题报告对你而言都将只是"这个基件在这个模态下长什么样"的具体化练习。

## 6. 技术路线对比与权衡

本节把前面分散在各基件里的"选择问题"集中到四张表：推理时注意力怎么选（6.1）、位置编码怎么选（6.2）、对齐管线怎么搭（6.3）、前馈用 dense 还是 MoE（6.4）。这四个选择基本决定了一个现代基座模型的架构画像——2026 年你读到任何一份新模型技术报告，前四页大概率就是在回答这四个问题。

### 6.1 注意力变体对比

| 方案 | KV cache（相对 MHA） | 质量 | 推理速度 | 代表模型 | 适用场景 |
|---|---|---|---|---|---|
| MHA | 1× | 最高 | 基线 | GPT-3 | 训练、质量优先 |
| MQA | 1/h | 有可见损失 | 最快 | PaLM 部分变体 | 极端延迟敏感 |
| GQA | 1/g（g≈4–8） | ≈MHA | ≈MQA | Llama 2/3、Qwen2 | 通用默认 |
| MLA | 低秩压缩（V2 报告 -93.3% [21]） | ≈MHA 甚至更好 | 高吞吐 | DeepSeek-V2/V3 | 大 MoE + 长上下文 |
| 滑窗注意力 | 窗口内全量 | 长程略降 | 长序列友好 | Mistral [19] | 中小模型长文本 |

核心权衡是**缓存大小 vs 表达容量**。值得强调：MLA 的思路（低秩瓶颈而非删减 head）可能是更本质的方向，因为它保留了逐 head 的多样性，只是约束其位于低维子空间。

还有一个常被忽略的部署维度：这些方案与**训练方式**耦合。MQA/GQA 可以从头训练，也可以像 GQA 论文 [16] 那样从已有 MHA 检查点 uptraining（5% 预训练算力）；MLA 的低秩结构则基本需要从头训练（或在架构改造时付出较大转换成本），因为它改变了注意力内部的投影拓扑。对于手握大量 MHA 时代检查点的团队，GQA 的"可转换性"是它在 2023 年迅速胜出的隐性原因——架构竞争有时不由质量曲线决定，而由迁移成本决定。

### 6.2 位置编码对比

| 方案 | 类型 | 外推性 | 相对位置感知 | 现状 |
|---|---|---|---|---|
| Sinusoidal | 绝对 | 理论可外推，实践差 | 隐式 | 历史意义 |
| Learned PE | 绝对 | 不可外推 | 弱 | ViT 内部仍常用 |
| ALiBi | 相对偏置 | 好 | 显式 | 少数模型（如早期 BLOOM/MPT） |
| RoPE | 旋转相对 | 需插值/YaRN 辅助 | 显式 + 长程衰减 | 文本 LLM 绝对主流 |

表中"现状"一栏值得一个解释：为什么外推性更好的 ALiBi 没有赢？三个原因。其一，线性偏置把"距离衰减"硬编码为固定斜率，而真实语言中有些长程依赖（如跨段落的指代、代码中遥相呼应的括号）恰恰不应衰减——RoPE 的衰减是可学习的函数形状，ALiBi 的是先验常数。其二，RoPE 只作用于 Q/K 而不改注意力结构，与 FlashAttention 等 kernel 完全兼容；ALiBi 的加性偏置矩阵在分块计算里需要额外处理（实践中也已解决，但生态跟进慢了一拍）。其三，YaRN 等插值技术 [18] 基本补齐了 RoPE 的外推短板，竞争窗口随之关闭。这段历史提醒我们：架构竞争的胜负往往由"短板可否被廉价补齐"决定，而非长板有多高。

### 6.3 对齐方法对比

| 方法 | 奖励来源 | 是否在线 | 训练成本 | 主要风险 | 代表 |
|---|---|---|---|---|---|
| PPO-RLHF | learned RM | 在线 | 高（actor+critic+RM+ref 四模型） | reward hacking、训练不稳 | InstructGPT [7] |
| DPO | 隐式（对数比参数化） | 离线 | 低（两模型、监督式） | 分布漂移、长度偏置 | Zephyr、众多开源模型 [14] |
| GRPO/RLVR | 程序化验证 | 在线 | 中（无 critic） | 仅适用可验证任务 | DeepSeek-R1 [24] |

实践中的现代配方往往是组合拳：SFT → DPO（或在线偏好优化）打底通用偏好，RLVR 专项强化推理/代码/数学。

三者的深层差异其实是**监督信号的来源**：PPO-RLHF 的信号来自 learned RM（拟合人类但可被欺骗），DPO 的信号来自静态偏好对（干净但会过时），RLVR 的信号来自程序验证（不可欺骗但覆盖面窄）。这个"来源光谱"解释了为什么没有任何单一方法能终结对齐研究——覆盖面、抗欺骗性、时效性构成不可能三角，每家实验室都在按自己的风险偏好选点。对入门者而言，这个三角也是评估任何新对齐论文的第一把尺子：它在三角上牺牲了什么、换来了什么？

### 6.4 Dense vs MoE

| 维度 | Dense | Sparse MoE |
|---|---|---|
| 参数/计算 | 耦合 | 解耦（总参大、激活小） |
| 训练稳定性 | 高 | 需负载均衡技巧（辅助损失或无辅助策略 [23]） |
| 推理成本 | 与参数成正比 | 与激活参数成正比，但显存需装全部专家 |
| 通信开销 | 标准张量并行 | 专家并行的 all-to-all 通信是系统难点 |
| 代表 | LLaMA、Qwen2-dense | Mixtral [20]、DeepSeek-V3 [23] |

2024–2026 年的趋势清晰：**旗舰模型几乎全面转向 MoE**，因为训练 FLOPs 相同时 MoE 质量更高、推理激活参数更少；dense 模型退守中小尺寸与端侧。但"转向"有一个重要的例外逻辑值得说明：当推理部署的显存预算紧张时（端侧、单卡场景），MoE"激活小但总参大"的特性反而是劣势——13B 激活的 Mixtral 仍需装下全部 47B 参数。所以 MoE 是**数据中心推理**的赢家，dense 仍统治**边缘推理**，这个分裂在可预见的将来不会消失。

## 7. 开放问题与未来方向

1. **注意力的"后二次"时代何时到来**：linear/sparse/hybrid 架构在基准上不断逼近全注意力，但在长程检索、in-context learning 等关键能力上仍有差距。多模态长序列（视频、长文档图像）会是这场竞赛的试金石。
2. **数据受限下的 scaling law**：文本数据逼近枯竭，多模态高质量配对数据更稀缺。数据重复的收益递减曲线、合成数据的 scaling 性质、跨模态迁移对配比的修正，都缺乏 Chinchilla 级别的系统性研究。
3. **MoE 的理论理解**：专家为何专业化、何时失效，路由的梯度动力学，细粒度专家的上界——目前几乎全是经验结论。
4. **RLVR 的边界**：可验证奖励之外，如何为开放式、主观性任务设计抗 hacking 的奖励？rubric-based 奖励与 learned RM 的混合是否是终局？
5. **架构-硬件协同的自动化**：V3 的设计决策由专家手工完成；能否把硬件约束纳入架构搜索（hardware-aware NAS for LLM）？
6. **统一的多模态 scaling 理论**：文本、图像、视频、音频 token 的最优混合比例随算力如何变化？这是把 Kaplan/Chinchilla 框架推广到多模态的核心问题，也是本系列后续篇章反复回到的理论锚点。
7. **推理时扩展与训练时扩展的联合最优**：RLVR 时代，同一个模型既要被训练（花算力）又要会推理时思考（花 token）。给定总预算，"训练更大"与"想更久"之间是否存在类似 Chinchilla 的配比定律？初步证据表明最优解强烈依赖任务的可验证性，但系统性的拟合研究尚未出现。
8. **位置编码的二维/时空推广**：RoPE 在文本的一维序列上是优雅的，但图像的高度-宽度、视频的时间-空间、音频的帧-频结构如何共享或分解旋转基？目前各家 MLLM 的方案（2D-RoPE、时空分解、简单可学习嵌入）高度碎片化，缺乏一维 RoPE 当年那样的统一理论。

## 8. 研究机会点评（给入门研究者的建议）

**算力门槛适中、问题清晰的子方向**：

- **长上下文位置编码与检索评测**：YaRN 类方法 [18] 的改进只需 7B 级模型 + 数张 GPU 微调即可验证；长上下文评测（如大海捞针类任务的更严谨版本）仍有方法学空白。
- **DPO 族的算法改进**：离线偏好优化的理论（何时退化、如何校正分布漂移）与改进算法，单卡即可做 1B–8B 级实验，顶会产出稳定。
- **KV cache 压缩**：MLA 之后的下一步（量化 + 低秩 + 稀疏的组合），与推理系统结合紧密，工程与学术价值兼备。
- **小规模 scaling law 研究**：用 400 个小模型复刻 Chinchilla 式拟合在学术预算内可行（这正是 Chinchilla 论文自己的方法），数据质量、数据重复、多模态配比等修正项都是好题目。

**需要正视的"巨头游戏"**：

- **旗舰 MoE 预训练**：DeepSeek-V3 级别（2.788M H800 小时）的投入是学术机构无法企及的；学术界应做配方创新（新路由、新均衡策略）并在中小规模验证，而非正面竞争。
- **大规模 RLHF/RLVR 训练**：在线 RL 的系统复杂度（rollout 集群、vLLM 推理引擎协同）本身就是工程壁垒；但算法层创新（新优势估计、新奖励塑形）可以用小模型完成概念验证。
- **基础架构替代（linear attention/SSM）**：需要在与 Transformer 同等规模的公平对比中取胜才有人信服，这个"同等规模"的门槛正在逐年抬高。

一条务实建议：入门者的第一篇工作尽量选"**小模型上可完整复现的配方级问题**"（如 GQA 组数的选择、RoPE base 频率的敏感性、DPO 的 β 与数据质量交互），这类问题基件级、可复现、且结论能迁移到大模型，是建立学术信誉的最快路径。

第二条建议关于读论文的顺序：本篇第 4 节的六篇深读是按"叙事骨架"而非难度排序的——Transformer（架构原点）→ ViT（模态扩展）→ Chinchilla（规模方法论）→ InstructGPT（对齐原点）→ DPO（对齐简化）→ DeepSeek-V3（现代集成）。按这个顺序读，你会发现每篇论文都在回答前一篇留下的问题，这种"问题接力"的读法比按时间线泛泛而读有效得多。

第三条建议：尽早建立自己的**复现环境**。哪怕只是用开源框架在小模型上跑一次 DPO、测一次 RoPE 外推曲线，亲手观察 loss 曲线的形状与超参的敏感性，是把"读过的知识"变成"可用的直觉"的唯一途径。大模型研究入门最大的陷阱不是算力不足，而是只读不做。

## 9. 术语表

1. **Self-Attention（自注意力）**：序列内部任意两位置直接计算相关性并加权聚合的机制，Transformer 的核心。
2. **MHA / MQA / GQA / MLA**：多头注意力及其 KV 压缩变体族：全量 K/V、单组共享、分组共享、低秩潜在压缩。
3. **KV cache**：自回归推理时缓存的历史 token 的键值对，决定推理显存与带宽开销。
4. **FFN（前馈网络）**：Transformer 块内逐位置的两层 MLP，承担约 2/3 参数，被认为是知识存储单元。
5. **SwiGLU**：带门控分支的 Swish 激活 FFN 变体，现代开源 LLM 标准配置。
6. **Pre-Norm / RMSNorm**：残差前归一化与均方根归一化，深层 Transformer 稳定训练的关键组合。
7. **RoPE（旋转位置编码）**：对 Q/K 施加与位置成正比的二维旋转，使注意力分数只依赖相对位置的编码方案。
8. **ALiBi**：在注意力分数上叠加与距离成正比的线性惩罚的位置方案，外推性好。
9. **YaRN**：按频率分组对 RoPE 做差异化插值的长上下文扩展方法。
10. **FlashAttention**：通过 SRAM 分块与在线 softmax 避免物化注意力矩阵的 IO 感知精确算法。
11. **MoE（混合专家）**：用路由网络把每个 token 分派给少数专家子网络，稀疏激活以解耦参数量与计算量。
12. **ViT（Vision Transformer）**：把图像切成 patch 序列直接过 Transformer 的视觉骨干。
13. **CLM / MLM**：因果语言建模（从左到右预测下一 token）与掩码语言建模（双向预测被遮盖 token）两大预训练目标。
14. **Scaling Law**：损失随参数/数据/算力呈幂律变化的经验规律，用于预测与分配训练预算。
15. **Chinchilla 最优**：参数与 token 数同比扩展（约 20 token/参数）的计算最优训练点。
16. **SFT（监督微调）**：用示范数据对预训练模型做指令跟随微调，对齐管线第一步。
17. **Reward Model（奖励模型）**：从人类偏好排序中学到的标量评分函数，RLHF 的奖励来源。
18. **PPO**：用 clipped surrogate objective 约束更新幅度的 on-policy 强化学习算法，经典 RLHF 的优化器。
19. **DPO**：把偏好学习闭式化为监督式分类损失的对齐方法，无需奖励模型与在线 RL。
20. **GRPO / RLVR**：去掉 critic、用组内相对奖励构造优势的 RL 算法，与程序化可验证奖励结合激发推理能力。

## 10. 参考文献

[1] Attention Is All You Need, Vaswani et al., 2017, arXiv:1706.03762, https://arxiv.org/abs/1706.03762
[2] Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer, Shazeer et al., 2017, arXiv:1701.06538, https://arxiv.org/abs/1701.06538
[3] Proximal Policy Optimization Algorithms, Schulman et al., 2017, arXiv:1707.06347, https://arxiv.org/abs/1707.06347
[4] Fast Transformer Decoding: One Write-Head is All You Need, Shazeer, 2019, arXiv:1911.02150, https://arxiv.org/abs/1911.02150
[5] Scaling Laws for Neural Language Models, Kaplan et al., 2020, arXiv:2001.08361, https://arxiv.org/abs/2001.08361
[6] Training Compute-Optimal Large Language Models, Hoffmann et al., 2022, arXiv:2203.15556, https://arxiv.org/abs/2203.15556
[7] Training language models to follow instructions with human feedback, Ouyang et al., 2022, arXiv:2203.02155, https://arxiv.org/abs/2203.02155
[8] RoFormer: Enhanced Transformer with Rotary Position Embedding, Su et al., 2021, arXiv:2104.09864, https://arxiv.org/abs/2104.09864
[9] Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity, Fedus et al., 2021, arXiv:2101.03961, https://arxiv.org/abs/2101.03961
[10] Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation, Press et al., 2021, arXiv:2108.12409, https://arxiv.org/abs/2108.12409
[11] GLaM: Efficient Scaling of Language Models with Mixture-of-Experts, Du et al., 2021, arXiv:2112.06905, https://arxiv.org/abs/2112.06905
[12] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, Dao et al., 2022, arXiv:2205.14135, https://arxiv.org/abs/2205.14135
[13] LLaMA: Open and Efficient Foundation Language Models, Touvron et al., 2023, arXiv:2302.13971, https://arxiv.org/abs/2302.13971
[14] Direct Preference Optimization: Your Language Model is Secretly a Reward Model, Rafailov et al., 2023, arXiv:2305.18290, https://arxiv.org/abs/2305.18290
[15] Llama 2: Open Foundation and Fine-Tuned Chat Models, Touvron et al., 2023, arXiv:2307.09288, https://arxiv.org/abs/2307.09288
[16] GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, Ainslie et al., 2023, arXiv:2305.13245, https://arxiv.org/abs/2305.13245
[17] FlashAttention-2: Better Attention with Better Parallelism and Work Partitioning, Dao, 2023, arXiv:2307.08691, https://arxiv.org/abs/2307.08691
[18] YaRN: Efficient Context Window Extension of Large Language Models, Peng et al., 2023, arXiv:2309.00071, https://arxiv.org/abs/2309.00071
[19] Mistral 7B, Jiang et al., 2023, arXiv:2310.06825, https://arxiv.org/abs/2310.06825
[20] Mixtral of Experts, Jiang et al., 2024, arXiv:2401.04088, https://arxiv.org/abs/2401.04088
[21] DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model, DeepSeek-AI, 2024, arXiv:2405.04434, https://arxiv.org/abs/2405.04434
[22] Qwen2 Technical Report, Yang et al., 2024, arXiv:2407.10671, https://arxiv.org/abs/2407.10671
[23] DeepSeek-V3 Technical Report, DeepSeek-AI, 2024, arXiv:2412.19437, https://arxiv.org/abs/2412.19437
[24] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, DeepSeek-AI, 2025, arXiv:2501.12948, https://arxiv.org/abs/2501.12948
[25] Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures, DeepSeek-AI, 2025, arXiv:2505.09343, https://arxiv.org/abs/2505.09343
[26] From RLHF to Direct Alignment: A Theoretical Unification of Preference Learning for Large Language Models, 2026, arXiv:2601.06108, https://arxiv.org/abs/2601.06108
[27] Position Encoding in Transformers: From Absolute and Relative Methods to Rotary Position Embeddings and Long-Context Scaling, 2026, arXiv:2608.10021, https://arxiv.org/abs/2608.10021
[28] PaLM: Scaling Language Modeling with Pathways, Chowdhery et al., 2022, arXiv:2204.02311, https://arxiv.org/abs/2204.02311
[29] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, Dosovitskiy et al., 2020, arXiv:2010.11929, https://arxiv.org/abs/2010.11929
[30] Emergent Abilities of Large Language Models, Wei et al., 2022, arXiv:2206.07682, https://arxiv.org/abs/2206.07682
[31] DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models, DeepSeek-AI, 2024, arXiv:2401.06066, https://arxiv.org/abs/2401.06066
[32] Speed Always Wins: A Survey on Efficient Architectures for Large Language Models, 2025, arXiv:2508.09834, https://arxiv.org/abs/2508.09834
[33] A Survey on Multimodal Large Language Models, Yin et al., 2023, arXiv:2306.13549, https://arxiv.org/abs/2306.13549
[34] Efficient Large Language Models: A Survey, Wan et al., 2023, arXiv:2312.03863, https://arxiv.org/abs/2312.03863
[35] A Survey of Generative Categories and Techniques in Multimodal Generative Models, 2025, arXiv:2506.10016, https://arxiv.org/abs/2506.10016

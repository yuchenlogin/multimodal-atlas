# 可直接插入的技术补文

日期：2026-10-16。以下是供主编辑整合的教学正文，按8组关键缺口组织；每组含变量/shape、算法或更新、工作示例、失败诊断和来源。章节号只是建议插入点。所有小规模数值例与实验配置均为本文构造或拟议协议；没有运行模型训练或新增论文性能数字。论文链接用于核对机制，实际采用时应固定版本。

## 1｜00：从位置旋转到稀疏专家的资源账

建议接在00的张量计算之后，作为时空位置与MoE成本的先修。

### 位置坐标怎样改变 attention

先把一个 head 的维度 $d_h$ 取为偶数。投影得到 $Q,K\in\mathbb R^{B\times h\times T\times d_h}$ 后，将最后一维拆成 $d_h/2$ 对。第 $j$ 对的频率为 $\theta_j=b^{-2j/d_h}$，$j=0,\ldots,d_h/2-1$，$b$ 是选定的基数；位置 $m$ 的二维旋转为

$$
R_j(m)=\begin{pmatrix}\cos(m\theta_j)&-\sin(m\theta_j)\\\sin(m\theta_j)&\cos(m\theta_j)\end{pmatrix}.
$$

对每对query/key分别旋转，value按此例保持原样。设未经位置旋转的两个向量为 $q,k$，则

$$
(R(m)q)^T(R(n)k)=q^TR(n-m)k.
$$

等式来自 $R(m)^TR(n)=R(n-m)$：共同移动两个坐标保留相位差，attention仍然可以依赖相对间隔。这是RoPE的关键构造。[RoFormer §3.2](https://arxiv.org/html/2104.09864)

用单个二维子空间手算。取 $q=k=(1,0)$、教学频率 $\theta=\pi/6$，位置为 $m=2,n=5$，内积为 $\cos(3\pi/6)=0$；整体移动到 $7,10$，内积仍为0。若只把key移动到6，相对间隔变4，内积变为 $\cos(2\pi/3)=-1/2$。这里选择简单频率便于计算，真实模型按各自频率表旋转。

实现可以广播位置张量 `pos[B,T]` 到heads和成对通道：

```text
angles = pos[..., None] * frequencies[None, None, :]  # [B,T,dh/2]
for Q and K:
    even, odd = paired_coordinates(tensor)           # [B,h,T,dh/2]
    rotated_even = even*cos(angles) - odd*sin(angles)
    rotated_odd  = even*sin(angles) + odd*cos(angles)
    interleave(rotated_even, rotated_odd)
```

这里采用“缓存已经旋转过的K”的约定：prefill对整段K旋转一次；decode只旋转新位置的Q/K，再追加K/V。读缓存时再次旋转会累计错误相位。batch内两条序列即使被padding到相同长度，其有效末位置也可能不同，下一步的position ID应按各自位置规则计算。一个实用检查是：固定权重、输入与mask，将同一序列整段计算和分段cache计算的最后一个logit比较到合理浮点容差；再改变padding长度重复检查。

多模态中的“位置”可以有多个坐标：文本序号、图像行列、视频真实时间。只要先定义每个rotary通道组采用哪种坐标，上面的二维更新仍适用。这样03/04的时空位置就能接回同一计算图。

### MoE怎样分开总参数和每token计算

将 $T_b$ 个有效token展平为 $X\in\mathbb R^{T_b\times d}$。router参数 $W_r\in\mathbb R^{d\times E}$，得到概率 $P=\mathrm{softmax}(XW_r)$。对每个token选 $k$ 个专家索引 $I\in\{1,\ldots,E\}^{T_b\times k}$，令选中门控归一化为 $g_{ti}$，输出

$$
Y_t=\sum_{i\in I_t}g_{ti}\,\mathrm{FFN}_i(X_t).
$$

以门控FFN为例，每个专家约有 $P_e=3dd_{ff}$ 参数；整个专家层约存 $EP_e+dE$ 参数，每token访问约 $kP_e$ 专家参数对应的计算路径。取 $d=2048,d_{ff}=8192,E=8,k=2$：单专家约50.3M参数，全部专家约402.7M，每token选中专家约100.7M，另加router。这个账只包含该MoE FFN；attention、embedding和共享模块仍需另计。

下面是一层的执行顺序：

```text
compute router scores for valid tokens
choose top-k expert IDs and gates
group token activations by expert; retain source-token IDs
dispatch grouped activations to expert owners
run each expert on its assigned activations
return expert outputs and scatter-add with gates
add the residual; log expert counts by input modality
```

专家分布在多卡时，dispatch/return搬运的是激活。忽略metadata，若每个token发送到k个专家、激活元素为s字节，两程合计约 $2kT_bds$ 字节。取 $T_b=1024,k=2,d=2048,s=2$，得到16MiB的逻辑激活流量；实际网络量还取决于专家是否本地、通信算法和padding。总权重容量没有因只选两个专家而消失。

负载也影响正确训练和速度。以top-1的容量限制示例，$T_b=16,E=4$，capacity factor=1，每个专家容量4；若路由计数为8、4、2、2，第一专家有4个token超出容量。Switch的该实现让溢出token经残差旁路，并用负载均衡损失调整分配；其他实现可以选择不同容量或调度策略。[Switch §2.2](https://arxiv.org/html/2101.03961) 由此可以做两种诊断：固定router看通信/容量成本；固定资源看改变均衡约束后的主任务loss和各模态掉路率。仅观察全局平均负载会漏掉“图像token挤到一个专家”的局部问题。

训练显存还需要一张独立的状态账。对1B参数，若权重、梯度和Adam一阶/二阶状态全为FP32，分别占4、4、4、4GB，共16GB，尚未计激活；若使用BF16权重与梯度、FP32 master weights和两份FP32状态，总数仍为16GB。冻结某模块去掉其梯度与优化器状态，checkpointing减少需要长期保存的激活；两种操作作用在不同组成项。这类逐项核算可与[CS336 lecture_02](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_02.py)的资源计算方式对照。

## 2｜01：Q-Former三目标怎样共用参数却使用不同mask

建议扩充01第3节，并接到第4节的SFT样本路径。

### 先固定接口和读出位置

设视觉塔输出 $F\in\mathbb R^{B\times N_v\times d_v}$，可学习queries为 $Q_0\in\mathbb R^{N_q\times d_q}$，文字序列长 $U$。Q-Former的query分支通过cross-attention读取F，query与text分支共享self-attention层。BLIP-2使用32个768维queries；它们的输出 $Z\in\mathbb R^{B\times32\times768}$ 是后续桥接的接口。第一阶段有三种不同的监督读出与query–text可见性。[BLIP-2 §3.1–3.2, Fig.2](https://arxiv.org/html/2301.12597)

把self-attention输入排列为 `[q1,q2,t1,t2,t3]`，mask中的1表示该行query可以读取该列key。三种mask分别为

```text
ITC                   ITG                   ITM
      q1q2 t1t2t3           q1q2 t1t2t3           q1q2 t1t2t3
q1     11  000        q1     11  000        q1     11  111
q2     11  000        q2     11  000        q2     11  111
t1     00  111        t1     11  100        t1     11  111
t2     00  111        t2     11  110        t2     11  111
t3     00  111        t3     11  111        t3     11  111
```

三种目标共享模型参数，依次执行时使用各自mask。ITC里的queries读图但不读文字，文字内部双向编码；ITG让文字读取视觉queries与已出现的文字，queries始终不读文字；ITM让两种输入双向交互。padding还要与这张访问表相交，不能只生成一张全局方阵。

ITC将query输出与text的CLS读出分别投影、归一化。对图像i和文本j，可写 $s_{ij}=\max_q z_{iq}^Tt_j/\tau$，再对图到文、文到图作对比交叉熵。最大值让某个query可以提供匹配证据。例如两个query与正确文本的点积为0.7、0.1，与负文本的最大点积为0.6，取 $1/\tau=10$，图到文对正确文本的softmax概率为 $e^7/(e^7+e^6)\approx0.731$。若queries已通过self-attention读到这条文本，模型就能把文本信息复制进图像表示，ITC将失去独立表示匹配的意义。

ITM输入匹配或不匹配的图文对，让每个融合后的query输出二分类logits $l_q\in\mathbb R^2$，取 $\bar l=N_q^{-1}\sum_q l_q$ 后做交叉熵。此时双方需要交互，才能检查局部图文关系。ITG则使用文本词表头，按 $-\sum_u\log p(t_u|F,t_{<u})$ 监督下一文字；视觉信息只能先由queries读取，再传给文字。若queries读取了未来文字，即使text→text是三角mask，未来信息仍可绕过三角阵经query泄漏。这是需要同时检查四个mask块的原因。

### 第二阶段怎样把表示接到冻结LLM

第二阶段加入 $W_p\in\mathbb R^{768\times d_L}$，得到视觉前缀 $P=ZW_p\in\mathbb R^{B\times32\times d_L}$。对decoder-only LLM，把P接在文本embedding前，训练文字条件生成；对encoder-decoder LLM，将P与文字前缀放入encoder，decoder预测后缀。[BLIP-2 §3.3, Fig.3](https://arxiv.org/html/2301.12597) 阶段一的ITC/ITM读出与阶段二冻结LLM的词表头应在计算图上分别标明。

| 模块 | 阶段一 | 阶段二 |
|---|---|---|
| 视觉塔 | 冻结，提供视觉特征 | 冻结，提供视觉特征 |
| queries与Q-Former | 更新，接收三种目标的梯度 | 更新，接收LLM生成loss的输入梯度 |
| ITC/ITM/ITG读出 | 为各自目标提供监督 | 按本阶段架构移除或不参与该loss |
| 到LLM维度的投影 | 未参与该桥接目标 | 更新 |
| LLM | 未接入第一阶段三目标 | 权重冻结，仍对其输入求梯度 |

冻结参数和切断计算图是两件可单独操作的事：

```text
vision.requires_grad_(False)
llm.requires_grad_(False)
with no_grad():
    F = vision(images)                 # upstream has no trainable input module here
Z = qformer(queries, visual_memory=F, task_mask=mask)
P = projector(Z)
logits = llm(embeddings=pack(P, text_embeddings))  # keep autograd to P
loss = answer_or_caption_cross_entropy(logits, shifted_labels)
loss.backward()                        # updates projector and Q-Former
```

这里若把LLM调用包进`no_grad`，投影与Q-Former也收不到这条loss的梯度。若把阶段一的ITM双向mask误用于生成，又会让teacher forcing读到未来答案。最小验收可以只用2个queries和3个文字位置：检查每个loss的非零梯度落在哪些输入/模块；再随机改变被遮蔽的未来文字，要求过去位置的生成logits保持一致。最后用32/64/128 queries在相同图像处理、数据曝光和训练计算下比较OCR与一般VQA，才能把接口带宽问题变成实验。

## 3｜02：ASR的隐式对齐与RVQ的可训练路径

建议补在波形时钟之后、codec排程之前。识别与重建对应不同的监督问题，两者共享输入时间轴，但输出概率分解不同。

### CTC把未知对齐作为路径求和

声学encoder输出 $H\in\mathbb R^{B\times T\times d}$，分类头给出 $p_t(k|x)$，词表大小为V，加一个blank。路径 $\pi\in(\mathcal V\cup\{\varnothing\})^T$ 的概率为 $\prod_t p_t(\pi_t|x)$；映射 $\mathcal B$ 先折叠相邻重复label，再删除blank。目标文字 $y_{1:U}$ 的概率为

$$
p(y|x)=\sum_{\pi:\mathcal B(\pi)=y}\prod_{t=1}^Tp_t(\pi_t|x),\qquad L_{CTC}=-\log p(y|x).
$$

例如目标为`aa`、输入只有3帧，则唯一合法路径是`a blank a`；`a a a`折叠后只有一个a。若每帧 $p(a)=0.6,p(blank)=0.4$，则这条目标的概率为 $0.6\times0.4\times0.6=0.144$。这也说明相邻重复目标需要额外声学时间位置。[CTC §3–4](https://www.cs.toronto.edu/~graves/icml_2006.pdf)

动态规划避免枚举所有路径。将目标扩为 $y'=[blank,y_1,blank,\ldots,y_U,blank]$，$\alpha_t(s)$表示前t帧结束在扩展位置s的概率。递推为

$$
\alpha_t(s)=p_t(y'_s)\left[\alpha_{t-1}(s)+\alpha_{t-1}(s-1)+c_s\alpha_{t-1}(s-2)\right],
$$

其中 $c_s=1$ 当且仅当 $y'_s$ 非blank且与 $y'_{s-2}$ 不同；越界项取0。初始只允许blank和首label，终点为最后label与最后blank之和。实际计算在log空间用`logsumexp`。张量可为 `[B,T,2U+1]` 的教学实现，时间复杂度约 $O(TU)$；变长batch要使用真实input/target lengths。把CTC loss误写成逐帧对齐交叉熵，会重新要求本来没有的边界标签。

### RNN-T同时推进声学时间与输出位置

RNN-T保留声学时间t，再加已生成文字长度u。encoder产生 $h_t$，prediction network从 $y_{\leq u}$ 得到 $g_u$，joint network例如用 $j_{tu}=\tanh(W_hh_t+W_gg_u+b)$ 后输出V+1种符号概率。完整logits可写成 `[B,T,U+1,V+1]`，所以长语音的训练实现需要关注这部分内存。

在 $(t,u)$ 状态，blank沿声学轴到 $(t+1,u)$，下一个label沿文字轴到 $(t,u+1)$。用与原文一致的一种索引约定，

$$
\alpha(t,u)=\alpha(t-1,u)p(\varnothing|t-1,u)+\alpha(t,u-1)p(y_u|t,u-1).
$$

初始 $\alpha(1,0)=1$，完整概率还乘终点blank；实现应明确起终点约定。[RNN-T §2.3–2.5](https://arxiv.org/html/1211.3711) 相同文字可以在同一声学位置连续发出，且label概率依赖此前输出，这与CTC的帧级条件独立分解不同。流式推理时先缓存声学encoder状态，再在当前t预测label，直到发出blank后推进声学时间；若encoder本身读取未来整句，采用RNN-T loss也不会使前端自动流式化。

### RVQ的离散选择如何训练

令codec encoder输出 $z\in\mathbb R^{B\times T_c\times D}$，Q个码本组成 $E\in\mathbb R^{Q\times K\times D}$；量化后索引为 `[B,Tc,Q]`，向量和 $\hat z$ 与z同shape。递归最近邻已定义后，straight-through路径可写

$$
z_{ST}=z+\operatorname{sg}(\hat z-z).
$$

前向 $z_{ST}=\hat z$，反向对z的局部导数按单位阵近似。decoder的波形/频谱/对抗等loss因此能更新encoder；commitment项 $\sum_q\|r_{q-1}-\operatorname{sg}(e_{q,k_q})\|^2$ 约束各层残差接近已选码字。EnCodec对码本使用EMA更新，并对未使用码字作替换；这条路径与“把码本当普通参数、加codebook loss再用optimizer更新”的VQ方案应分开说明。[EnCodec §3.2、3.4](https://arxiv.org/html/2210.13438)

对某层码本条目k，当前batch分配数为 $n_k$、分配残差向量和为 $s_k$，可以将EMA教学实现写为

$$
N_k\leftarrow\lambda N_k+(1-\lambda)n_k,\quad
S_k\leftarrow\lambda S_k+(1-\lambda)s_k,\quad
e_k\leftarrow S_k/(N_k+\epsilon).
$$

数值例：二维输入 $(1.2,-0.3)$，第一层选 $(1,0)$，残差为 $(0.2,-0.3)$；第二层选 $(0.25,-0.25)$，重建为 $(1.25,-0.25)$，平方误差和为0.005。第一层旧计数2、旧向量和 $(2,0)$，只分配这一个新样本，取 $\lambda=0.9$，则新计数1.9、向量和 $(1.92,-0.03)$，码字约更新为 $(1.0105,-0.0158)$。码本在向残差均值移动，encoder则沿重建梯度与commitment梯度移动；两者的更新来源不同。

训练循环应记录这一分工：

```text
z = encoder(waveform)
choose active Q from the bandwidth schedule
for q in active quantizers:
    select nearest code from detached distances
    accumulate residual commitment and EMA counts/sums
    subtract the selected code from the residual
decode z + stop_gradient(sum_codes - z)
backprop reconstruction/perceptual losses and commitment into trainable networks
apply EMA updates to codebooks; inspect empty/rare codes
```

多码率训练抽取不同前缀码本数，使同一个decoder在不同带宽下工作；使用层数必须进入码率和质量评估。检查失败时应看每层code usage与commitment、重建频谱、不同带宽的梯度，而不是只看总MSE。若STE写成 `stopgrad(z)+q(z)-stopgrad(q(z))` 且q只含离散查表，encoder路径可能被切断；若EMA与optimizer同时更新同一码本，又会改变预期算法。

最后把RVQ索引交给语言模型前，应画出因果排程。一个纯教学delay pattern取Q=3、T=3，令码本q延迟q列：

```text
column        0    1    2    3    4
codebook 0   c00  c10  c20   -    -
codebook 1    -   c01  c11  c21   -
codebook 2    -    -   c02  c12  c22
```

每列内并行输出，列间AR；第一帧全部码在第2列齐备，完整序列需要T+Q−1=5列，展平AR需要9个串行变量。实际decoder是否还读未来块，应接到04的lookahead账，而不能由这张索引表直接推断首个可播放波形的时刻。

## 4｜03/04：时空position IDs、块可见性与播放提交

建议将前三段插入03时间编码节，其余插入04流式实现节。以下把位置、数据到达、缓存和输出提交放进同一个可检查接口。

### 从帧时间到三轴旋转

假设采样后有F帧，每帧编码成 $H_v\times W_v$ 个token，视觉张量为 `[B,F,Hv,Wv,d]`。展平顺序选 `frame → row → column`，所以索引 $j=fH_vW_v+hW_v+w$。保留 `frame_times[B,F]`，再构造 `position_ids[3,B,T]`，三个轴分别是时间、行、列。

为便于实现，先给出一个教学约定：时间单位 $\Delta=0.04$s，$p_t(f)=\mathrm{round}(t_f/\Delta)$，$p_h=h,p_w=w$。采样时间0、0.6、2.0秒对应0、15、50，而不是0、1、2。同帧所有patch共享时间ID；同一空间位置在不同帧上使用相同行列ID。Qwen2.5-VL将MRoPE时间分量对齐实际时间；Qwen2.5-Omni进一步给出每个时间ID对应40ms的音视频对齐方案。[Qwen2.5-VL §2.1.3](https://arxiv.org/html/2502.13923)、[Qwen2.5-Omni §2.2](https://arxiv.org/html/2503.20215) 上述round规则和示例是教学接口，具体checkpoint仍要使用官方processor的位置规则。

将每个head的成对维度分成时间、行、列三组，组内按00的RoPE旋转。设 $q=(q_t,q_h,q_w)$，则两个视觉token的内积由三项组成：

$$
q_t^TR_t(\Delta p_t)k_t+
q_h^TR_h(\Delta p_h)k_h+
q_w^TR_w(\Delta p_w)k_w.
$$

每组分配多少维、使用哪些频率，应从模型配置读取。若text的三轴ID取相同序号，该构造可退回一维位置形式。插入视觉段后的text起点也需要明确；一个教学选择是从前段三轴最大ID加1继续，这与直接使用flatten token个数是不同的编号法。decoder cache应保存确切position IDs或可重建它们的元数据。

```text
for frame f at physical time t[f]:
    for patch (h,w):
        emit visual_token[f,h,w]
        emit coordinates(time_to_id(t[f]), h, w)
apply rotary groups using the three coordinate arrays
save source time and coordinate IDs with the flattened token sequence
```

用同一视频做两次采样，一次保留0、0.6、2.0秒，另一次多加入1.0秒；原来三帧的时间坐标应按同一个时间尺度解释。若只是重新编号成0、1、2、3，插入一帧就改变了其他帧所表达的物理间隔。检查这一点，比只看张量shape更能发现时序错误。

### 空间与时间attention怎样分配计算

设每帧N个token、共F帧。联合self-attention每head约有 $(FN)^2$ 个query–key配对。先在每帧内做空间attention，再对同一空间槽跨帧做时间attention，则约为 $FN^2+NF^2$。取F=16、N=256，前者约16.78M配对，后者约1.11M；投影、MLP与层数尚未计入。节省来自限制这一层的交互方式：时空因素需要通过多层传播，而不是在一层中直接任意配对。若先对每帧所有patch均值池化，再做时间attention，空间排列已经被压缩，无法要求后端从均值中恢复局部物体轨迹。

一个最小对照用相同的四张帧，排列成“物体先进入再离开”与“先离开再进入”；静态对象问题共享答案，顺序问题答案相反。固定可见帧和token预算，分别比较joint/factorized/pooling，才能把时间建模与采样召回分开。

### 位置对齐之后，还要定义哪些数据已到达

每块记录两种时间：事件覆盖 $[t_j^{start},t_j^{end}]$，以及本机单调时钟下的到达时刻 $a_j$。位置编码描述事件时间；在线mask还需要满足数据可用性。决策时刻 $\tau$ 只能使用 $a_j\leq\tau$ 的块。若块内attention是双向的，该块对应的可见内容应等到所需块完整到达后再提交。

Qwen2.5-Omni将音视频按2秒时间片交错，片内先视觉再音频；其音频encoder采用2秒块内attention。输出声学DiT使用当前块、2个lookback块和1个lookahead块。[Qwen2.5-Omni §2.2–2.4, Fig.3–4](https://arxiv.org/html/2503.20215) 2秒输入分片、音频表示的40ms时间单位和声学输出块是三个不同接口，应分别记录长度与缓存。模型报告中的这些设计本身没有定义完整产品的双工取消策略。

可以在实现中给每个token增加块号c和可用时刻：

```text
allowed(query i, key j) =
    key j has arrived before this decision
    AND j belongs to the architecture's permitted context blocks
    AND the within-block mask permits i -> j
```

如果教学系统规定收满2秒才编码，那么首块自然包含最多2秒的数据等待；若使用更小的独立块，则等待与可用上下文一起改变。把输入分片改小以后，encoder mask、训练条件和cache接口也要与之匹配。

### 用一条时间线核算lookahead

构造一个输出codec：每块表示80ms声音，当前块需要下一块提供上下文。假设语音码chunk 0在wall-clock 100ms齐备，chunk 1在180ms齐备，mel计算12ms、vocoder 8ms。于是chunk 0最早在200ms成为可播放波形；只需要当前块的同成本decoder则可在120ms完成。差额80ms来自等待未来码块。2块lookback只使用过去已经存在的信息，主要增加状态与计算；首块还需要规定左侧padding或启动上下文。

```text
code ready:       c0 @100ms       c1 @180ms       c2 @260ms
decode c0:                       [mel12][wav8] -> playable @200ms
decode c1:                                         [12][8] -> @280ms
audio represented: c0 is 80ms; c1 is the following 80ms
```

这组数值是接口手算。实际系统允许阶段重叠时，每个节点的开始时刻等于“资源空闲时刻”和“全部前置数据ready时刻”的最大值；沿依赖图求首个playable节点，比分别相加平均耗时更准确。首包之后还需检查每80ms能否持续交付下一块，避免平均RTF合格但缓冲耗尽。

同步也需要明确策略。例如某视频帧事件时间为2.00秒、到达2.08秒；对应音频段覆盖2.00–2.04秒、到达2.10秒。在2.09秒作决定时可以用该视频帧，但音频尚不可用。一个要求同步窗口的实现可维护watermark $w=\min(w_{audio},w_{video})$，只提交两路都完整覆盖的窗口；等待超时后则显式标记缺失模态。重新把迟到音频编号成“下一段”会造成位置正确、内容错位的隐蔽错误。

### 打断时提交的是用户听到的前缀

每个输出块附 `(response_epoch, chunk_id, audio_start_sample, audio_end_sample)`。用户插话被检测后，作如下转换：

```text
mark epoch e invalid
cancel pending generation for e
remove all unplayed e packets from application and playback queues
continue input capture; reject any late packet whose epoch is invalid
read the final actually-played sample offset from the player
save the heard prefix and interruption event in conversation state
begin a new response epoch using that reconciled state
```

若24kHz播放器只播放了某chunk的6400个样点，实际听到约266.7ms；不能把该chunk后半段对应的未播文字记成已经说出。文字与音频没有逐词对齐时，应保存可追溯的片段边界或不确定前缀，而不是凭完整文本覆盖实际播放状态。测试时向旧epoch人为延迟一个包，要求取消后它被丢弃；再分别测从插话开始到检测、从检测到最后旧音频播放的时间。这样模型的感知速度与播放器的取消速度成为两个可独立改进的量。

## 5｜05/06：速度场、采样器与多峰动作头

建议把连续生成的定义放05，动作条件化部分放06；两篇共享数学记号，但分别计量图像质量和闭环任务结果。

### Flow matching学习什么速度

为避免与DDPM的“$x_0$表示数据”混淆，这里把噪声记为 $\eta\sim\mathcal N(0,I)$，数据记为z，插值时间 $\tau=0$ 为噪声、$\tau=1$ 为数据。最简单的条件路径为

$$
x_\tau=(1-\tau)\eta+\tau z,\qquad u=z-\eta,
\qquad L_{CFM}=\mathbb E\|v_\theta(x_\tau,\tau;c)-u\|^2.
$$

其中c是文本、类别或其他条件；图像latent可以是 `[B,Cz,Hz,Wz]`，$x_\tau,u,v_\theta$有同样shape，时间为 `[B]` 并广播到空间维。训练时随机抽数据、噪声、时间即可形成监督，无须先数值积分出一条模型轨迹。条件flow matching通过可计算的条件路径学习边际流，是FM论文的核心构造。[Flow Matching §3–4](https://arxiv.org/html/2210.02747)

平方回归还给出一个直接解释。对固定 $(x,\tau,c)$，令 $m=E[u|x,\tau,c]$，则

$$
E[\|v-u\|^2|x,\tau,c]=\|v-m\|^2+E[\|u-m\|^2|x,\tau,c].
$$

第二项不依赖v，所以最优预测是条件平均速度。多条噪声–数据配对可能在相近位置相遇，网络学的是它们在该位置的平均；单条条件插值为直线，并不推出模型积分轨迹处处为直线。例如取教学配对 $\eta=-1,z=3$，在 $\tau=0.25$ 得到 $x=0,u=4$；以理想常速度走 $\Delta\tau=0.25$ 到1，恰好等于该条件直线的中点。这只是可手算的一条监督路径。

DDPM常见 $v_{DDPM}=a_t\epsilon-b_tx_{data}$ 的目标由其噪声调度定义；上式 $u=z-\eta$ 由选定插值求导得到。实现中保存 `path_type/time_direction/prediction_type`，才有可能正确互换网络输出与采样方向。

### 网络输出怎样成为下一状态

对学到的ODE $dx/d\tau=v_\theta(x,\tau;c)$，Euler和Heun分别为

$$
x' = x+h\,v_\theta(x,\tau;c),
$$

$$
k_1=v_\theta(x,\tau;c),\quad \tilde x=x+hk_1,\quad
k_2=v_\theta(\tilde x,\tau+h;c),\quad
x'=x+\tfrac h2(k_1+k_2).
$$

Euler每步一次网络调用，Heun通常每步两次。用已知ODE $dx/d\tau=x$、$x(0)=1$、h=0.5比较：Euler为1.5，Heun为1.625，精确解 $e^{0.5}\approx1.6487$。这个有曲率的例子说明高阶更新在增加调用数后可能减少积分误差；训练向量场误差仍需另外评估。比较采样器时同时报告时间表、步数、NFE、每次batch中的条件分支数和wall-clock时间。

对DDPM，模型给出的均值还需接上随机更新 $x_{t-1}=\mu_\theta(x_t,t)+\sigma_t\xi$，并规定最后一步的随机项。DDIM可在同一噪声模型上使用确定性跳步：设 $s<t$，

$$
\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\hat\epsilon}{\sqrt{\bar\alpha_t}},\qquad
x_s=\sqrt{\bar\alpha_s}\hat x_0+\sqrt{1-\bar\alpha_s}\hat\epsilon.
$$

这是DDIM随机系数取0的更新。[DDPM Algorithm 2](https://arxiv.org/html/2006.11239)、[DDIM §4](https://arxiv.org/html/2010.02502) 取 $x_t=1,\hat\epsilon=0.5,\bar\alpha_t=0.25,\bar\alpha_s=0.64$，先得到 $\hat x_0\approx1.1340$，再得到 $x_s\approx1.2072$。由此可手查实现究竟用了累计 $\bar\alpha$ 还是单步 $\alpha$、时间方向是否反转。

```text
training: sample (data, condition), noise and time; regress the declared target
sampling: initialize noise with saved shape/seed
for each pair of times in the declared schedule:
    evaluate the declared prediction type
    convert to the vector/noise required by the chosen update
    apply Euler/Heun/DDIM/etc. with explicit random term
decode the final latent using the training-time scaling convention
```

训练正常但采样失败时，先拿一个二维高斯或一维已知向量场检查这个接口。时间从1走到0却沿正向FM速度积分、把DDPM v直接当FM u、把latent scaling做两次，都会给出shape完全合法的错误采样。固定seed对比后，才扩展到质量、覆盖与提示遵循实验。

### 动作预测为什么需要分布而非单个均值

给定观测和任务c，一次动作块 $A\in\mathbb R^{H\times d_a}$。行为克隆最一般的形式是 $L_{BC}=-E_{(c,A)\sim D_E}\log\pi_\theta(A|c)$。若选择固定方差单高斯，优化等价于MSE；若同一条件下标量动作以相同概率取−1或+1，则

$$
E[(a-A)^2]=\tfrac12(a+1)^2+\tfrac12(a-1)^2=a^2+1,
$$

最优a=0。将左右绕行表示为两个模式时，均值可能落到障碍物中。这是分布表达问题；更精确地拟合单一均值不会改变该目标的最优解。Diffusion Policy以条件去噪生成动作序列，并通过滚动时域执行将序列重新接回反馈。[Diffusion Policy §3–4](https://arxiv.org/html/2303.04137)

| 动作头 | 输出shape示例 | 训练目标与依赖 |
|---|---|---|
| 确定性回归 | `[B,H,da]` | MSE/Huber，单一输出 |
| 每维独立分类 | `[B,H,da,K]` | 每维交叉熵；独立读出可能丢失跨维模式相关性 |
| AR分类 | `[B,H*da,K]` | 按既定顺序条件生成，显式建立前缀依赖 |
| 混合分布 | `[B,M]`权重、`[B,M,H,da]`均值/尺度 | 对混合密度做负log likelihood |
| 连续flow头 | `[B,H,da]`速度 | 对带噪动作块作条件速度回归 |

对于flow动作头，取 $\Xi\sim N(0,I)^{H\times d_a}$，构造 $A_\tau=(1-\tau)\Xi+\tau A$，目标为 $A-\Xi$，网络条件只包含动作开始前可取得的图像、语言和状态。动作区域内部可对整个带噪块双向交互，因为整块是待生成变量；输入未来真实观测或真实未加噪动作，则改变了部署时可用条件。训练时需要按有效动作维度mask掉padding的本体维度和轨迹末尾，分母使用实际有效元素数。

生成分布还有时间一致性要求：每一步独立抽“左/右”，可能在两个合法模式间来回跳；对整块联合采样能表达一条连贯候选，但每次重规划仍需观察模式是否无故切换。最小实验可用二维绕障数据同时测模式覆盖、整段碰撞率和扰动恢复，分布图与任务结果分别记录。

### 预测块长与实际反馈频率

取20Hz底层控制，即 $\Delta t=50$ms，H=16意味着预测0.8秒，执行前k=4步意味着执行0.2秒。如果每次先停下收观测、推理80ms，再执行4步，同步更新周期为280ms，约3.57Hz；底层命令仍按20Hz发送。若采用异步缓冲，每200ms取新观测并在旧块执行期间完成80ms推理，可以尝试每200ms更新，但必须核对新块的起始状态与剩余旧动作。

在理想异步周期下，扰动刚发生在一次观测之后，下一次观测最多等200ms，随后推理80ms，因此新策略反应可能晚约280ms，另加感知/通信/执行调度。记录观测版本、预测起点与真正执行时刻，可以定位这段等待；把过期块放进队列只增加动作数量，不增加反馈。对照实验固定策略与H，只改变k和调度，再报告任务成功、碰撞、反应延迟与每步计算。

## 6｜08：用校准误差和通信张量解释加速

建议替换量化概述，并在端到端测量之前加入并行与调度。一个部署配置同时决定存储格式、实际算子、切分方式与请求调度，四者应有独立记录。

### 校准数据为什么进入量化目标

取线性层 $W\in\mathbb R^{d_{out}\times d_{in}}$，校准输入 $X\in\mathbb R^{d_{in}\times n}$，原输出 $Y=WX$。量化得到 $\hat W$ 后，层输出误差为

$$
E(W,\hat W;X)=\|(\hat W-W)X\|_F^2
=\operatorname{tr}\!\left[(\hat W-W)XX^T(\hat W-W)^T\right].
$$

因此相同的权重误差在不同激活方向上会产生不同输出误差。$XX^T/n$ 反映校准输入的二阶结构；GPTQ按层输出重建目标利用这类信息补偿量化误差。[GPTQ §3](https://arxiv.org/html/2210.17323) 若只最小化 $\|\hat W-W\|^2$，就相当于没有显式考虑目标输入在哪些方向有较大能量。

给一个两通道例子。令一行权重误差为 $\Delta w=(0.1,-0.1)$。对校准输入 $x_{text}=(1,1)^T$，输出误差正好为0；对另一输入 $x_{visual}=(10,1)^T$，误差为0.9，平方为0.81。这里两个输入是构造例子；它说明一套只覆盖某个输入子空间的校准数据可能把重要误差完全隐藏。实际多模态校准应明确覆盖视觉条件、OCR、语音/视频长度和文字请求中的哪些分布，并把校准集与最终质量测试分开。

一种等价缩放把 $W$ 改成 $WD$、输入改成 $D^{-1}X$，其中D是可逆对角阵。浮点输出保持 $WD D^{-1}X=WX$；量化以后则比较

$$
\|Q(WD)D^{-1}X-WX\|_F^2.
$$

AWQ使用激活统计搜索通道scale，目标是降低量化后的输出差异。[AWQ §3.2](https://arxiv.org/html/2306.00978) 例如 $D=\mathrm{diag}(2,1)$ 会把第一权重通道放大、第一输入通道减半；共同量化尺度也可能因放大而改变，因此scale需要在整组输出上评价。局部通道被保护，不代表其他通道的误差固定。

一个可检查的简单校准过程如下；它是教学搜索方案，不冒充完整GPTQ或AWQ实现：

```text
freeze a calibration record-ID list and preprocessing revision
for each quantized layer:
    capture X and full-precision target Y using the declared input path
    for group size, clipping threshold and channel-scale candidate:
        quantize W, execute the intended dequantization/kernel path
        evaluate ||Y_quantized - Y||² / (||Y||² + epsilon)
    choose the candidate on calibration/development data
    save scales, zero-points, group layout, packing and kernel version
evaluate the complete quantized model on the untouched quality test trace
```

如果使用原始浮点输入X作逐层诊断，测到的是该层局部敏感性；若前面各层已量化后再采集X，则还包含上游误差传播。两种实验应分别命名，并最终检查整个模型。

### 有效bit数和实际算子

以对称4-bit、每128个权重一组、每组一个FP16 scale为例，128个权重占64 bytes，scale占2 bytes，有效平均位数为 $4+16/128=4.125$ bits/weight。该例零点固定为0；非对称量化、padding或额外metadata会继续增加字节。7B权重的这个理想打包大小约3.609GB，而不是恰好3.5GB；尚未计未量化层、KV、激活和分配器。

还要记录矩阵乘究竟采用什么路径：weight-only packed weights在kernel内部反量化并与高精度激活相乘；权重和激活同时量化则可采用另一类整数矩阵乘。文件精度标签不足以确定每个阶段的速度。基准至少比较同一请求的质量、权重/KV/峰值显存、TTFT、TPOT，并在batch=1与目标并发下分别测量。

### 一层MLP的tensor parallelism

设输入 $X\in\mathbb R^{B\times T\times d}$，两层MLP为 $Y=\phi(XW_1)W_2$，$W_1\in\mathbb R^{d\times f}$、$W_2\in\mathbb R^{f\times d}$。在p张卡上，把 $W_1$ 按输出列分为 $W_1^{(r)}\in\mathbb R^{d\times f/p}$，把 $W_2$ 按输入行分为 $W_2^{(r)}\in\mathbb R^{f/p\times d}$。每卡计算

$$
H^{(r)}=\phi(XW_1^{(r)}),\qquad
Y^{(r)}=H^{(r)}W_2^{(r)},\qquad
Y=\sum_{r=1}^pY^{(r)}.
$$

第一个矩阵乘之后可以局部激活，第二个之后需要对 `[B,T,d]` 的部分输出做all-reduce。这是Megatron-LM的核心MLP切分。[Megatron-LM §3, Fig.3](https://arxiv.org/html/1909.08053) 门控FFN的两条上投影也需要按相同中间维切分，保持逐元素门控局部完成。

取B=1、decode的T=1、d=4096、BF16，归约张量只有8192 bytes，但每层都要支付通信启动与同步；取prefill的T=2048，同一归约张量成为16MiB。理想ring all-reduce中，p=4时每卡发送体积约为 $2(p-1)/p=1.5$ 倍张量大小，即前者12KiB、后者24MiB，另有接收和实际网络协议。小消息可能受延迟限制，大消息可能受链路带宽限制，故同一种TP切分在prefill和decode中的收益可以不同。

### 数据、流水线、专家并行分别切什么

| 并行方式 | 每卡持有什么/做什么 | 主要交换与取舍 |
|---|---|---|
| 纯data parallelism | 完整模型，处理不同样本 | 训练时归约梯度；本身不降低完整权重容量 |
| Tensor parallelism | 每层矩阵的一部分 | 层内activation归约/收集；降低权重分片容量但引入频繁通信 |
| Pipeline parallelism | 连续的一组层 | stage边界传activation；调度microbatches并处理bubble |
| Expert parallelism | 一部分专家 | 按router的选择dispatch/return，负载不均会产生等待 |

优化器/梯度分片可与DP组合，改变训练状态账；它与“每张卡保留完整状态”的纯DP定义不同。一个仅用于理解bubble的理想前向流水线有p个等长stage、m个microbatches，耗时约 $(m+p-1)\tau$，其中填充/排空的占比为 $(p-1)/(m+p-1)$。取p=4、m=8，占比约27.3%；真实训练的forward/backward调度还需单独建模。B=1的交互推理通常缺少大量独立microbatches，不能把大batch流水线利用率直接移作其延迟结论。

例如7B模型仅BF16权重约14GB，若单卡可用空间不足，纯DP复制无法解决单请求装载；TP或PP可以分片权重，前提是剩余KV/激活与通信时间也满足要求。四卡TP把上述权重主项分为约3.5GB/卡，并不意味着端到端延迟严格变为1/4；每层归约、未切分算子和输入编码都进入新时间账。

### 服务调度和KV布局要独立测量

Continuous batching在某请求完成后及时补入新请求，使不同长度的decode共享运行批次；paged KV则把序列的逻辑块映射到按需分配的物理块，两者改变不同的资源问题。[PagedAttention §3–4](https://arxiv.org/html/2309.06180) 若每块16个token，长度33的序列需要3块，最后一块只用1个位置、剩15个位置是尾块内部余量；分页没有减少33个有效token对应的K/V元素，只改变分配浪费和共享方式。

一个基准trace可以固定短文字、1024视觉token的OCR和长上下文三类请求到达时间，回放相同输入与输出长度上限。依次比较固定batch、continuous batching、再加分页KV，记录每请求排队/编码/prefill/decode、活动batch大小、物理块使用量和失败。若吞吐上升而p95 TTFT变差，下一步应检查调度等待；若只有KV可用容量改善，结果应按并发收益解释。测量落到阶段和请求，优化的作用才可被区分。

## 7｜09：从偏好数据到GRPO更新的一条完整训练链

建议扩充09的RL部分，保留现有DPO推导，并把以下训练路径与其并列。统一输入记为x，其中包含文字、图像以及确定的预处理结果。

### 奖励是学出来的分数还是可执行谓词

若使用偏好对训练奖励模型，令 $r_\phi(x,y)$ 是一个标量，Bradley–Terry负对数似然为

$$
L_{RM}=-E_{(x,y_w,y_l)}\log\sigma\!\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right).
$$

例如两回答分数1.3和0.1，预测偏好概率约0.769，loss约0.263。这个loss更新 $\phi$；策略随后根据采样回答的分数更新 $\theta$。DPO则利用偏好对直接更新策略，省去独立拟合该reward model的步骤。推导与模型角色可对照[DPO §3–4](https://arxiv.org/html/2305.18290)。

程序奖励可将 $r(x,y)$ 定义为“从回答解析到的最终计数与图像标注相等”的0/1谓词。实现要同时保存原始回答、解析结果、标准答案和失败类型。若只检查JSON语法，reward就只测格式；若在图像计数任务中只对数字相等给分，则标注错误、解析歧义仍会改变训练目标。独立人工抽检或另一套验证器可以检查reward与真实任务一致性。

### 三个策略副本各做什么

GRPO的一次rollout使用冻结快照 $\pi_{old}$；正在更新的策略为 $\pi_\theta$；参考策略 $\pi_{ref}$ 用于约束相对漂移。对每个输入x采样G个回答，保存回答tokens、EOS/有效mask、old logp和reference logp。设回答最大长U，则这些数组可为 `[B,G,U]`，奖励为 `[B,G]`。

取每组奖励均值 $\bar r$，明确采用总体标准差 $s_r=\sqrt{G^{-1}\sum_i(r_i-\bar r)^2}$，得到

$$
\hat A_i=\frac{r_i-\bar r}{s_r+\epsilon_{std}}.
$$

Outcome reward下同一回答的有效token共享这个优势。GRPO原始目标使用逐token的新旧策略比率，并在每个回答内部对有效长度归一化。[DeepSeekMath §4.1, Eq.(3), Algorithm 1](https://arxiv.org/html/2402.03300)

$$
\rho_{it}=\exp(\log\pi_\theta(y_{it}|x,y_{i,<t})-\log\pi_{old}(y_{it}|x,y_{i,<t})),
$$

$$
J=\frac1G\sum_i\frac1{L_i}\sum_{t\in valid_i}
\left[\min\!\left(\rho_{it}\hat A_i,
\operatorname{clip}(\rho_{it},1-\varepsilon,1+\varepsilon)\hat A_i\right)-\beta k_{it}\right].
$$

这里 $L_i$ 是回答有效长度，prompt、图像条件和padding不进入这个回答目标。reward、优势、old/ref logp都应detach；当前策略logp保留梯度。整体训练最小化 $-J$。

### 用一组奖励检查clip方向

取G=4、rewards=`[0,0,1,1]`，则均值0.5、总体标准差0.5，优势约为`[-1,-1,+1,+1]`。设 $\varepsilon=0.2$：

- 正优势token若 $\rho=1.3$，两项为1.3和1.2，取1.2；进一步增大概率的收益被clip截住。
- 负优势token若 $\rho=0.7$，两项为−0.7和−0.8，取−0.8；继续降低概率的收益也达到这一侧边界。
- 负优势token若 $\rho=1.3$，两项为−1.3和−1.2，取−1.3；朝错误方向增加概率仍受到惩罚。

若实现对负优势也使用`max`，就会改变这个保守目标。若使用样本标准差，以上优势变为约±0.866，应在配置中记录；采用何种分母属于梯度尺度的定义。

当四个reward全为1或全为0时，中心化优势为0。该组仍可能产生KL梯度，但没有相对成功/失败方向。训练日志应报告nonzero-advantage group比例、各组reward分布和回答长度，才看得出有效训练信号是否稀少。

### KL项的抽样条件

一种常用逐token量为 $u=\log\pi_{ref}-\log\pi_\theta$，$k=e^u-u-1\geq0$。在当前策略分布下取期望，

$$
E_{y\sim\pi_\theta}[e^u-u-1]
=D_{KL}(\pi_\theta\|\pi_{ref}),
$$

其中利用了 $E_{\pi_\theta}[\pi_{ref}/\pi_\theta]=1$。若固定old rollout做多轮更新而策略已变化，这个抽样条件需要通过重权或明确的近似处理；不要把每个条件下得到的样本平均都称为精确KL。参考分布必须覆盖需要比较的输出支持集。标准实现的目标、mask与KL估计应连同checkpoint一起保存。

```text
freeze reference policy for the declared training interval
repeat:
    snapshot policy as old; sample B prompts
    generate G responses per prompt with old
    compute rewards and old/reference logp; store valid response masks
    compute detached per-prompt advantages
    for the declared number of reuse updates:
        recompute current-policy token logp on the stored responses
        compute token ratios, clipped surrogate and declared KL term
        normalize by response lengths, then groups/batch
        backpropagate, clip gradients and update policy
    collect fresh rollouts; log drift, entropy, rewards and validity
```

这一顺序可与[CS336 lecture_17](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_17.py)的生成、reward、logp、centered/normalized reward与clipped loss路径对照。最小实现先用长度固定、答案可验证的排序或计数任务，检查成功回答概率上升、失败回答概率下降，再加入变长EOS、图像预处理与多设备归约。

### 算法更新与完整训练配方分开记录

R1 v1报告区分R1-Zero与R1。R1采用冷启动SFT、推理RL、拒绝采样收集混合数据后SFT、再进行面向多场景的RL；其中蒸馏小模型采用生成数据作SFT。[DeepSeek-R1 v1 §2.2–2.4](https://arxiv.org/html/2501.12948v1) 报告一个多模态“R1式训练”时，应分别列每阶段初始权重、数据来源、哪些模型更新、采样预算与reward，而不是用GRPO这一优化器名字替代全部配方。

视觉必要性需要进入具体数据。例如同一问题“蓝色圆形有几个？”，图像A和B只改变一个圆形颜色，正确答案分别为2和3。reward验证最终计数，评价同时要求两张图都答对；仅文本基线保留相同问题与标签分布。训练和测试按原始场景家族切分，避免A进入训练、B进入测试。保存counterfactual两图的答案、置信或拒答、推理长度与verifier输出，才能判断奖励更新改善的是视觉条件依赖、格式还是语言先验。

## 8｜11：从阈值表到每条样本的训练权重

建议扩充11的过滤、混合与训练卡。以下完整数值例为构造数据，用于让读者亲自选择阈值、核算exposure与loss归一化。

### 先定义过滤器要保留什么

假设目标是筛出“文字被图像支持、可用于训练”的图文对。使用一个固定打分器输出 $s\in[0,1]$；在与最终测试隔离的1000条开发记录上，人工标注其中800条符合目标、200条不符合。记录有事先可取得的来源类别：200条OCR类，其中160条符合；800条其他类，其中640条符合。

用阈值 $s\geq\theta$ 保留记录，计算 $precision=TP/(TP+FP)$、$recall=TP/(TP+FN)$。得到如下教学表：

| 全局阈值 | 保留条数 | TP | FP | 总precision | 总recall | OCR TP/160 | OCR recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 900 | 760 | 140 | 84.44% | 95.00% | 150 | 93.75% |
| 0.30 | 730 | 680 | 50 | 93.15% | 85.00% | 100 | 62.50% |
| 0.40 | 560 | 540 | 20 | 96.43% | 67.50% | 40 | 25.00% |

为了能复算分层选择，再给出分组计数：OCR在上述三阈值的 `(TP,FP)` 分别为 `(150,30)`、`(100,10)`、`(40,2)`；其他类分别为 `(610,110)`、`(580,40)`、`(500,18)`。先规定本次工程约束为总precision至少90%、OCR recall至少80%，三个单阈值均未同时满足。

试用OCR阈值0.20、其他类阈值0.30：总TP为150+580=730，FP为30+40=70，保留800条，总precision为91.25%，总recall为91.25%，OCR recall为93.75%。这解释了按来源分层校准怎样改善已声明的目标，而不是把某个阈值当成普遍“高质量”定义。来源标签如果来自分类器，其误差也应进入开发集检查。

```text
freeze source/family split and the human-label definition
score only the training/development partition with a fixed scorer revision
tabulate TP/FP/FN and retained counts by source/language/task slice
choose one declared threshold policy on development data
freeze the policy and write retained record IDs to a manifest
train downstream models under fixed code and compute budgets
evaluate once on the untouched downstream test protocol
```

人工“可用于训练”标签只是过滤目标的代理；下游效果仍需训练实验。DataComp提供了固定训练代码和计算预算、将数据集作为变量的设计，可用来理解这一步与纯打分准确率的区别。[DataComp §3](https://arxiv.org/html/2304.14108) 比较随机子集、全局阈值和分层阈值时，既要记录保留条数，也要记录去重后独特条数、来源组成、实际训练exposure与下游能力。开发阈值可被选择，最终测试不能继续参与选择。

### 两个数据源的采样概率

设来源A有1,000,000条，来源B为10,000条，来源采样概率为

$$
q_k=\frac{n_k^\tau}{\sum_jn_j^\tau},\qquad
P(\text{record }i\in k)=q_k/n_k.
$$

| 配方 | q(A) | q(B) | 单条A被抽概率 | 单条B被抽概率 |
|---|---:|---:|---:|---:|
| 按数量，τ=1 | 100/101 | 1/101 | 约9.901×10⁻⁷ | 约9.901×10⁻⁷ |
| 按来源，τ=0 | 1/2 | 1/2 | 5×10⁻⁷ | 5×10⁻⁵ |

第二行中每条B被抽概率是每条A的100倍；相较第一行，B自身概率提高50.5倍。两个比较的分母不同，应直接写出倍数含义。

若独立有放回抽取1,010,000条exposures，第一种配方对A/B的期望曝光分别为1,000,000/10,000，每条平均期望1次；第二种对A/B各505,000次，相当于A每条期望0.505次、B每条50.5次。这些是期望，并不表示第一种恰好无重复地遍历一遍。单条被抽中概率p时，D次抽取后至少见过一次的概率为 $1-(1-p)^D$；记录unique IDs才能区分重复训练与新增信息。

### 采样比例怎样进入loss

取每条A含100个有效目标token，每条B含1000个；按来源各50%抽取。在长期累计位置数中，A:B为 $0.5\times100:0.5\times1000=1:10$。若将全batch所有有效token loss求和后除总token数，B占约10/11的目标位置；若先对每条样本的token求均值、再对样本平均，则两来源的记账权重各约1/2。

设第i条样本有效长度为 $U_i$，两种归约明确写成

$$
L_{global\ token}=\frac{\sum_i\sum_{t=1}^{U_i}\ell_{it}}{\sum_iU_i},
\qquad
L_{sample}=\frac1B\sum_i\frac1{U_i}\sum_{t=1}^{U_i}\ell_{it}.
$$

“贡献多少位置”与“实际梯度范数”仍有区别；不同模态的难度、loss尺度和网络路径会改变后者。前者可以从manifest与batch日志精确核算，后者需要记录梯度统计。视觉输入位置没有SFT label时，它们仍产生编码和attention计算，不能从有效目标token数直接推出完整FLOPs。

如果目标是按来源权重 $r_A=0.9,r_B=0.1$ 的样本平均loss，却为采样多样性使用 $q_A=q_B=0.5$，可将样本loss乘 $\lambda_k=r_k/q_k$，即A乘1.8、B乘0.2；其期望恢复 $\sum_kr_kE[L_i|k]$。这是一种明确的importance weighting，思路与[CS336 lecture_14的importance sampling示例](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_14.py)对应。若想要不同目标，则直接定义新的r，不必把每次配比变化解释为无偏修正。

```text
source ~ categorical(q)
record ~ uniform(eligible_IDs[source])
build inputs and labels; count valid target tokens U
sample_loss = sum(valid_token_losses) / U
weighted_loss = (r[source] / q[source]) * sample_loss
aggregate with the declared batch/distributed reduction
log source, record_ID, family_ID, U, lambda and cumulative exposures
```

变长分布式batch还需核对全局分母。若各卡各自按有效token平均，再简单平均梯度，会让有效token少的卡拥有不同隐含权重；应按框架的sum/mean梯度约定推导缩放。用“相同记录、不同padding长度”和“两卡有效长度不同”的教学batch，可以直接检验实际目标是否与公式一致。

### 课程与回放也应进入同一本账

一份最小训练卡可以给两个拟议阶段：阶段一固定视觉塔与LLM，仅更新connector，使用低分辨率与短序列；阶段二选择性更新LLM、提高分辨率，并加入独立记录的纯文本回放。这里的阶段设计是实验变量，不能由其常见性推断效果。为比较是否发生语言遗忘，应固定一套纯文本held-out任务与视觉/OCR任务，按相同训练计算画检查点曲线；同时列每阶段q、loss归约、学习率、可训练模块、视觉长度和独特样本/曝光。训练完成后，读者应能从这些记录重建“模型实际受了多少次、哪类、哪种权重的监督”。

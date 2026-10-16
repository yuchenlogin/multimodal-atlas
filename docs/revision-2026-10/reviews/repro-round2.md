# 第二轮：机制实验、图表与可复现性

审稿日期：2026-10-16。角色：系统实现与实验方法审稿。对象为当前工作树中的 13 篇报告、40 张 PNG、`labs/run_probes.py`、新增的 `labs/check_mechanisms.py` 和 `labs/planning_probe.py`。本轮不是 CLI 代码 review，不取代 adversarial-review；没有运行基础模型、GPU benchmark 或机器人，也不把 DOI 元数据视为论文数值证据。本轮尚未检查即将重建的最终 PDF。

## 实际复算

环境：仓库 `.venv`，Python 3.12.14、NumPy 2.3.4、Matplotlib 3.10.7。审稿代码在临时内存 writer 中运行，不覆盖作者 CSV、JSON 或 PNG。

1. 重跑 8 项机制检查，结果与 `sources/mechanism-checks.json` 的 `checks` 精确相等；19 份原始 CSV 的头、行、数值表示逐字节一致。没有只检查 `passed: true` 字段。
2. 独立读取原始 CSV，并以另一条计算路径核验：复数相位实现 RoPE；枚举记录的 CTC 路径、独立计算 emission 乘积与折叠；以 2×2 正交矩阵逆变换检查 diffusion 参数化；重新由 scores/V 计算稳定全量 softmax；从含 EOS 的实际 token log probabilities 重建 DPO margin；重新按 cluster 聚合配对差并算标准误；由原始 a/c/y 重新解 OLS 正规方程。
3. 重跑 planning probe，包含模型系数、全部 72 条系统×episode 结果和保存轨迹的完整 JSON 与落盘精确相等。另从保存的 trial 0 状态轨迹反推动作，独立检查动作限幅、每步动力学和累计代价。全 episode 的动作轨迹没有保存，因此独立状态回放仅覆盖显式保存的三个 trial 0 轨迹；完整脚本重跑覆盖全部 episode。

| 检查 | 实际核验结果 | 解释边界 |
|---|---|---|
| 00 RoPE/cache | 相对分数误差 1.78e-15；cache 输出误差 2.22e-16；错误 query 位置造成约 1.041 差异 | 合成单 head，无模型速度结论 |
| 01 loss gradient | CLIP/SigLIP logit 有限差分最大误差约 1.01e-10/1.67e-10；bias 误差 7.23e-11 | 单向 CLIP 行平均，SigLIP 全 pair 求和再除 N；非 encoder 梯度 |
| 02 CTC | 全 27 路径总概率 1；A、AA、AB 概率为 0.341、0.042、0.211 | 三帧合法分类分布，不是语音识别结果 |
| 05 参数化 | 所有逆变换误差小于 8e-16；alpha=0.02 时误差放大约 7 | 已知向量/精确 target；非生成质量 |
| 08 online softmax | block=1/4/7/19 均与 full 输出相差小于 2e-16 | 全 scores 为参考刻意保存；tile 字节数不是实测峰值 RAM |
| 09 长度归一化 | 6/21 个有效 answer token（含 EOS）；sum margin=-0.1，token mean margin=1/140 | 合法定常 categorical 模型；没有执行 DPO 训练 |
| 10 cluster | 同样 80/10/20/90 记录计数；item SE=0.027225，cluster SE=0.088109；4000 次区间分别 [0,0.105]、[-0.1,0.2] | 新构造的 20 个完美相关 cluster；不追溯否定原独立 item toy |
| 12 adjusted OLS | naive≈0.934986，adjusted≈0.202325，randomized≈0.202418 | 新独立 seeds；与旧 probe 的 0.936760/0.200958 分开 |
| planning | PD=24/24、mean cost=11.891695；fitted MPC=24/24、11.810523；shuffle=0/24、11518.513707，24 次越界 | 无噪声已知线性系统，模型族完全匹配；不是视觉/机器人结果 |

planning 正文的动力学、12 步 horizon、256 候选（含 21 条常值）、24 配对 episode、80 步、成功和越界规则均与实际代码/输出一致。两种 MPC 在每个配对 episode 使用同一候选随机流；状态分化后它们评价的是各自状态下的同一动作候选，不能解释成共享轨迹。正文正确没有将 11.81 对 11.89 的微小代价差声称为 MPC 普适优势。

## 第一轮问题的关闭情况

- **M1 原箭头反向/消失：关闭旧实现问题。** 现在按矩形边界裁剪，逐图可见原来 02、09、12 的关键前向箭头朝向正确。新增的 05 监督箭头仍有独立布局问题，见下文 R2-M1。
- **M2 pooling 分支：关闭。** 01 图明确从保留 patch 维的 encoder 输出分到 pooling 与 connector，生成路径不再经过池化向量。
- **M3 状态时间：关闭。** 06 保存的是更新结束状态，横轴已改为 0.05…4.0 秒；JSON 写出扰动在 1.5 秒更新之前注入。正文的扰动时刻与该约定一致。
- **M4 依赖/完整 CLI：关闭。** NumPy、Matplotlib 已固定；README 给出 probe→机制检查→planning→绘图→HTML→PDF 顺序。
- **M5 原文版本：关闭记录缺失问题。** 13 组均已有显式 vN 和定位记录。逐表数值真实性由独立证据审稿核验，本轮不以版本字段代替原文核查。

## 当前仍须修订的图面问题

### R2-M1（medium）：05 的 target noise 箭头被 DiT 框遮挡

`05-method.png` 中从 `noise + schedule` 到 `regression loss` 的目标监督路径与同一横排的 DiT 预测路径共线；线段在中间 DiT 框后面，被框遮住，其 `target noise` 标签直接压在 `DiT(xt,t,condition)` 字上。读者看不到预测和 target 两条独立输入。应增加独立 noise-target 节点，或将监督边从顶部绕过 DiT，同时将标签移到空白处。修后需要看原尺寸图，不能只通过脚本运行成功验收。

### R2-M2（medium）：08 的 compute/timing 标签叠合

`08-method.png` 的 GPU→prefill `compute` 与 decode→end-to-end metrics `timing` 标签在同一位置，原尺寸看成 `comptiming`。数据路径本身方向正确，但标签已不可辨。请分别调整标签位置或边路由，保持计算供给与测量两个接口独立可读。

### 编辑残留（已在审稿期间删除）

首次读到 02 的“建议补在波形时钟之后、codec排程之前”和 03 的“建议将前三段插入03时间编码节，其余插入04流式实现节”；已告知主编，随后工作树检索确认删除。这类给编辑的指令不能进入读者正文。

## 40 张图逐图检查

通过 5 张 contact sheet 浏览全部 40 图，并对 00/05/08/09 方法图查看原尺寸。方法示意、作者报告值与教学 probe 都有明确类型标识；各柱图从零起始，标签能对应 JSON，未把不同设置连成模型能力增长曲线。图内字采用英文以保证跨环境字体，正文有中文解释；完整来源和 caveat 放在正文 caption 与 JSON，图脚用于定位，不单独承担协议说明。

| 章节 | evidence | method | probe / additional |
|---|---|---|---|
| 00 | 通过 | 前向方向正确；短 QK→softmax 箭头仍可辨 | KV 预算通过 |
| 01 | 通过 | 两分支通过 | loss 基线量纲已说明 |
| 02 | 通过 | 编码、残差与解码方向通过 | 标量量化明确 |
| 03 | 通过 | 采样/时间/查询路径通过 | MC 与解析线重叠，建议改散点（low） |
| 04 | 通过 | cancel 分别作用 Talker/输出，通过 | completion delay 单位正确 |
| 05 | 通过 | R2-M1 待修 | 误差放大与 alpha 标签通过 |
| 06 | 通过 | 闭环方向通过 | 更新结束时间已修 |
| 07 | 通过 | 验证/恢复路径通过 | retry 的强假设正文已列 |
| 08 | 通过 | R2-M2 待修 | 权重项算术强度通过 |
| 09 | 通过 | log-ratio 与 loss 路径通过 | 导数而非训练收益 |
| 10 | 通过 | paired/cluster 接口通过 | 旧 4000 次 CI 显示精度已缩短 |
| 11 | 通过 | provenance→split→train/eval 通过 | family zero 线应在 legend 标 analytic（low） |
| 12 | 通过 | 干预检查路径通过 | OLS 和新增 planning 的范围分别正确 |

## Low 与验收边界

- 03 曲线的 MC 与解析结果几乎重叠，建议 MC 用散点；这是可读性补充，不是数值不符。
- 11 的 family-disjoint 零线按定义直接设零，建议图例标 `analytic zero baseline`，避免暗示另跑了一套 family split。
- 旧 13-probe 使用一条共享 seed=336 的 RNG 流；完整 CLI 可精确复算，但单独复制 10 并重置 336 不会得到相同尾部分位点。新 8 项检查已用独立 spawn keys；README/正文最好说明两套 seed 约定。
- 新 planning 图目前出现于正文图 4，而后面 evidence 仍称图 3。建议按阅读顺序重排编号。
- 当前尚未看到最终 10-cluster 补图及正文中引用新增 mechanism-checks 的最终版本，待主编通知后补核；不能提前声称此部分已经过图面审稿。
- 最终网页缩放/移动端与 PDF 字体、页内图尺寸属于最终渲染检查，不能由 PNG 可读推定已通过。

当前结论：新增机制检查和 planning 的真实性与数值复算通过；上述 2 个 medium 图面问题须修后复看，方可关闭本轮。

## 修订后终核与 closure

同日第二次查看更新后的 05-method、08-method、10-cluster、12-planning 原尺寸 PNG，重读机制结果引用段，并独立从 bootstrap CSV 计算区间。

- **R2-M1 关闭：** 05 的监督边改成顶部曲线，target 标签与 DiT 正文分离；预测和监督路径现在都可辨。
- **R2-M2 关闭：** 08 删除重叠的 timing 标签，compute、read、cache ready 标签清楚；测量路径的箭头仍保留。
- 新增 **10-cluster** 检查通过：每组 4000 次；item 范围 [-0.045,0.165]、95% percentile [0,0.105]；cluster 范围 [-0.2,0.4]、95% percentile [-0.1,0.2]。绘图范围覆盖所有观测值，横轴明确为百分点，图例、已知构造与正文一致。正文明确这组使用新独立子流，不能替换原独立 item probe 的落盘区间。
- 00/01/05/08/09/10/12 新增机制结果均与 JSON/CSV 已核验数值一致；12 的 adjusted OLS、新随机动作样本与旧图样本分开。更正本审稿前文表格 05 的舍入描述：所有参数化逆变换中的最大误差约为 **8.88e-16**（上界可写 8.89e-16），不是严格小于 8e-16；正文所写 8.88e-16 正确。
- 检索确认给主编的插入建议已删除；脚本检查 13 篇的图号均按正文出现顺序从 1 连续递增。12 的 planning/evidence 现为图 3/4。
- 图总数现为 **41**。逐个重算 `sources/figure-manifest.json` 中 41 个 PNG 的 SHA256，与落盘文件全部一致。前次已看过其余 40 图，本次补看新增 cluster 图及修改图。

剩余 low 精确措辞建议：05 图同时允许 `predict eps or v`，监督箭头与 loss 可从 `target noise` / `vs sampled noise` 改为 `known target (eps / v)`，以覆盖正文的两类参数化。此项没有否定 epsilon 路径的正确性。前述 03 散点、11 analytic-zero 图例与旧 RNG 说明仍是可读性补充，不影响已验证数值。

**本轮结论：内容、PNG 图面与已执行 CPU 实验范围内，没有尚未解决的 high/medium；通过本角色审稿。** 这不等于最终 PDF 已验收，也不代替来源审稿或 adversarial CLI 代码 review。

最终 PDF QA 建议：除编译和 missing-glyph 日志外，优先实看 05 顶部监督曲线是否被图框裁掉，08 方法图缩到正文宽度后文字，10 两种区间与长图注是否跨页，12 三栏 planning 的坐标和成功计数；检查新增深推导长公式与 reference URL 的溢出。每篇至少检查含方法/证据图的一页与长公式页，研究方案重点检查状态标签与旧实验数字确已撤回。

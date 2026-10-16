# 来源核验记录

访问日期：2026-10-16。主证据包括 `evidence.json` 的 13 组原论文数值，以及 `core-reading.json` 的机制来源。当前正文引用的 35 个唯一 arXiv ID 全部有真实 DataCite 身份交叉核验；CTC 使用作者网站的原版论文。此记录不为旧稿“370+ 引用全部核验”的说法背书。

## 检索失败与替代路径

arXiv Atom API 第一次超时，重试返回 HTTP 429；Semantic Scholar batch 也返回 HTTP 429，没有将这些调用记作成功。随后逐篇读取 arxiv.org 官方全文，并使用真实 DataCite REST endpoint `https://api.datacite.org/dois/10.48550/arXiv.<id>` 核对标题、DOI、年份及作者。35 条成功记录在 `primary/datacite-verified.json`，可用 `scripts/verify_primary_metadata.py` 重新请求；任何未核对或标题不匹配的记录都会使命令停止，且不会覆盖上次完整结果。

SigLIP HTML 不可用，改读官方 PDF：Algorithm 1 的 loss 除以 batch size n，不是 n²。PagedAttention HTML 读取失败，改读官方 v1 PDF。MMMU 与 Chameleon 大 PDF 超过浏览工具读取限制，使用 curl 下载官方 PDF，再以 Poppler 提取和渲染对应页复核。POPE 编号为 2305.10355，准备检索时误配的 2305.10403 实际为 PaLM 2，未保留为 POPE 来源。

## 逐表复核与版本

13 组结果全部锁定到实际读到的 vN，包含表格位置、单位、模型和条件；版本解析记录见 `primary/source-versions.json`。机制来源中已锁定的版本也保存在 `core-reading.json`，其余注明本次访问日期。论文 DOI 核验只确认来源身份，不能代替逐表核验，更不是本站复现模型结果。

独立审稿纠正了两项表格误读：MMMU v4 Table 5 的 Art & Design 子集 44.2/51.7/65.8 曾被当成 overall；最终采用 B.1、Table 4 的 validation overall 34.4/36.4/56.8（900题）。Chameleon 的 60.4 与 51.6 曾混用不同分母；最终采用 v2 Tables 11–12 同一 overall 行 58.8/51.6（1048 prompts），注明基线接入 DALL-E 3、平局计半次。原文摘要/正文与完整表格的口径冲突在报告里保留说明。

CLIP 三项输入均为 224px；RT-2 为未见物体/背景/环境条件，而非新任务；OSWorld 12.24% 是 GPT-4 + accessibility tree；DataComp 的最终数据集、上游候选池和累计 exposure 分开命名。其他已核对的差异包括 Chinchilla 摘要 67.5/表格 67.6、DiT 的 FID/recall 取舍、FlashAttention ± 数字不冒充95%区间。

## 证据类别

原论文数字由原表支撑；13 个基础 probe、8 项机制检查和线性 MPC 教学系统均实际在 CPU 执行，有生成器与 JSON/CSV。它们不能冒充基础模型训练、GPU 测速或实体机器人试验。尚未执行的大模型与机器人方案明确标注状态，不填写预期提升。

CS336 官方课程与 executable lectures 用作组织参照：从对象与假设推到实现、资源与测量；没有把课程材料当成本项目原创研究。审稿原始意见、修订与限制见 `docs/revision-2026-10/reviews/`。

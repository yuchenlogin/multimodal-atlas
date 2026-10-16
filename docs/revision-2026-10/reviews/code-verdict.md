# CLI adversarial code review · 最终判定

## Intent

将 13 篇报告改成可追溯、可复算的技术报告，正确发布共享源稿生成的 HTML/PDF，并撤回旧研究稿中没有真实实验支持的结果。教学 CPU 计算不承担基础模型复现的证据责任。

## Verdict: PASS

三位真实 `codex exec` reviewer 已分别从 Skeptic、Architect、Minimalist 角度审查代码及构建。去重后 4 个初轮 medium 与 1 个 low 已修正；一次聚焦复核再发现 2 个浏览器验证 medium，也已修正并经第二次聚焦 CLI 复核关闭。没有剩余已确认的 high/medium。

## Findings 与 Lead Judgment

| 问题 | Lens | 判断与修正 | 验证 |
|---|---|---|---|
| PDF 链接标签含 inline code 时，递归格式化丢失 placeholder 所属 stash | Skeptic / Architect | 接受；递归共享 token 所属上下文 | 含 inline code 的链接标签回归通过，代码中的 Markdown 链接仍保持字面量 |
| 全量/单篇 PDF 构建共用 TeX 输入与发布路径，可互相覆盖 | Skeptic | 接受；在整个转换、编译与发布阶段持有进程锁 | 已持锁时第二次构建退出，未改 TeX/PDF；平台范围写入 README |
| 失败浏览器检查可能留下旧成功记录 | Skeptic | 接受；开始即写 running，失败写 failed，保存输入哈希 | 最终成功记录与当前输入逐项匹配；故障注入得到 failed |
| 可配置 seed/draws 可覆盖发布结果，而文中数字固定 | Minimalist | 接受；发布生成器固定配置，拒绝 CLI 覆盖 | 重生成后数值不变，原始 CSV/生成器哈希核对通过 |
| 同一导航目标反复 GET | Minimalist | 接受；使用 Set 逐 URL 检查一次 | 全页渲染检查保留，本地链接仍逐唯一目标验证 |
| 浏览器 manifest 漏掉用来计算预期图数的 Markdown 源稿 | 聚焦复核 | 接受；纳入 13 篇 report.md 与 proposal.md | 最终记录含源稿 SHA256 |
| 写 passed 早于 browser/server 清理，清理失败仍可能留成功状态 | 聚焦复核 | 接受；清理完成后才写 passed，外层错误处理写 failed | 在真实 browser.close 后注入异常：退出码1、记录failed且保留输入哈希；CLI终核PASS |

没有将已确认的问题作为风格争议拒绝。审稿中读到的旧渲染结果、Python最低版本、图注方括号等缺口均已修正后重新验证，原始意见保留供追溯。

## What Went Well

- 三位 reviewer 均实际核对了 CPU 结果、原始 CSV、图像哈希、源稿与生成 TeX/HTML；未把只读代码审查冒充浏览器或 PDF 视觉检查。
- 数据/机制/未执行方案的分类保持一致，原独立研究稿的虚构结果已撤回。
- 失败调用和缺失的 skill 资源均明确记录，没有补造审稿或执行结果。

## 调用与限制

使用 `/Users/yuchen/.agents/skills/adversarial-review/SKILL.md` 的真实 CLI 路径；该安装没有 `brain/principles.md`，因此只使用其实际提供的 lens 与 principle 名称，未捏造原则内容。原始报告的行号对应审阅时工作树，文件链接改成仓库相对路径。

最初继承全局 gpt-5.5 的三次调用都因账户不支持而失败，未计入评审；读取本地实际模型目录后，仅对本次命令覆盖为 gpt-6-astra，三位 reviewer 成功返回。后续一次较大范围复核因使用额度停止，也未计为通过；改用聚焦修复范围、实际可用的 gpt-6-sol CLI 完成两轮复核（第一轮REJECT、第二轮PASS），未修改用户全局配置。

三位初轮报告分别为 `code-skeptic.md`、`code-architect.md`、`code-minimalist.md`；聚焦复核为 `code-closure-focused.md` 与 `code-closure-final.md`。后者只确认针对修复，不声称重新逐行审查全部报告内容。最终成品渲染由主线程另行验证。

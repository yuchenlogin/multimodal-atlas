# 公式渲染修复：代码复核

日期：2026-10-13。Reviewer CLI：`codex exec --sandbox read-only`。

## Intent

修复 13 篇报告在 Markdown → HTML → MathJax 与 Markdown → LaTeX → PDF 两条路径中的公式排版问题，重新生成产物，并使仓库能独立构建。

## Verdict: PASS

按 `adversarial-review` skill 调用了 Skeptic、Architect、Minimalist 三个 CLI reviewer，三个非空结果文件均已读取。未发现高或中等级问题，两项低等级问题已修正。

## Findings

1. **[low, resolved] 测试依赖默认文本编码。** Skeptic 与 Minimalist 均指出，中文源稿测试应明确用 UTF-8。已修复所有测试文件读写，并重新通过测试。Principle：`prove-it-works`。
2. **[low, resolved] PDF 直接复制可能覆盖有效文件后中断。** Skeptic 指出 `shutil.copy2` 的目标文件写入不是原子操作。现在先在 `assets/` 同一文件系统中的临时目录完成编译，再用 `Path.replace` 原子替换 PDF；已实际编译一份报告验证。Principle：`serialize-shared-state-mutations`。

Architect 在最终代码中未提出可执行问题。

## What Went Well

- 源稿公式与生成 HTML 逐项一致，浏览器中全部 589 个公式渲染成功。
- Markdown、图片、转换器与模板的路径可在单独克隆的仓库中解析。
- 最终 13 份 PDF 编译日志没有缺字或行间公式溢出。

## Lead Judgment

接受两项低等级问题：它们分别影响验证的可移植性与已生成 PDF 的保存完整性。均通过小范围修改解决，不改变公式含义。

Skill 引用的 `brain/principles.md` 与关联原则文件未随本地 skill 安装提供；没有虚构其内容。此次 reviewer 使用了 skill 中实际存在的三种 lens，以上 principle 名称来自 lens 定义。

## [1.1.0] - 2026-10-13

### Features
- 修复并重新生成 13 篇报告的 HTML 与 PDF 公式排版。
- 补齐报告 Markdown 源稿、PDF 转换器、构建依赖与渲染回归检查，使仓库可独立构建。

### Design Rationale
- 用 Arithmatex 在 Markdown 解析阶段保护 TeX，避免斜体、属性列表和转义规则修改公式。
- MathJax 仅处理标记的公式元素，并固定版本为 3.2.2。
- PDF 同时支持单行与多行 `$$`；编译成功后再替换对应的发布 PDF。

### Notes & Caveats
- 网页 MathJax 依赖 CDN；PDF 重建需要 Tectonic，首次编译可能下载 TeX 包。
- 长行间公式在自身区域内滚动；保留原稿的公式含义和报告内容。
- 本次不重建 `paper/` 中的独立论文。

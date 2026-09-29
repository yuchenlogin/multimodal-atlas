# 多模态图谱 · Multimodal Atlas

面向有深度学习基础读者的多模态大模型全景知识库：13 个方向、293 页深度报告、370+ 条经核验的文献引用，从 Transformer 基础到 2026 学术前沿。

**在线阅读**：https://yuchenlogin.github.io/multimodal-atlas/

## 目录

- `index.html` — 首页（知识地图 / 报告卡片 / 演化时间线）
- `reports/<topic>/index.html` — 13 篇专题报告（MathJax 公式、深色模式、阅读进度）
- `paper/index.html` — 顶会风格论文《Verifier-Gated Visual Decoding》在线阅读
- `assets/` — 架构图与全部 PDF 版本
- `build.py` — Markdown → HTML 构建脚本（`python build.py` 重新生成报告页）
- `css/site.css` / `js/site.js` — 设计系统与交互

## 内容来源

报告由 Codex + sub-agent swarm 于 2026-09 调研撰写，全部 arXiv 引用经独立审计（见仓库 `logs/citation-audit.md`）。论文实验数字为内部协议设计稿，已在 Limitations 声明。

## License

内容仅供学习研究使用。

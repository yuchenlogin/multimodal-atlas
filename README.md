# 多模态图谱 · Multimodal Atlas

面向有深度学习基础读者的多模态大模型全景知识库：13 个方向、293 页深度报告、370+ 条经核验的文献引用，从 Transformer 基础到 2026 学术前沿。

**在线阅读**：https://yuchenlogin.github.io/multimodal-atlas/

## 目录

- `index.html` — 首页（知识地图 / 报告卡片 / 演化时间线）
- `reports/<topic>/index.html` — 13 篇专题报告（MathJax 公式、深色模式、阅读进度）
- `paper/index.html` — 顶会风格论文《Verifier-Gated Visual Decoding》在线阅读
- `assets/` — 架构图与全部 PDF 版本
- `reports/<topic>/report.md` — 报告源稿，与生成页面一起保存在仓库中
- `build.py` — Markdown → HTML 构建脚本
- `build_pdf.py` / `tex/` — Markdown → LaTeX → PDF 构建脚本与模板
- `css/site.css` / `js/site.js` — 设计系统与交互

## 内容来源

报告由 Codex + sub-agent swarm 于 2026-09 调研撰写，全部 arXiv 引用经独立审计（见仓库 `logs/citation-audit.md`）。论文实验数字为内部协议设计稿，已在 Limitations 声明。

## 重新渲染

在仓库根目录执行；有 Conda 环境时先激活环境，也可使用 Python 3.10+ 的 venv：

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python build.py
python -m unittest discover -s tests -v
```

重建全部报告 PDF 还需要安装 [Tectonic](https://tectonic-typesetting.github.io/)。首次编译需要联网下载 TeX 包：

```sh
python build_pdf.py
```

PDF 输出到 `assets/`，中间 LaTeX 和编译日志在 `tex/`（不提交到 Git）。`paper/` 的独立论文 PDF 不由此脚本生成。

验证浏览器中的实际公式渲染：

```sh
npm ci
npx playwright install chromium
npm run test:browser
```

此检查使用与 GitHub Pages 一致的 `/multimodal-atlas/` 路径，遍历全部报告，验证公式全部渲染、无 TeX 错误、桌面和手机页面无横向溢出。截图与结果写入 `test-results/`。MathJax 3.2.2 从 CDN 加载，浏览器检查需要联网。

## 公式格式修复（2026-10-13）

网页旧构建脚本把 TeX 直接交给普通 Markdown 解析：下标 `_` 被当成斜体，`attr_list` 又把 `{...}` 当成 HTML 属性。例如 SigLIP 损失变成 `<em i_j="i,j">…</em>`，公式字符顺序和边界遭到破坏，MathJax 无法补救。

现在使用 [Arithmatex](https://facelessuser.github.io/pymdown-extensions/extensions/arithmatex/) 在 Markdown 转换中保护公式，输出独立的 `.arithmatex` 元素和 `\(...\)` / `\[...\]` 分隔符。MathJax 仅扫描这些元素，避免把代码和价格中的 `$` 当成公式。长行间公式在阅读栏内部横向滚动。

PDF 旧转换器只识别分隔符单独成行的 `$$` 块，同一行 `$$公式$$` 被当成行内公式并留下多余 `$`。现已统一转换成 `\[...\]`；同时修复正文 `π`、`≈`、`≠`、`↔` 的缺字，以及代码中的中文字体回退。完整原因与验证结果见 [公式修复记录](docs/math-rendering-fix.md)。

## License

内容仅供学习研究使用。

# 多模态图谱 · Multimodal Atlas

13 篇面向有深度学习基础读者的技术报告。每篇围绕一个可检验的问题，展开张量接口、训练目标、实现边界、资源核算与实验设计。讲解组织参考 Stanford CS336 的原理—实现—测量路径；本项目与课程无隶属关系。

[在线阅读](https://yuchenlogin.github.io/multimodal-atlas/) · [本轮修订设计](docs/revision-2026-10/plan.md) · [审稿记录](docs/revision-2026-10/reviews/)

## 内容与证据

- `reports/<topic>/report.md`：13 篇源稿；同目录 `index.html` 为生成页面。
- `assets/technical/`：原创技术图，分为机制示意、原论文结果与 CPU 教学实验。图注说明证据类型和限制。
- `sources/evidence.json`：13 组原论文数值及版本、表格位置、指标与协议边界。
- `sources/core-reading.json`、`sources/primary/`：机制来源和真实 DataCite API 的交叉核验结果。元数据核验仅确认论文身份，数值另查原始表格。
- `sources/retrieval-log.md`：访问失败、替代渠道和复核记录。
- `labs/`：实际可执行的 CPU 教学计算；不代表大型模型训练、GPU 基准或机器人复现。
- `paper/proposal.md`：**未执行的研究方案**，与报告使用相同 HTML/PDF 构建链。

旧版独立 VGD 页面和 PDF 曾包含没有真实实验支持的性能数字。本次已撤回相应数字及优越性宣称，将原下载入口替换为明确标注状态的研究方案。旧版“370+ 引用全部核验”等宣传性统计也已撤下；当前核验范围以 `sources/` 的实际记录为准。

## 从源码复算和重新渲染

需要 Python 3.11+（本次验证环境为 Python 3.12.14）；使用 Conda 时先激活自己的环境。以下以 venv 为例：

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python labs/run_probes.py
python labs/check_mechanisms.py
python labs/planning_probe.py
python scripts/make_technical_figures.py
python build.py
python -m unittest discover -s tests -v
```

实验生成器会写入 `sources/`；绘图生成器刷新 `assets/technical/`，然后 HTML 构建读取 Markdown。各脚本都不调用模型服务，也不会生成“模拟的模型分数”。论文结果图读取手工逐表核验的数值，不由教学实验代替。

要独立重查论文元数据，运行以下联网命令；失败会显式停止，不会把 HTTP 成功当成标题核验：

```sh
python scripts/verify_primary_metadata.py
```

在 macOS/Linux 上重建 13 篇报告和 1 份研究方案 PDF 需要 [Tectonic](https://tectonic-typesetting.github.io/)，首次编译会下载 TeX 包：

```sh
python build_pdf.py
# 修改单篇后可选择重建：
python build_pdf.py 01-vision-language
```

最终 PDF 保存在 `assets/`；生成的 LaTeX 和日志在 `tex/`，不提交。各 PDF 编译成功且没有字体缺字后才替换对应成品；进程锁阻止全量与单篇构建互相覆盖。研究方案继续使用旧的 `assets/paper-verifier-gated-visual-decoding.pdf` URL，避免旧链接继续返回错误结果。

验证浏览器中的真实 MathJax 渲染、图像与图注，以及桌面/手机布局：

```sh
npm ci
npx playwright install chromium
npm run test:browser
```

检查使用 GitHub Pages 的 `/multimodal-atlas/` 路径。MathJax 3.2.2 从 CDN 加载，浏览器检查需要联网；已有兼容 Chromium 时可通过 `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` 指定。结果与截图写入 `test-results/`。

## 排版与页面

`build.py`、`home.template.html`、`css/site.css`、`js/site.js` 负责静态页面，首页卡片与报告导航由同一份 metadata 生成。`tex/md2tex.py` 与 `tex/report_preamble.tex` 负责 PDF。源稿中的 TeX 在 Markdown 解析前由 Arithmatex 保护，避免 `_`、`{...}` 被当成斜体或属性。

最初截图中的公式故障来自 Markdown 先破坏 TeX、PDF 又漏识别同一行 `$$...$$`。其根因及回归验证见 [公式修复记录](docs/math-rendering-fix.md)。后续内容修订继续使用这条保护链，并同时检查网页与 PDF。

内容仅供学习研究；引用原论文时请遵循对应来源的许可与署名要求。

#!/usr/bin/env python3
"""Build multimodal-atlas site: convert reports/*.md -> HTML pages with shared shell."""
import re, os
from html import escape
from pathlib import Path
import markdown
from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "reports")

META = {
 '00-foundations': ('Transformer 的形状、目标与资源预算', '基础架构', '从 attention 张量、位置编码与 MoE，到训练计算和 KV cache', '🏛️'),
 '01-vision-language': ('视觉信息怎样进入语言模型', '视觉与语言', '对比目标的梯度、Q-Former masks、视觉接口与答案监督', '👁️'),
 '02-audio-speech': ('音频表示、对齐与语音生成的时间尺度', '音频与语音', '从波形、CTC/RNN-T 到残差量化和流式生成', '🎧'),
 '03-video': ('视频理解首先是时间信息的保留问题', '视频理解', '采样遗漏、时空位置、长度预算与时序对照', '🎬'),
 '04-omni': ('全模态系统的表示共享与流式调度', '全模态系统', 'Thinker–Talker、同步、因果窗口和打断状态', '🌐'),
 '05-generation': ('从去噪目标到可执行的生成采样器', '多模态生成', 'DDPM 推导、DiT、flow matching 与离散更新', '🎨'),
 '06-embodied': ('从视觉语言动作预测到闭环控制', '具身与动作', '动作契约、生成头、动作块与反馈稳定性', '🤖'),
 '07-agent-tooluse': ('从多模态观察到可恢复的工具执行', '工具与智能体', '状态、类型契约、事务边界与任务成功率', '🧭'),
 '08-efficiency': ('从资源模型定位多模态推理瓶颈', '效率与部署', 'Roofline、在线 softmax、量化校准和并行通信', '⚡'),
 '09-alignment-safety': ('偏好优化能否改善视觉事实性', '对齐与安全', 'DPO、GRPO 训练链与视觉反事实评测', '🛡️'),
 '10-benchmarks-eval': ('多模态评测如何产生可信结论', '评测方法', '估计对象、配对 bootstrap、cluster 与判分器', '📏'),
 '11-data-training': ('数据选择如何改变训练分布', '数据与训练', '来源、重复族、采样权重和固定预算对照', '🧪'),
 '12-frontier': ('把前沿主张改写成可检验的实验', '研究问题', '共享训练、视觉推理与动作条件世界模型', '🔭'),
}

SHELL_HEAD = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · 多模态图谱</title>
<link rel="stylesheet" href="{prefix}css/site.css">
<script>window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']],processEscapes:true}},svg:{{fontCache:'global'}},options:{{ignoreHtmlClass:'.*',processHtmlClass:'arithmatex',skipHtmlTags:['script','noscript','style','textarea','pre','code']}}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script>
</head>
<body class="report-body" id="top">
<nav class="topnav"><a class="brand" href="{prefix}index.html">◈ 多模态图谱</a><div class="topnav-links"><a href="{prefix}index.html#reports">报告</a><a href="{prefix}paper/index.html">研究方案</a><a href="https://github.com/yuchenlogin/multimodal-atlas" target="_blank" rel="noopener">GitHub ↗</a><button id="theme-toggle" aria-label="切换主题">◐</button></div></nav>
<div class="layout">
<aside class="sidenav" id="sidenav">
<div class="sidenav-title">系列报告</div>
{sidenav}
<div class="sidenav-title" style="margin-top:1.2rem">更多</div>
<a class="sidenav-link" href="{prefix}paper/index.html">研究方案 · 尚待实验验证</a>
<a class="sidenav-link" href="{prefix}index.html#about">ℹ️ 关于本站</a>
</aside>
<main class="content">
<header class="report-header"><div class="report-kicker">{kicker}</div><h1>{h1}</h1><div class="report-sub">{subtitle}</div>
<div class="report-actions"><a class="btn" href="{prefix}assets/{slug}.pdf" target="_blank" rel="noopener">⬇ 下载 PDF 版</a><button class="btn ghost" id="reading-progress-btn" title="阅读进度">0%</button></div>
</header>
<article class="report-content" id="article">
"""

SHELL_FOOT = """
</article>
<footer class="footer"><span>多模态图谱 · 技术报告修订 2026-10-16</span><a href="#top">回到顶部 ↑</a></footer>
</main>
</div>
<div class="reading-bar" id="reading-bar"></div>
<script src="{prefix}js/site.js"></script>
</body>
</html>
"""

def sidenav_html(active):
    links = []
    for slug in sorted(META):
        title, kicker, desc, icon = META[slug]
        cls = "sidenav-link active" if slug == active else "sidenav-link"
        links.append(f'<a class="{cls}" href="../{slug}/index.html"><span class="sidenav-icon">{icon}</span><span>{slug[:2]} · {kicker}</span></a>')
    return "\n".join(links)

MD_EXTS = ["tables", "fenced_code", "toc", "pymdownx.tilde", "attr_list", "md_in_html", "pymdownx.arithmatex"]


class FigureCaptions(Treeprocessor):
    """Turn standalone figures into visible, accessible captions after inline parsing."""
    def run(self, root):
        for paragraph in list(root.iter("p")):
            if len(paragraph) == 1 and paragraph[0].tag == "img" and not (paragraph.text or "").strip():
                img = paragraph[0]
                if (img.tail or "").strip():
                    continue
                paragraph.tag = "figure"
                img.set("loading", "lazy")
                caption = ET.SubElement(paragraph, "figcaption")
                caption.text = img.get("alt", "")


class FigureExtension(Extension):
    def extendMarkdown(self, md):
        md.treeprocessors.register(FigureCaptions(md), "figure_captions", 5)


def render_markdown(text):
    """Protect TeX before Markdown parses underscores, braces and backslashes."""
    return markdown.markdown(text, extensions=MD_EXTS + [FigureExtension()], extension_configs={
        "toc": {"toc_depth": "2-3"},
        "pymdownx.arithmatex": {"generic": True},
    })

def convert(slug):
    title, kicker, desc, icon = META[slug]
    md_path = os.path.join(SRC, slug, "report.md")
    text = Path(md_path).read_text(encoding="utf-8")
    # The source heading is authoritative; the navigation label stays short.
    heading = re.match(r"^# (.+)", text)
    if heading:
        title = heading.group(1)
    # strip leading H1 (we render our own header)
    text = re.sub(r"^# .*\n", "", text, count=1)
    # demote heading levels by one (## -> # handled by CSS anyway) — keep as-is
    # Rewrite source image paths for the GitHub Pages project subpath.
    text = text.replace("../../figures/", "/multimodal-atlas/assets/")
    text = text.replace("../../assets/", "/multimodal-atlas/assets/")
    # render
    html_body = render_markdown("[TOC]\n\n" + text)
    # heading ids for TOC anchors: python-markdown toc extension adds ids
    out_dir = os.path.join(ROOT, "reports", slug)
    os.makedirs(out_dir, exist_ok=True)
    page = SHELL_HEAD.format(
        title=escape(title), prefix="../../", kicker=f"{icon} 专题 {slug[:2]} · {kicker}",
        h1=escape(title), subtitle=escape(desc), slug=slug, sidenav=sidenav_html(slug)) + html_body + SHELL_FOOT.format(prefix="../../")
    Path(out_dir, "index.html").write_text(page, encoding="utf-8")
    return len(text)

def convert_proposal():
    source = Path(ROOT, "paper", "proposal.md").read_text(encoding="utf-8")
    title = source.splitlines()[0][2:]
    body = render_markdown("[TOC]\n\n" + re.sub(r"^# .*\n", "", source, count=1))
    nav = sidenav_html(None).replace('href="../', 'href="../reports/')
    page = SHELL_HEAD.format(title=escape(title), prefix="../", kicker="研究方案 · 尚待实验验证",
        h1=escape(title), subtitle="算法、对照与证伪标准；旧版未经实验支持的数字已撤回。",
        slug="paper-verifier-gated-visual-decoding", sidenav=nav) + body + SHELL_FOOT.format(prefix="../")
    Path(ROOT, "paper", "index.html").write_text(page, encoding="utf-8")


def convert_home():
    cards = []
    for slug, (_, kicker, desc, icon) in META.items():
        heading = Path(SRC, slug, "report.md").read_text(encoding="utf-8").splitlines()[0][2:]
        cards.append(f'<a class="card" href="reports/{slug}/index.html"><div class="card-top"><span class="card-icon">{icon}</span><span class="card-num">{slug[:2]} / {kicker}</span></div><h3>{escape(heading)}</h3><p>{escape(desc)}</p><div class="card-foot">阅读全文与 PDF →</div></a>')
    template = Path(ROOT, "home.template.html").read_text(encoding="utf-8")
    Path(ROOT, "index.html").write_text(template.replace("<!-- REPORT_CARDS -->", "\n".join(cards)).replace("<!-- FIGURE_COUNT -->", str(len(list(Path(ROOT, "assets", "technical").glob("*.png"))))), encoding="utf-8")

if __name__ == "__main__":
    for slug in sorted(META):
        n = convert(slug)
        print(f"built {slug} ({n} chars)")
    convert_proposal()
    convert_home()
    print("built proposal and home")

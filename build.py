#!/usr/bin/env python3
"""Build multimodal-atlas site: convert reports/*.md -> HTML pages with shared shell."""
import re, os, sys, json, shutil
import markdown

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "..", "reports")

META = {
 "00-foundations": ("从 Transformer 到多模态基座", "基础架构", "Attention、RoPE、MoE、Scaling Laws、RLHF——整个系列的地基", "🏛️"),
 "01-vision-language": ("视觉-语言模型：从 CLIP 到原生多模态", "视觉-语言", "CLIP → BLIP-2 → LLaVA → Qwen2.5-VL 的完整谱系", "👁️"),
 "02-audio-speech": ("听见与开口：音频语音多模态大模型", "音频语音", "Whisper、RVQ codec、VALL-E、全双工语音", "🎧"),
 "03-video": ("时间的维度：视频理解与大模型", "视频理解", "Video-LLaVA、长视频、流式视频、Video-MME", "🎬"),
 "04-omni": ("任意到任意：全模态（Omni）大模型", "全模态", "Chameleon、Thinker-Talker、端到端全双工", "🌐"),
 "05-generation": ("生成的艺术：扩散、自回归与统一理解-生成", "多模态生成", "DDPM 完整推导、DiT、Flow Matching、统一模型", "🎨"),
 "06-embodied": ("从像素到动作：具身智能与 VLA 模型", "具身智能", "RT-2、OpenVLA、π0、世界模型", "🤖"),
 "07-agent-tooluse": ("会用工具的眼睛：多模态智能体", "多模态 Agent", "ReAct、OSWorld、computer-use、MCP", "🧭"),
 "08-efficiency": ("让大象起舞：高效化与端侧部署", "高效化", "量化、投机解码、视觉 token 压缩、vLLM", "⚡"),
 "09-alignment-safety": ("对齐、幻觉与安全", "对齐与安全", "POPE、VCD/OPERA、jailbreak、red-teaming", "🛡️"),
 "10-benchmarks-eval": ("度量智能：评测基准与评估方法", "评测方法", "MME、MMMU、Video-MME、leaderboard 批判", "📏"),
 "11-data-training": ("数据的炼金术：数据工程与训练方法", "数据与训练", "DataComp、合成数据、训练配方", "🧪"),
 "12-frontier": ("潮水的方向：2025–2026 前沿专题", "前沿专题", "统一架构、多模态推理、世界模型、炒作 vs 实质", "🔭"),
}

SHELL_HEAD = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · 多模态图谱</title>
<link rel="stylesheet" href="{prefix}css/site.css">
<script>window.MathJax={{tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']],processEscapes:true}},svg:{{fontCache:'global'}},options:{{skipHtmlTags:['script','noscript','style','textarea','pre']}}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body class="report-body">
<nav class="topnav"><a class="brand" href="{prefix}index.html">◈ 多模态图谱</a><div class="topnav-links"><a href="{prefix}index.html#reports">报告</a><a href="{prefix}paper/index.html">论文</a><a href="https://github.com/yuchenlogin/multimodal-atlas" target="_blank" rel="noopener">GitHub ↗</a><button id="theme-toggle" aria-label="切换主题">◐</button></div></nav>
<div class="layout">
<aside class="sidenav" id="sidenav">
<div class="sidenav-title">系列报告</div>
{sidenav}
<div class="sidenav-title" style="margin-top:1.2rem">更多</div>
<a class="sidenav-link" href="{prefix}paper/index.html">📄 顶会风格论文</a>
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
<footer class="footer"><span>多模态图谱 · 由 Codex + sub-agent swarm 于 2026-09 调研撰写</span><a href="#top">回到顶部 ↑</a></footer>
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
        links.append(f'<a class="{cls}" href="../{slug}/index.html"><span class="sidenav-icon">{icon}</span><span>{slug[3:]} · {kicker}</span></a>')
    return "\n".join(links)

MD_EXTS = ["tables", "fenced_code", "toc", "pymdownx.tilde", "attr_list", "md_in_html"]

def convert(slug):
    title, kicker, desc, icon = META[slug]
    md_path = os.path.join(SRC, slug, "report.md")
    text = open(md_path, encoding="utf-8").read()
    # strip leading H1 (we render our own header)
    text = re.sub(r"^# .*\n", "", text, count=1)
    # demote heading levels by one (## -> # handled by CSS anyway) — keep as-is
    # rewrite image paths ../../figures/x.png -> ../assets/x.png
    text = text.replace("../../figures/", "../assets/")
    # render
    html_body = markdown.markdown(text, extensions=MD_EXTS, extension_configs={"toc": {"toc_depth": "2-3"}})
    # heading ids for TOC anchors: python-markdown toc extension adds ids
    out_dir = os.path.join(ROOT, "reports", slug)
    os.makedirs(out_dir, exist_ok=True)
    page = SHELL_HEAD.format(
        title=title, prefix="../../", kicker=f"{icon} 专题 {slug[:2]} · {kicker}",
        h1=title, subtitle=desc, slug=slug, sidenav=sidenav_html(slug)) + html_body + SHELL_FOOT.format(prefix="../../")
    open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8").write(page)
    return len(text)

if __name__ == "__main__":
    for slug in sorted(META):
        n = convert(slug)
        print(f"built {slug} ({n} chars)")
    print("done")

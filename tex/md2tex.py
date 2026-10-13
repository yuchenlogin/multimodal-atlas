#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
md2tex.py — 把本系列中文 Markdown 报告转换为 LaTeX (ctexart) 源文件。

用法:
    python3 md2tex.py <report.md> [-o output.tex]

特性:
  - 一级标题 # → \\title; 标题下 "> 副标题：..." / "> 版本日期：..." → \\subtitle / \\date
    (两行 blockquote 用 "\\\\" 断行, 不再合并)
  - ## → \\section, ### → \\subsection, #### → \\subsubsection (剥离手工编号, LaTeX 自动编号)
  - ## 摘要 → abstract 环境 (shaded 灰底)
  - 文末 [n] 连续行 → description 列表排版, URL → \\url{}
  - Markdown 表格 → booktabs longtable (L 列 = 自适应宽度 raggedright)
  - $$...$$ 独立成段 → \\[ ... \\], 内容原样保留
  - 行内 $...$ 原样保留, 公式内容绝不转义
  - ![alt](path) → figure + \\includegraphics, 图片路径相对于 Markdown 源稿解析
  - ``` 代码块 → ttfamily 阴影小框 (framed 包, 冷启动友好)
  - **x** → \\textbf, *x* → \\emph, `x` → \\texttt
  - LaTeX 特殊字符转义 (\\ & % $ # _ { } ~ ^), 公式 / URL 内容受保护
"""

import re
import os
import argparse
from pathlib import Path

PREAMBLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "report_preamble.tex")
DEFAULT_AUTHOR = "多模态大模型系列报告"

SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
ESC_RE = re.compile(r"([\\&%$#_{}~^])")
URL_RE = re.compile(r"https?://[^\s|)\]}>，。；：、（）]+")

BLOCK_START_RE = re.compile(
    r"^(#{1,4}\s|>|\$\$|```|\||!\[|[-*+]\s|\d+[.)]\s|\[\d+\]\s|-{3,}\s*$)")


# 正文拉丁字体 (lmroman) 缺字形、需回退到中文字体的字符
CJK_FALLBACK = {
    "★": r"\CJKfamily+{zhsong}★",
    "☆": r"\CJKfamily+{zhsong}☆",
    "●": r"\CJKfamily+{zhsong}●",
    "○": r"\CJKfamily+{zhsong}○",
    "■": r"\CJKfamily+{zhsong}■",
    "□": r"\CJKfamily+{zhsong}□",
    "▲": r"\CJKfamily+{zhsong}▲",
    "✓": r"\CJKfamily+{zhsong}✓",
    "✗": r"\CJKfamily+{zhsong}✗",
}

# Latin Modern's text font lacks these mathematical Unicode characters.
TEXT_MATH = {
    "π": r"\ensuremath{\pi}",
    "↔": r"\ensuremath{\leftrightarrow}",
    "≠": r"\ensuremath{\ne}",
    "≈": r"\ensuremath{\approx}",
}


def escape_text(s):
    s = ESC_RE.sub(lambda m: SPECIALS[m.group(1)], s)
    for ch, rep in CJK_FALLBACK.items():
        s = s.replace(ch, rep)
    for ch, rep in TEXT_MATH.items():
        s = s.replace(ch, rep)
    return s


def escape_code_line(s):
    """代码块逐字符转义; 拉丁字体缺字形的中文/符号回退到宋体族。"""
    out = []
    for ch in s:
        if ch in SPECIALS:
            out.append(SPECIALS[ch])
        elif ch in TEXT_MATH:
            out.append(TEXT_MATH[ch])
        elif ord(ch) > 0x2E7F or ch in CJK_FALLBACK:
            out.append(r"\CJKfamily+{zhsong}%s" % ch)
        else:
            out.append(ch)
    return "".join(out)


def protect_and_format(text):
    """普通文本 → LaTeX: 公式/URL 保护 → 转义 → 行内格式。"""
    stash_tokens = []

    def stash(latex):
        stash_tokens.append(latex)
        return "\x00%d\x00" % (len(stash_tokens) - 1)

    # Protect code before scanning dollars; Chinese code needs a CJK font.
    text = re.sub(r"`([^`\n]+?)`",
                  lambda m: stash(r"\texttt{%s}" % escape_code_line(m.group(1))), text)
    # 1. URL → \url{} (内容不转义)
    text = URL_RE.sub(lambda m: stash(r"\url{%s}" % m.group(0)), text)
    # 2. 行内公式 $...$ 原样保留
    text = re.sub(r"\$([^$\n]+?)\$",
                  lambda m: stash("$%s$" % m.group(1)), text)
    # 3. 转义
    text = escape_text(text)
    # 4. 行内格式
    text = re.sub(r"\*\*([^*]+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"__([^_]+?)__", r"\\textbf{\1}", text)
    text = re.sub(r"(?<![*\w])\*([^*\n]+?)\*(?!\*)", r"\\emph{\1}", text)
    text = re.sub(r"~~(.+?)~~", r"\\sout{\1}", text)
    # 5. 还原
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\x00(\d+)\x00",
                      lambda m: stash_tokens[int(m.group(1))], text)
    return text


def strip_heading_number(title):
    return re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", title).strip()


class Converter:
    def __init__(self, md_path):
        self.md_path = md_path
        self.lines = Path(md_path).read_text(encoding="utf-8").splitlines()
        self.out = []
        self.i = 0
        self.title = None
        self.subtitle = None
        self.date = None
        self.warnings = []

    def warn(self, msg):
        self.warnings.append("line %d: %s" % (self.i + 1, msg))

    def convert(self):
        while self.i < len(self.lines):
            line = self.lines[self.i]
            if self.try_heading(line):
                continue
            if self.try_blockquote(line):
                continue
            if self.try_display_math(line):
                continue
            if self.try_code_fence(line):
                continue
            if self.try_table(line):
                continue
            if self.try_image(line):
                continue
            if self.try_hrule(line):
                continue
            if self.try_list(line):
                continue
            if self.try_references(line):
                continue
            self.paragraph(line)
        return "\n".join(self.out)

    # ----- 标题 -----
    def try_heading(self, line):
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if not m:
            return False
        level = len(m.group(1))
        raw = m.group(2).strip()
        if level == 1:
            self.title = raw
            j = self.i + 1
            # 跳过标题与元信息之间的空行
            while j < len(self.lines) and not self.lines[j].strip():
                j += 1
            meta = []
            while j < len(self.lines) and self.lines[j].startswith(">"):
                meta.append(self.lines[j].lstrip("> ").strip())
                j += 1
            consumed_meta = False
            for item in meta:
                label = re.split(r"[:：]", item, 1)
                if label[0].strip() == "副标题" and len(label) == 2:
                    self.subtitle = label[1].strip()
                    consumed_meta = True
                elif label[0].strip() == "版本日期" and len(label) == 2:
                    d = label[1]
                    self.date = d.split("|")[0].strip()
                    consumed_meta = True
            if consumed_meta:
                self.i = j  # 元信息行全部被吃掉
                return True
            self.i += 1
            return True
        title = strip_heading_number(raw)
        self.out.append("")
        if title in ("摘要", "Abstract", "abstract"):
            self.out.append("\\begin{abstract}")
            self.i += 1
            buf = []
            while self.i < len(self.lines) and not BLOCK_START_RE.match(self.lines[self.i]):
                if self.lines[self.i].strip():
                    buf.append(self.lines[self.i].strip())
                self.i += 1
            self.out.append(protect_and_format(" ".join(buf)))
            self.out.append("\\end{abstract}")
            self.out.append("")
            return True
        cmd = {2: "section", 3: "subsection", 4: "subsubsection"}[level]
        self.out.append("\\%s{%s}" % (cmd, protect_and_format(title)))
        self.i += 1
        return True

    # ----- blockquote -----
    def try_blockquote(self, line):
        if not line.startswith(">"):
            return False
        buf = []
        while self.i < len(self.lines) and self.lines[self.i].startswith(">"):
            buf.append(self.lines[self.i].lstrip("> ").strip())
            self.i += 1
        self.out.append("")
        self.out.append("\\begin{quote}\\small\\kaishu")
        self.out.append((r"\\" + "\n").join(protect_and_format(b) for b in buf))
        self.out.append("\\end{quote}")
        return True

    # ----- 行间公式 -----
    def try_display_math(self, line):
        stripped = line.strip()
        if not stripped.startswith("$$"):
            return False
        # Both $$equation$$ and delimiter-only lines occur in the reports.
        if len(stripped) > 4 and stripped.endswith("$$"):
            self.out.extend(["", "\\[", stripped[2:-2].strip(), "\\]", ""])
            self.i += 1
            return True
        if stripped != "$$":
            self.warn("display math opening delimiter must be on its own line")
            return False
        self.i += 1
        buf = []
        while self.i < len(self.lines) and self.lines[self.i].strip() != "$$":
            buf.append(self.lines[self.i])
            self.i += 1
        if self.i >= len(self.lines):
            self.warn("unterminated $$ block")
        else:
            self.i += 1
        self.out.append("")
        self.out.append("\\[")
        self.out.extend(buf)
        self.out.append("\\]")
        self.out.append("")
        return True

    # ----- 代码块 -----
    def try_code_fence(self, line):
        if not line.strip().startswith("```"):
            return False
        self.i += 1
        buf = []
        while self.i < len(self.lines) and not self.lines[self.i].strip().startswith("```"):
            buf.append(self.lines[self.i])
            self.i += 1
        if self.i < len(self.lines):
            self.i += 1
        else:
            self.warn("unterminated code fence")
        self.out.append("")
        self.out.append("\\begin{shaded}\\footnotesize")
        for ln in buf:
            stripped = ln.lstrip()
            indent = "~" * (len(ln) - len(stripped))
            self.out.append(indent + escape_code_line(stripped) + r"\par")
        self.out.append("\\end{shaded}")
        self.out.append("")
        return True

    # ----- 表格 -----
    def try_table(self, line):
        if not line.strip().startswith("|"):
            return False
        rows = []
        while self.i < len(self.lines) and self.lines[self.i].strip().startswith("|"):
            rows.append(self.lines[self.i].strip())
            self.i += 1
        if len(rows) < 2 or not re.match(r"^\|[\s:|-]+\|?$", rows[1]):
            self.warn("table without separator row")
            for r in rows:
                self.out.append(protect_and_format(r))
            return True
        header = self._split_row(rows[0])
        body = [self._split_row(r) for r in rows[2:]]
        ncol = len(header)
        # 按表头视觉宽度加权分配列宽 (中文按 2 计), 防宽表溢出
        def vw(s):
            return max(sum(2 if ord(c) > 127 else 1 for c in s), 3)
        weights = [vw(c) for c in header]
        total = sum(weights)
        spec = "".join(
            "Y{\\dimexpr(\\linewidth-2\\tabcolsep*%d-1.2em)*%d/%d\\relax}"
            % (ncol, w, total) for w in weights)
        head = " & ".join("\\textbf{%s}" % protect_and_format(c) for c in header) + " \\\\"
        self.out.append("")
        self.out.append("{\\small\\setlength{\\tabcolsep}{4pt}%")
        self.out.append("\\begin{longtable}{%s}" % spec)
        self.out.append("\\toprule")
        self.out.append(head)
        self.out.append("\\midrule")
        self.out.append("\\endfirsthead")
        self.out.append("\\toprule")
        self.out.append(head)
        self.out.append("\\midrule")
        self.out.append("\\endhead")
        self.out.append("\\bottomrule")
        self.out.append("\\endfoot")
        for r in body:
            cells = [protect_and_format(c) for c in r]
            cells += [""] * (ncol - len(cells))
            self.out.append(" & ".join(cells[:ncol]) + " \\\\")
        self.out.append("\\end{longtable}%")
        self.out.append("}")
        self.out.append("")
        return True

    @staticmethod
    def _split_row(row):
        row = row.strip()
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|"):
            row = row[:-1]
        return [c.strip() for c in row.split("|")]

    # ----- 图片 -----
    def try_image(self, line):
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", line.strip())
        if not m:
            return False
        alt, path = m.group(1), m.group(2)
        if not os.path.isabs(path):
            path = str((Path(self.md_path).resolve().parent / path).resolve())
        self.out.append("")
        self.out.append("\\begin{figure}[htbp]")
        self.out.append("  \\centering")
        self.out.append("  \\includegraphics[width=0.92\\textwidth]{%s}" % path)
        if alt:
            self.out.append("  \\caption{%s}" % protect_and_format(alt))
        self.out.append("\\end{figure}")
        self.out.append("")
        self.i += 1
        return True

    # ----- 水平线 -----
    def try_hrule(self, line):
        if re.match(r"^\s*(-{3,}|\*{3,})\s*$", line):
            self.out.append("\\bigskip")
            self.i += 1
            return True
        return False

    # ----- 列表 -----
    def try_list(self, line):
        m = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$", line)
        if not m:
            return False
        ordered = bool(re.match(r"\d", m.group(2)))
        env = "enumerate" if ordered else "itemize"
        self.out.append("\\begin{%s}" % env)
        while self.i < len(self.lines):
            m = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$", self.lines[self.i])
            if not m:
                break
            if bool(re.match(r"\d", m.group(2))) != ordered:
                self.warn("mixed list markers")
            self.out.append("  \\item " + protect_and_format(m.group(3).strip()))
            self.i += 1
        self.out.append("\\end{%s}" % env)
        self.out.append("")
        return True

    # ----- 参考文献 -----
    def try_references(self, line):
        if not re.match(r"^\[\d+\]\s", line.strip()):
            return False
        items = []
        while self.i < len(self.lines):
            m = re.match(r"^\s*\[(\d+)\]\s+(.*)$", self.lines[self.i])
            if not m:
                break
            items.append((m.group(1), m.group(2)))
            self.i += 1
        if len(items) < 2:
            for n, t in items:
                self.out.append(protect_and_format("[%s] %s" % (n, t)))
            return True
        self.out.append("")
        self.out.append("\\begin{description}[style=nextline, itemsep=2pt,"
                        " font=\\normalfont\\bfseries\\color{accent}, leftmargin=2.6em]")
        for n, text in items:
            self.out.append("  \\item[[%s]] \\small %s" % (n, protect_and_format(text)))
        self.out.append("\\end{description}")
        self.out.append("")
        return True

    # ----- 段落 -----
    def paragraph(self, line):
        if not line.strip():
            self.out.append("")
            self.i += 1
            return
        buf = [line.strip()]
        self.i += 1
        while self.i < len(self.lines):
            nxt = self.lines[self.i]
            if not nxt.strip() or BLOCK_START_RE.match(nxt):
                break
            buf.append(nxt.strip())
            self.i += 1
        self.out.append(protect_and_format(" ".join(buf)))
        self.out.append("")


def build_document(conv, body):
    title = protect_and_format(conv.title or "未命名报告")
    lines = [
        "% !TEX program = xelatex",
        "% Auto-generated by md2tex.py — 请勿直接手改, 改 Markdown 后重新生成",
        "\\input{%s}" % PREAMBLE_PATH,
        "",
        "\\title{%s}" % title,
    ]
    if conv.subtitle:
        lines.append("\\subtitle{%s}" % protect_and_format(conv.subtitle))
    lines += [
        "\\author{%s}" % DEFAULT_AUTHOR,
        "\\date{%s}" % (protect_and_format(conv.date) if conv.date else r"\today"),
        "",
        "\\begin{document}",
        "\\maketitle",
        "\\tableofcontents",
        "\\newpage",
        "",
        body,
        "",
        "\\end{document}",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Convert series report Markdown to LaTeX.")
    ap.add_argument("md")
    ap.add_argument("-o", "--output")
    args = ap.parse_args()

    conv = Converter(args.md)
    body = conv.convert()
    doc = build_document(conv, body)

    out_path = args.output
    if not out_path:
        base = os.path.basename(os.path.dirname(os.path.abspath(args.md)))
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), base + ".tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)

    print("written:", out_path)
    for w in conv.warnings:
        print("  warning:", w)


if __name__ == "__main__":
    main()

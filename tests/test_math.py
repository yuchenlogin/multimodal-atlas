"""Regressions for the two broken rendering paths and all report formulas."""
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build import META, render_markdown
from tex.md2tex import Converter, protect_and_format


class MathHTML(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.formulas = []
        self.tags_inside_math = []
        self.current = None
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if self.current is not None:
            self.tags_inside_math.append(tag)
        if dict(attrs).get("class") == "arithmatex":
            self.current = ""

    def handle_data(self, data):
        if self.current is not None:
            self.current += data

    def handle_endtag(self, tag):
        if self.current is not None and tag in ("div", "span"):
            self.formulas.append(self.current[2:-2].strip())
            self.current = None


def source_formulas(text):
    # Exclude code; compare the remaining TeX independently of the Markdown parser.
    text = re.sub(r"(?ms)^```.*?^```[^\n]*", "", text)
    text = re.sub(r"`[^`\n]+`", "", text)
    pattern = r"(?<!\\)\$\$(.*?)\$\$|(?<![\\$])\$(?![\s$])([^$\n]*?\S)\$(?!\$)"
    return [(m.group(1) if m.group(1) is not None else m.group(2)).strip()
            for m in re.finditer(pattern, text, re.S)]


class MathRenderingTests(unittest.TestCase):
    def test_markdown_keeps_tex_atomic(self):
        tex = r"\mathcal{L}_{DPO} = \mathbb{E}_{(x,y_w,y_l)}\frac{\pi_\phi(y_w|x)}{\pi_{ref}(y_w|x)}"
        for source in (f"$${tex}$$", f"$$\n{tex}\n$$", f"行内 ${tex}$。"):
            with self.subTest(source=source):
                parsed = MathHTML(render_markdown(source))
                self.assertEqual(parsed.formulas, [tex])
                self.assertEqual(parsed.tags_inside_math, [])

    def test_code_currency_and_regular_markdown(self):
        html = render_markdown("**正文** ~$32k，`$x_i$`\n\n```tex\n$$x_i$$\n```")
        self.assertEqual(MathHTML(html).formulas, [])
        self.assertIn("<strong>正文</strong>", html)
        self.assertIn("<code>$x_i$</code>", html)

    def test_pdf_link_label_can_include_inline_code(self):
        output = protect_and_format("[`run()`](https://example.com)")
        self.assertEqual(output, r"\href{https://example.com}{\texttt{run()}}")
        literal = protect_and_format("`[label](https://example.com)`")
        self.assertNotIn(r"\href", literal)

    def test_inline_math_in_table(self):
        html = render_markdown("| 公式 |\n| --- |\n| $R_{\\Theta,n-m}$ |")
        self.assertEqual(MathHTML(html).formulas, [r"R_{\Theta,n-m}"])
        self.assertIn("<table>", html)

    def test_source_images_are_in_repository(self):
        for slug in META:
            path = ROOT / "reports" / slug / "report.md"
            for image in re.findall(r"^!\[.*\]\(([^)]+)\)\s*$", path.read_text(encoding="utf-8"), re.M):
                with self.subTest(slug=slug, image=image):
                    self.assertTrue((path.parent / image).is_file())

    def test_all_generated_formulas_match_sources(self):
        for slug in META:
            with self.subTest(slug=slug):
                source = (ROOT / "reports" / slug / "report.md").read_text(encoding="utf-8")
                parsed = MathHTML((ROOT / "reports" / slug / "index.html").read_text(encoding="utf-8"))
                self.assertEqual(parsed.tags_inside_math, [])
                self.assertEqual(parsed.formulas, source_formulas(source))

    def test_tex_accepts_both_display_styles(self):
        tex = r"\mathcal{L}_{DPO} = \mathbb{E}_{(x,y_w,y_l)}"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.md"
            path.write_text(f"$${tex}$$\n\n$$\n{tex}\n$$\n\n`$x_i$`", encoding="utf-8")
            converter = Converter(path)
            output = converter.convert()
            self.assertEqual(output.count("\\["), 2)
            self.assertEqual(output.count(tex), 2)
            self.assertNotIn(r"\$$", output)
            self.assertEqual(converter.warnings, [])

    def test_tex_text_symbols_and_code(self):
        self.assertEqual(protect_and_format("π ≈ ≠ ↔"),
                         r"\ensuremath{\pi} \ensuremath{\approx} \ensuremath{\ne} \ensuremath{\leftrightarrow}")
        output = protect_and_format("`$x_i$（模型）`")
        self.assertIn(r"\$x\_i\$", output)
        self.assertIn(r"\CJKfamily+{zhsong}模", output)
        self.assertEqual(protect_and_format("https://arxiv.org/abs/2410.21276（模型发布为）"),
                         r"\url{https://arxiv.org/abs/2410.21276}（模型发布为）")

    def test_image_after_abstract_is_not_plain_text(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.md"
            path.write_text("## 摘要\n\n摘要正文。\n\n![图](figure.png)\n\n## 引言\n", encoding="utf-8")
            output = Converter(path).convert()
            self.assertIn(r"\end{abstract}", output)
            self.assertIn(r"\begin{figure}", output)
            self.assertIn(str(Path(temp) / "figure.png"), output)
            self.assertNotIn("![图]", output)

    def test_figure_caption_can_contain_interval_brackets(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.md"
            path.write_text("![图 1：95% interval [0,10.5] pp](figure.png)\n", encoding="utf-8")
            output = Converter(path).convert()
            self.assertIn(r"\includegraphics", output)
            self.assertIn("interval [0,10.5]", output)
            self.assertNotIn("![", output)

    def test_all_tex_display_counts_match_sources(self):
        for slug in META:
            with self.subTest(slug=slug):
                path = ROOT / "reports" / slug / "report.md"
                converter = Converter(path)
                output = converter.convert()
                expected = len(re.findall(r"(?<!\\)\$\$(.*?)\$\$", path.read_text(encoding="utf-8"), re.S))
                self.assertEqual(output.count("\\["), expected)
                self.assertNotIn(r"\$$", output)
                self.assertEqual(converter.warnings, [])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Rebuild report PDFs with Tectonic; publish each PDF only after compilation."""
from pathlib import Path
import shutil
import subprocess
import tempfile

from build import META, ROOT
from tex.md2tex import Converter, build_document


def main():
    root = Path(ROOT)
    if not shutil.which("tectonic"):
        raise SystemExit("Tectonic is required to build PDFs: https://tectonic-typesetting.github.io/")
    for slug in sorted(META):
        converter = Converter(root / "reports" / slug / "report.md")
        document = build_document(converter, converter.convert())
        if converter.warnings:
            raise RuntimeError(f"{slug}: {converter.warnings}")
        source = root / "tex" / f"{slug}.tex"
        source.write_text(document, encoding="utf-8")
        # Keep the completed file on the destination filesystem for atomic replacement.
        with tempfile.TemporaryDirectory(prefix=".pdf-build-", dir=root / "assets") as output:
            result = subprocess.run(
                ["tectonic", "--keep-logs", str(source), "--outdir", output],
                capture_output=True, text=True,
            )
            log = Path(output) / f"{slug}.log"
            if log.exists():
                shutil.copy2(log, root / "tex" / log.name)
            if result.returncode:
                raise RuntimeError(f"{slug}:\n{result.stdout}\n{result.stderr}")
            (Path(output) / f"{slug}.pdf").replace(root / "assets" / f"{slug}.pdf")
        print(f"built PDF {slug}", flush=True)


if __name__ == "__main__":
    main()

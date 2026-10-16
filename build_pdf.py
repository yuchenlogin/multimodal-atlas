#!/usr/bin/env python3
"""Rebuild report/proposal PDFs; publish each only after successful compilation."""
import argparse
import fcntl
from pathlib import Path
import shutil
import subprocess
import tempfile

from build import META, ROOT
from tex.md2tex import Converter, build_document

PROPOSAL = 'paper-verifier-gated-visual-decoding'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('topics', nargs='*', help='Report slug(s), or the proposal PDF slug; default: all 14')
    args = parser.parse_args()
    root = Path(ROOT)
    sources = {slug: root / 'reports' / slug / 'report.md' for slug in sorted(META)}
    sources[PROPOSAL] = root / 'paper' / 'proposal.md'
    unknown = set(args.topics) - sources.keys()
    if unknown:
        parser.error(f'Unknown topics: {sorted(unknown)}')
    if not shutil.which('tectonic'):
        raise SystemExit('Tectonic is required: https://tectonic-typesetting.github.io/')
    # Hold a process lock through compilation and publication, including selective builds.
    # flock is released by the OS even if a build is interrupted or crashes.
    with (root / 'tex' / '.pdf-build.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another PDF build is active; retry after it finishes.')
        for slug in args.topics or sources:
            converter = Converter(sources[slug])
            document = build_document(converter, converter.convert())
            if converter.warnings:
                raise RuntimeError(f'{slug}: {converter.warnings}')
            source = root / 'tex' / f'{slug}.tex'
            source.write_text(document, encoding='utf-8')
            with tempfile.TemporaryDirectory(prefix='.pdf-build-', dir=root / 'assets') as output:
                result = subprocess.run(
                    ['tectonic', '--keep-logs', str(source), '--outdir', output],
                    capture_output=True, text=True,
                )
                log = Path(output) / f'{slug}.log'
                if log.exists():
                    shutil.copy2(log, root / 'tex' / log.name)
                if result.returncode:
                    raise RuntimeError(f'{slug}:\n{result.stdout}\n{result.stderr}')
                if log.exists() and 'Missing character:' in log.read_text(errors='replace'):
                    raise RuntimeError(f'{slug}: missing font glyphs; inspect {root / "tex" / log.name}')
                (Path(output) / f'{slug}.pdf').replace(root / 'assets' / f'{slug}.pdf')
            print(f'built PDF {slug}', flush=True)


if __name__ == '__main__':
    main()

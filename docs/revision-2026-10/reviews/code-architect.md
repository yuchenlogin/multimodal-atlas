1. **Medium — PDF inline formatting leaks placeholder state across recursive calls.** [tex/md2tex.py:123](../../../tex/md2tex.py#L123) protects inline code in a local token stash, then line 127 recursively formats link labels with a new, empty stash. This valid Markdown reproducibly raises `IndexError: list index out of range`:

   ```markdown
   [`run_probes.py`](https://example.org/run_probes.py)
   ```

   Adding such a link to a report makes `build_pdf.py` abort, while HTML renders successfully. This is a confirmed latent bug; the current manuscripts do not trigger it.

   **Mapping:** boundary-discipline, foundational-thinking, redesign-from-first-principles. Keep protected tokens within their owning formatting context, or parse links and their children structurally. Add a regression covering code inside link labels.

Verification: the 13 original probes, planning output, eight mechanism checks, and 19 raw CSVs matched in-memory recomputation. All 14 generated TeX files matched current sources; PDFs contained the expected figure counts. Seven read-only regression tests passed. Concurrent fixes resolved the caption-count mismatch observed earlier, so it is excluded from findings.

Limitations: `brain/principles.md` was unavailable; mappings use only your supplied lens. No files were changed or agents spawned. Full browser execution, PDF visual inspection, and remote deployment freshness were not independently verified.

**PASS** — no confirmed shipping blocker; one medium build-correctness defect remains.
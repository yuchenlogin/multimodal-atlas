Reviewed tracked changes against `fbb3820` and untracked sources for code/build correctness. `brain/principles.md` was unavailable; mappings below use only the supplied principle names and Skeptic lens. No files were modified or agents spawned.

1. **Medium — Valid Markdown links can crash PDF conversion.**  
   [tex/md2tex.py:123](../../../tex/md2tex.py#L123) protects inline code before recursively formatting link labels. The recursive call receives placeholders belonging to the outer call but has an empty token stash. Confirmed read-only reproduction: calling `protect_and_format` on ``[`run()`](https://example.com)`` raises `IndexError: list index out of range`. Any report introducing this valid syntax stops the PDF build. Current reports do not trigger it.  
   **Fix:** preserve token ownership across recursive formatting and add a regression covering code inside link labels.  
   **Mapping:** `fix-root-causes`.

2. **Medium — Concurrent PDF builds share mutable compilation inputs.**  
   [build_pdf.py:32](../../../build_pdf.py#L32) writes every invocation’s input to `tex/<slug>.tex`; only output directories are unique. An overlapping full build and selective rebuild can overwrite that input before Tectonic reads it. An older invocation can also finish last and replace a newer PDF at line 46. Atomic replacement prevents partial files, but does not guarantee freshness. This is a retained concurrency weakness, not evidence that current PDFs are corrupted.  
   **Fix:** serialize overlapping builds through publication, or use isolated input snapshots with freshness checks before replacement.  
   **Mapping:** `serialize-shared-state-mutations`.

3. **Medium — Existing browser verification does not cover the current content.**  
   [test-results/browser-math.json:33](../qa.json) records 26 formulas for generation; current HTML contains 59. Twelve report counts differ from the saved run. Additionally, [tests/browser_math.cjs:84](../../../tests/browser_math.cjs#L84) replaces the success artifact only after completion, leaving an old successful record intact after a failed rerun. This is a confirmed verification gap, not a confirmed browser-rendering defect.  
   **Fix:** rerun against the final tree and associate results with input hashes and run status.  
   **Mapping:** `prove-it-works`.

Read-only checks passed: seven non-writing regression tests, HTML regeneration comparisons, all teaching-probe recomputations, mechanism-check CSV comparisons, figure hashes, and metadata coverage. All 14 PDFs were readable; the subsequently rebuilt chapter 10 PDF contains its fourth figure. Python-version documentation and bracketed-caption parsing were corrected during review and are excluded from findings.

**PASS — no confirmed high-severity defect; three medium findings remain.**
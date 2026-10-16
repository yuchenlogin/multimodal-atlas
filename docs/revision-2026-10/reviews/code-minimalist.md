1. **Medium — Configurable experiments can silently invalidate publication captions.** [labs/check_mechanisms.py:651](../../../labs/check_mechanisms.py#L651) accepts `--bootstrap-draws` and `--seed`, then overwrites the canonical publication data. However, [scripts/make_technical_figures.py:99](../../../scripts/make_technical_figures.py#L99) always labels the cluster chart “4000 resamples per scheme”; report results are also fixed text.
   
   **Confirmed input:** I executed the calculation in memory with 100 draws and verified it produces 100 CSV records. Running that supported configuration followed by the figure generator would publish a falsely labelled chart, without failing validation. The current default artifacts are consistent.
   
   **Recommendation:** Remove the configuration flags from the publication generator unless alternate runs are needed. Otherwise, keep exploratory outputs separate and derive chart annotations from their configuration.
   
   **Principles:** subtract-before-you-add; outcome-oriented-execution.

2. **Low — The browser check repeatedly fetches identical navigation targets.** [tests/browser_math.cjs:55](../../../tests/browser_math.cjs#L55) checks every local link on every report, including the shared navigation. The current pages cause **238 GET requests for 29 unique URLs**; the proposal alone is fetched 28 times. These repeated requests provide no additional link-existence coverage.
   
   **Recommendation:** Collect resolved URLs into a shared `Set` and request each once, preserving the per-page rendering checks.
   
   **Principle:** cost-aware-delegation, applied to avoiding redundant verification work.

Read-only verification passed: all 15 HTML pages match an in-memory rebuild; all 14 generated TeX files match their sources; figure and raw-data hashes match; probe, planning, and eight mechanism-check results reproduce exactly. Seven tests that require no filesystem writes passed. All 14 PDFs are readable, and their logs contain no missing-character or overfull-box warnings. I did not rerun browser rendering or PDF compilation.

`brain/principles.md` was unavailable; mappings use only the supplied principle names and Minimalist lens. No files were mutated or agents spawned.

**PASS — no confirmed ship-blocking code/build defect; the configurable-output mismatch should be fixed.**
1. **Source hashes: closed.** Every `reports/<slug>/report.md` used to calculate expected figure counts is included in `report.inputs`.
2. **Cleanup status: closed.** `passed` is written only after `browser.close()` and `server.close()` complete. The injected close fault correctly produced a failed report and exit code 1.

No remaining defects within the requested scope. **PASS.**
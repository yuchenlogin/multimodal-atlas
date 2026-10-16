**REJECT**

1. The SHA256 manifest omits `reports/*/report.md`, even though the test reads those files to set its expected figure counts. A report source can change while the recorded inputs still appear unchanged, making an earlier pass look current.

2. `browser-math.json` is marked `passed` before `browser.close()` runs. If cleanup rejects, the command exits with an error while the artifact still says `passed`—the stale-success condition this fix is meant to prevent. Mark success only after cleanup completes.
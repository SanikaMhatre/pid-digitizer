# Three-minute demo script

Show only what was built during the hackathon. State measured numbers only after you have run them.

| Time | Screen | Say |
|---|---|---|
| 0:00-0:20 | A scanned-looking drawing (`demo/noisy_scan.png`) | "Every plant runs on drawings like this. The structure (which pump feeds which vessel) is locked in pixels. Digitizing it is slow manual work and it blocks modifications and safety reviews." |
| 0:20-1:00 | UI, upload `demo/planted_errors.png`, Run | "The whole pipeline is Muse-native: one model, Muse Spark Open. It finds symbol boxes tile by tile, then reads each one: class, tag, confidence, as schema-constrained JSON." Overlay appears. "Pipes become nets with a side of attachment, so flow direction is known." |
| 1:00-1:40 | Agent log + Review queue tabs | "Uncertain symbols go to a verification agent with tools: zoom, re-ask at higher reasoning, commit a correction with evidence, or flag for a human. Fixed budget; everything is logged. When it can't be sure, it says so instead of guessing." |
| 1:40-2:25 | Violations tab, then the natural-language rule box | "Violations come from a deterministic rule engine, not the model. Here the check valve I removed from the pump line is flagged, and the missing relief valve." Type a rule in plain English, show the compiled YAML and preview, approve, re-run. "The model writes rules; code decides." |
| 2:25-2:50 | README results table | Show **real** live numbers: symbol F1, connectivity F1, planted errors found, seconds and dollars per sheet, and the no-agent ablation. |
| 2:50-3:00 | Title card | "Open weights, so plants can run it on their own infrastructure. Built with Muse Spark Open." |

Recording checklist: live mode (not offline simulated), credits visible nowhere on screen, cache warm for the demo sheet, fallback take recorded,
browser zoom 110-125%, overlay tab opened full width.

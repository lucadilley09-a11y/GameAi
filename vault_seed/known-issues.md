---
title: "Known issues"
tags:
  - "issues"
---

- Growing small -> base is approximate (base has fewer head groups); other size pairs are exact.
- Batch-piece tuning was only tested logically (no GPU in Claude's workspace); first real run is on the user's PC.
- The planner test failed once and then passed on every rerun: possibly flaky.
- Gaming detection misses small-window games from unknown folders (those count as "using": half speed).
- Learning can't release its own model's GPU memory while paused.
- play_template.html links Google Fonts (falls back offline).
- Workbench: git commands fail if the install folder isn't a git repo (installed from zips); "tests" runs the sandbox runner on a project folder.

Related:
- [[publishing-and-fixing]]

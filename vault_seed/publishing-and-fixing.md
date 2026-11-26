---
title: "Publishing and fixing"
tags:
  - "process"
---

- Repo: github.com/lucadilley09-a11y/GameAi (main). Publishing: run the tests, `python -m gameai.update --manifest`, commit, push. The app checks GitHub every 15 seconds and installs new versions by itself (verified with a SHA-256 list); models and data are never touched.
- Failures: the autopilot writes reports/errors.jsonl (step, version, end of logs/job.log). Claude's overnight scheduled task reads it, fixes bugs only, publishes, and records reports/claude_handled.json. The autopilot retries a failed step when a newer version is installed, and also after a 3-hour break (up to 4 rounds).
- Keep fixes to bugs: anything about AI speed or capability needs the user's approval.

Related:
- [[user-rules]]
- [[autopilot]]

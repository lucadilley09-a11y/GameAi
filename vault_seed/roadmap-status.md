---
title: "Roadmap status"
tags:
  - "status"
---

As of 2026-11-25:

- Phase 0 (measuring, bake-off): built; the user's bake-off finished on 2026-11-25. The autopilot chooses and installs the winner next.
- Phase 1/2 (from-scratch recipe model, game rules, teacher, playtesters): built.
- Optimise phase: built (prompt caching, per-model tuning, imatrix Q4, length-grouped batches, Unsloth, torch.compile, polite, storage).
- Phase 4 (code skill): built; runs through the autopilot after the base model is installed.
- Phase 5 (interfaces): renderer, layout checker, GameUI, UI tasks, checker, practice, the ui skill; games have GameUI pause/win/game-over screens.
- Training speed-ups 1-3 (grow, distill, batch pieces): built, approved.
- Next: Phase 3, the brain and chatbot on the chosen base model (start when the user's usage resets).

Related:
- [[autopilot]]
- [[ideas-for-later]]

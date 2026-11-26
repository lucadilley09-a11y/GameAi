---
title: "Architecture"
tags:
  - "code"
---

Main parts (folder gameai/):

- app.py: the local web app (port 8765), jobs, autopilot loop, auto-update loop. web_template.html is its page; play_template.html is a game page (engine between the "engine" markers; headless.py runs the same engine for playtester bots).
- From-scratch recipe model: model.py (decoder transformer), train.py (Muon+AdamW, WSD schedule, grow/distill/batch tuning), grow.py, datagen.py, recipe.py, bench.py, all.py (data -> train -> test -> export).
- Base model (option B): phase0/ (bake-off, choose, runtime = llama.cpp server/quantize/imatrix, tune, speedups), lora_train.py (skills: game_design, code, code_repair, ui), adapters.py (versions, promote only if better on validation).
- Data safety: splits.py, sealed.py (encrypted sealed tests), guard.py (exact + near-duplicate leak guard).
- Code skill: checks/ (Prettier, TypeScript, ESLint, ruff, StyLua, Lua compile), repair/ (repo map, structured edits, debugger sandboxes for JS/Python/Lua, atomic changes), coding/ (tasks, verify, self-teaching, teacher, mutation testing, GRPO), planner/ (project planner, component library).
- Interfaces (Phase 5): ui/ (render.py headless Edge/Chrome, layout_check.js, tasks.py, verify.py, practice.py, showcase.py), kit/gameui.css (GameUI: clean, pixel, neon, cozy).
- Sharing the PC: polite.py (away / using / heavy / gaming), storage.py (cleaning, offloading).
- Self-repair: autopilot.py writes reports/errors.jsonl on failures; retries when a newer version arrives.
- Workbench: workbench.py + mcp_server.py (gameai_mcp.py launcher): tools for Claude; notes in memory/vault.

Related:
- [[roadmap-status]]
- [[publishing-and-fixing]]

# Release and rollback procedure

This is how a new version of the AI goes from "trained" to "in use", and how to go back. It follows the plan:
validation decides, the sealed sets only measure the locked choice, and nothing ships that needs the internet.

## Two kinds of versions

| What | Where | Kept | How to go back |
| --- | --- | --- | --- |
| The app's code | the Game AI folder (updated from GitHub) | the previous version, in `backup/previous` | `/update rollback` (automatic if a new version crashes right after starting) |
| The AI itself (game AI today; base + adapters from Phase 2) | `models/`, versions in `runs/` and `bench/` | the last 20 adapter versions, the last 3 compressed releases | `/versions`, then `/rollback <n>` |

## Publishing the app's code (developer)

1. Every file goes into git first, then the SHA-256 list is written (`python -m gameai.update --manifest`), so the list
   covers every file of the version.
2. Tests must pass: `python -m tests.test_updater` (includes "the SHA-256 list matches every file") and
   `python -m tests.test_sealed`.
3. Push, then download GitHub's own copy of that version and check it passes all three checks
   (pinned version, SHA-256 list, GitHub file hashes). Only then is it announced.

On the PC, an update is installed only if it is the exact version asked for and every file matches both lists.
It never restarts the app during a job or a request. The previous version is copied to `backup/previous` first.

## Releasing a new AI version (Phase 2 onward)

1. **Train** new adapter versions (QLoRA). Every candidate is compared on **validation** only.
2. **Choose** the configuration on validation: separate adapters (default), or merged into the full-precision base and
   quantized once (only if it measurably beats separate adapters).
3. **Lock** the choice. Only now run the **sealed** sets once (`python -m gameai.sealed run <set>`): a total per set,
   recorded with the version. It never changes the choice; a disappointing score means starting again from validation.
4. **Re-test the compressed release** (4-bit GGUF): quantizing can lose quality.
5. **End-to-end suite** (before a release only).
6. **Independence test**: `/phase0 independence` (best with the Wi-Fi off). Any failure blocks the release.
7. **Bundle**: weights or adapters, tokenizer, config, runtime, licence text and the change log, kept together.
8. Keep the last 3 compressed releases and all adapter versions; older ones are deleted (lean storage).

## Rolling back

- App code: `/update rollback`, or let the launcher do it when a new version crashes at start. The broken version is
  remembered (`skip` in `update.json`) and not installed again; the next fixed version installs normally.
- The AI: `/versions` lists every kept version with its scores; `/rollback <n>` switches back. Nothing is deleted by a
  rollback.
- Updates off completely: `/updates off` (the app never goes online for them); `/updates manual` asks first.

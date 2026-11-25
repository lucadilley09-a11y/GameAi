"""Repair tools (plan: Phase 4, session 3): what lets a small model change existing code safely.

    repomap   a Tree-sitter map of a project: files, imports, exports, classes, functions, types, signatures
    edits     structured edits: unique anchors, expected match counts, file versions, a syntax check after each edit
    debugger  runs the tests and shows why one failed: error, stack with source lines, expected vs actual, variable values
    atomic    multi-file changes staged in a copy; applied only if nothing gets worse; history for rollback

    python -m gameai.repair map my_game [--focus src/player.js]
    python -m gameai.repair find my_game spawnEnemy
    python -m gameai.repair test my_game [--inspect]
    python -m gameai.repair apply my_game edits.json
    python -m gameai.repair history my_game
    python -m gameai.repair rollback my_game
"""

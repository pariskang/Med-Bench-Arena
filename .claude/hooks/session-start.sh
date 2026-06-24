#!/bin/bash
# Med-Bench-Arena — SessionStart hook for Claude Code on the web.
# Installs deps so the CLI / tests / preflight work, then profiles the MCQ eval
# set automatically (pinned revisions + answer-parse-rate check) on every session.
set -euo pipefail

# Only run in the remote (web) environment; local sessions skip.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# 1) Install the package + all non-GPU extras (cached after the first session).
pip install -e ".[all]" >/dev/null 2>&1 || pip install -e . >/dev/null 2>&1 || true
pip install Pillow >/dev/null 2>&1 || true

# 2) Make `python tests/...` and ad-hoc imports work for the whole session.
echo 'export PYTHONPATH="."' >> "$CLAUDE_ENV_FILE"

# 3) Profile the MCQ eval set. --strict surfaces any dataset that parses < 100%
#    (a mis-mapped field_map, a broken pin, an unexpected answer encoding).
#    Non-blocking + time-bounded: a transient network issue never stops the
#    session from starting. The first session downloads + caches the data;
#    later sessions reuse the cache and run fast.
echo "[session-start] profiling MCQ eval set: medeval preflight --strict"
timeout 600 python -m medeval preflight configs/catalog_mcq.yaml --strict \
  || echo "[session-start] preflight flagged issues or timed out (non-blocking — see log above)"

echo "[session-start] ready."

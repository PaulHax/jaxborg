#!/usr/bin/env bash
# Automated parity loop: fuzzer finds gaps, coding agent fixes them.
#
# Usage:
#   bash scripts/parity_loop.sh              # uses codex by default
#   AGENT=claude bash scripts/parity_loop.sh # use claude code instead
#
# Environment variables:
#   AGENT           - "codex" (default) or "claude"
#   MAX_ITERATIONS  - max fix cycles (default: 20)
#   FUZZ_SEEDS      - number of seeds to fuzz (default: 20)
#   FUZZ_STEPS      - steps per seed (default: 100)
set -euo pipefail
cd "$(dirname "$0")/.."

AGENT="${AGENT:-codex}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20}"
FUZZ_SEEDS="${FUZZ_SEEDS:-20}"
FUZZ_STEPS="${FUZZ_STEPS:-100}"

invoke_agent() {
    local prompt="$1"
    case "$AGENT" in
        codex)
            codex exec --full-auto "$prompt"
            ;;
        claude)
            claude -p "$prompt" --allowedTools "Read,Write,Edit,Grep,Glob,Bash,Task,WebFetch"
            ;;
        *)
            echo "Unknown AGENT=$AGENT (use 'codex' or 'claude')" >&2
            exit 1
            ;;
    esac
}

for i in $(seq 1 "$MAX_ITERATIONS"); do
    echo ""
    echo "============================================"
    echo "  Parity Loop — Iteration $i / $MAX_ITERATIONS"
    echo "============================================"

    echo "Running fuzzer (seeds=0-$((FUZZ_SEEDS-1)), steps=$FUZZ_STEPS)..."
    output=$(uv run python -u -c "
from tests.differential.fuzzer import run_differential_fuzz
r = run_differential_fuzz(seeds=range($FUZZ_SEEDS), max_steps_per_seed=$FUZZ_STEPS, verbose=True)
if r:
    print(f'MISMATCH|{r.seed}|{r.step}|{r.field_name}|{r.host_or_agent}|{r.cyborg_value}|{r.jax_value}')
    print(r.all_diffs_str)
else:
    print('CLEAN')
")

    if echo "$output" | grep -q "^CLEAN$"; then
        echo ""
        echo "No mismatches found across $FUZZ_SEEDS seeds x $FUZZ_STEPS steps."
        echo "Parity achieved after $i iterations!"
        exit 0
    fi

    mismatch=$(echo "$output" | grep "^MISMATCH|" | head -1)
    diffs=$(echo "$output" | grep -v "^MISMATCH|" | grep -v "^---" | grep -v "^  Seed" | grep -v "^Total:" | grep -v "^Running" || true)

    IFS='|' read -r _ seed step field host cyborg_val jax_val <<< "$mismatch"
    echo ""
    echo "FOUND GAP: seed=$seed step=$step $field [$host] cyborg=$cyborg_val jax=$jax_val"
    echo "$diffs"

    prompt="The differential fuzzer found a CybORG/JAX state divergence.

Mismatch at seed=$seed, step=$step: $field [$host] cyborg=$cyborg_val jax=$jax_val

All diffs at this step:
$diffs

Follow the 'Fixing Differential Gaps' workflow in CLAUDE.md:
1. Write a failing regression test that reproduces this mismatch at the specific seed/step. Place it in the appropriate test file based on the subsystem involved (see CLAUDE.md for guidance), NOT always in test_fuzz_gaps.py.
2. Read the CybORG source at .venv/lib/python3.11/site-packages/CybORG/ to understand the root cause.
3. Fix the JAX code in src/jaxborg/ to match CybORG's behavior.
4. Verify: uv run pytest tests/ -v --ignore=tests/test_env_smoke.py --ignore=tests/test_training_parity.py -x
5. Lint: uv run ruff check --fix . && uv run ruff format .
6. Commit with a message describing the gap and fix."

    echo ""
    echo "Invoking $AGENT..."
    invoke_agent "$prompt"

    echo ""
    echo "Agent finished iteration $i. Continuing to next fuzzer run..."
done

echo ""
echo "Hit max iterations ($MAX_ITERATIONS). Run again or increase MAX_ITERATIONS."

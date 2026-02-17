#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MAX_RETRIES=3

while true; do
    NEXT_JSON=$(uv run python -c "
import json, sys
from tests.catalog import get_next_incomplete
s = get_next_incomplete()
if s is None:
    print('DONE')
else:
    print(json.dumps({
        'id': s.id,
        'name': s.name,
        'description': s.description,
        'cyborg_source_paths': s.cyborg_source_paths,
        'jax_target_files': s.jax_target_files,
    }))
")

    if [ "$NEXT_JSON" = "DONE" ]; then
        echo "All 22 subsystems complete!"
        break
    fi

    ID=$(echo "$NEXT_JSON" | uv run python -c "import json,sys; print(json.load(sys.stdin)['id'])")
    NAME=$(echo "$NEXT_JSON" | uv run python -c "import json,sys; print(json.load(sys.stdin)['name'])")
    DESC=$(echo "$NEXT_JSON" | uv run python -c "import json,sys; print(json.load(sys.stdin)['description'])")
    CYBORG_PATHS=$(echo "$NEXT_JSON" | uv run python -c "import json,sys; print(' '.join(json.load(sys.stdin)['cyborg_source_paths']))")
    JAX_FILES=$(echo "$NEXT_JSON" | uv run python -c "import json,sys; print(' '.join(json.load(sys.stdin)['jax_target_files']))")

    echo ""
    echo "================================================================"
    echo "  Subsystem $ID: $NAME"
    echo "  $DESC"
    echo "================================================================"
    echo ""

    ATTEMPT=0
    PASSED=false

    while [ $ATTEMPT -lt $MAX_RETRIES ]; do
        ATTEMPT=$((ATTEMPT + 1))
        echo "--- Attempt $ATTEMPT/$MAX_RETRIES ---"

        claude -p "You are porting CC4 (CAGE Challenge 4) from CybORG to JAX.

Subsystem $ID: $NAME
Description: $DESC

CybORG source files to read for exact behavior:
$CYBORG_PATHS

JAX target files to implement/modify:
$JAX_FILES

Workflow:
1. Read the CybORG source files to understand the exact behavior
2. Read the current JAX target files to understand existing code
3. Implement the JAX code for this subsystem
4. Create differential tests in tests/subsystems/test_${NAME}.py
5. Run: uv run pytest tests/subsystems/test_${NAME}.py -v
6. Fix any failures until tests pass
7. Run: uv run pytest tests/ -v --ignore=tests/test_env_smoke.py to verify no regressions
8. Mark subsystem as passing: uv run python -c \"from tests.catalog import mark_passing; mark_passing($ID)\"
9. Commit changes: git add -A && git commit -m 'subsystem $ID: $NAME'

Important:
- All host-indexed arrays are padded to GLOBAL_MAX_HOSTS=137, use host_active mask
- 5 blue agents, 6 red agents (hardcoded, no masking)
- Use flax.struct.dataclass for state, jax.lax.cond for branching
- Reference CC2 port at jaxmarl/integration/jaxmarl/environments/cage/ for patterns
" \
            --allowedTools "Read,Edit,Write,Bash,Grep,Glob"

        if uv run pytest "tests/subsystems/test_${NAME}.py" -v 2>/dev/null; then
            PASSED=true
            break
        else
            echo "Tests failed on attempt $ATTEMPT"
        fi
    done

    if [ "$PASSED" = false ]; then
        echo "FAILED after $MAX_RETRIES attempts on subsystem $ID: $NAME"
        echo "Stopping loop. Fix manually, then re-run this script."
        exit 1
    fi

    echo ""
    echo "Subsystem $ID ($NAME) complete."
    echo "Review the commit, then press Enter to continue (or Ctrl-C to stop)."
    read -r
done

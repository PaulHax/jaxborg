#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MAX_RETRIES=${MAX_RETRIES:-5}
MAX_SUBSYSTEMS=${MAX_SUBSYSTEMS:-5}
COMPLETED=0

CYBORG_PREFIX=".venv/lib/python3.11/site-packages"

while [ $COMPLETED -lt $MAX_SUBSYSTEMS ]; do
    NEXT_JSON=$(uv run python -c "
import json
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
        'depends_on': s.depends_on,
    }))
")

    if [ "$NEXT_JSON" = "DONE" ]; then
        echo "All 22 subsystems complete!"
        break
    fi

    ID=$(echo "$NEXT_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
    NAME=$(echo "$NEXT_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['name'])")
    DESC=$(echo "$NEXT_JSON" | python3 -c "import json,sys; print(json.load(sys.stdin)['description'])")
    CYBORG_PATHS=$(echo "$NEXT_JSON" | python3 -c "import json,sys; print(' '.join(json.load(sys.stdin)['cyborg_source_paths']))")
    JAX_FILES=$(echo "$NEXT_JSON" | python3 -c "import json,sys; print(' '.join(json.load(sys.stdin)['jax_target_files']))")

    echo ""
    echo "================================================================"
    echo "  Subsystem $ID: $NAME ($((COMPLETED + 1))/$MAX_SUBSYSTEMS)"
    echo "  $DESC"
    echo "================================================================"
    echo ""

    ATTEMPT=0
    PASSED=false

    while [ $ATTEMPT -lt $MAX_RETRIES ]; do
        ATTEMPT=$((ATTEMPT + 1))
        echo "--- Attempt $ATTEMPT/$MAX_RETRIES ---"

        claude -p "You are porting CC4 (CAGE Challenge 4) from CybORG to JAX.
Working directory: $(pwd)

Subsystem $ID: $NAME
Description: $DESC

CybORG source files to read for exact behavior (prefix with $CYBORG_PREFIX/):
$CYBORG_PATHS

JAX target files to implement/modify:
$JAX_FILES

Workflow:
1. Read the CybORG source files to understand the exact behavior.
   The installed CybORG is at $CYBORG_PREFIX/CybORG/
   Use Glob and Read to find and read the relevant files.
2. Read the current JAX target files to understand existing code.
   Read src/jaxborg/constants.py and src/jaxborg/state.py for the schema.
3. Implement the JAX code for this subsystem.
4. Create differential tests in tests/subsystems/test_${NAME}.py
   The tests should verify JAX behavior matches CybORG where possible.
   If CybORG is not available, tests should verify JAX logic independently.
5. Run: uv run pytest tests/subsystems/test_${NAME}.py -v
6. Fix any failures until tests pass.
7. Run: uv run pytest tests/ -v --ignore=tests/test_env_smoke.py to verify no regressions.
8. Mark subsystem as passing:
   uv run python -c \"from tests.catalog import mark_passing; mark_passing($ID)\"
9. Commit: git add -A && git commit -m 'subsystem $ID: $NAME'

Important:
- All host-indexed arrays are padded to GLOBAL_MAX_HOSTS=137, use host_active mask for real hosts
- 5 blue agents, 6 red agents (hardcoded, no masking)
- Use flax.struct.dataclass for state, jax.lax.cond for branching
- Reference CC2 port at /home/paulhax/src/cyber/jaxmarl/integration/jaxmarl/environments/cage/ for patterns
- pyproject.toml has pythonpath=[\".\"] so tests can import tests.catalog etc.
- Do not add comments describing what you changed
- Prefer functional programming
" \
            --allowedTools "Read,Edit,Write,Bash(uv run*),Bash(git add*),Bash(git commit*),Bash(git status*),Bash(git diff*),Bash(ls*),Bash(python3*),Grep,Glob"

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

    COMPLETED=$((COMPLETED + 1))
    echo ""
    echo "Subsystem $ID ($NAME) complete. [$COMPLETED/$MAX_SUBSYSTEMS done]"
done

echo ""
echo "Completed $COMPLETED subsystems."
echo "Review the commits, then press Enter to continue (or Ctrl-C to stop)."
read -r

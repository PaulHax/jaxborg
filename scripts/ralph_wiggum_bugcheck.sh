#!/usr/bin/env bash
# Bug-hunting differential test runner.
#
# Runs targeted differential tests that would catch the 40 bugs found in
# the CC2 jaxmarl-integration port. If any of these patterns exist in the
# CC4 jaxborg port, the tests will fail.
#
# Usage:
#   ./scripts/ralph_wiggum_bugcheck.sh          # run all bugcheck tests
#   ./scripts/ralph_wiggum_bugcheck.sh -k obs   # run only obs-related tests
#   ./scripts/ralph_wiggum_bugcheck.sh -v        # verbose output

set -euo pipefail

cd "$(dirname "$0")/.."

echo "================================================================"
echo "  ralph_wiggum bugcheck: CC2 bug pattern differential tests"
echo "================================================================"
echo ""
echo "Testing for 40 known CC2 bug patterns in the CC4 jaxborg port:"
echo "  - Host ordering (alphabetical)"
echo "  - Observation encoding parity"
echo "  - Blue action encoding (Restore/Decoy order)"
echo "  - Red exploit encoding (host/type)"
echo "  - Red initial foothold privilege level"
echo "  - Remove behavior (user-level only)"
echo "  - Restore behavior (preserves foothold)"
echo "  - Exploit determinism"
echo "  - Impact prerequisites"
echo "  - Exploit privilege levels"
echo "  - Reward calculation"
echo "  - Decoy mechanics"
echo "  - Subnet adjacency"
echo "  - Monitor behavior"
echo "  - Full episode state parity"
echo "  - Action mask parity across steps"
echo ""

EXTRA_ARGS="${*:-}"

echo "--- Running bugcheck tests ---"
uv run pytest tests/test_cc2_regression.py -v $EXTRA_ARGS

echo ""
echo "--- Running existing differential tests ---"
uv run pytest tests/test_action_mask_differential.py -v $EXTRA_ARGS

echo ""
echo "--- Running subsystem fuzz tests ---"
uv run pytest tests/subsystems/test_full_episode_fuzzing.py -v $EXTRA_ARGS

echo ""
echo "================================================================"
echo "  All bugcheck tests passed!"
echo "================================================================"

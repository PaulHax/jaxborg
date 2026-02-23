# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

JAX port of CybORG CAGE Challenge 4 (CC4) — a multi-agent cybersecurity simulation (9 subnets, ~80 hosts, 5 blue agents, 6 red agents, 3 mission phases). Re-implements CC4 as JIT-compilable JAX arrays for GPU-accelerated parallel RL training via JaxMARL's `MultiAgentEnv` interface.

## Commands

```bash
uv sync                                              # install deps
uv run pytest tests/ -v --ignore=tests/test_env_smoke.py  # all tests (skip slow CybORG smoke)
uv run pytest tests/subsystems/test_red_discover.py -v     # single test file
uv run pytest tests/subsystems/test_red_discover.py::TestClassName::test_name -v  # single test
```

Training output goes to `../jaxborg-exp/`.

## Architecture

### Core Data Structures (`src/jaxborg/state.py`)

Two `flax.struct.dataclass` PyTrees:

- **CC4Const** — static topology: host properties, subnet adjacency, data links, agent assignments, phase rewards. Built once per episode via `build_topology()` (pure JAX) or `build_const_from_cyborg()` (extracts from CybORG instance).
- **CC4State** — dynamic per-step state: compromise levels, red sessions/privilege/discovery, activity tracking, decoys, blocked zones, messages, FSM states. Created via `create_initial_state()`.

All host-indexed arrays are padded to `GLOBAL_MAX_HOSTS=137` with `host_active` masking. State updates use `state.replace(field=new_value)` and `array.at[idx].set(value)`.

### Action System (`src/jaxborg/actions/`)

Actions are integer-encoded. `encoding.py` defines the action space layout (ranges of ints mapping to action type + target). Red actions dispatch through `apply_red_action()` which decodes then branches via `jax.lax.cond` to per-action handlers (discover, scan, 8 exploit types, privesc, impact). Blue actions dispatch through `apply_blue_action()`.

Each action module (e.g., `red_exploit.py`, `blue_monitor.py`) exports an `apply_*` function: `(CC4State, CC4Const, agent_id, target) -> CC4State`.

### Topology (`src/jaxborg/topology.py`)

Builds the static `CC4Const` from seed. Hardcoded subnet adjacency (NACLs), router backbone, host generation with per-subnet counts. Two entry points: `build_topology(seeds, num_steps)` for pure JAX, `build_const_from_cyborg(env)` for extracting from a live CybORG instance.

## Development Workflow

Implementation is driven by a 22-subsystem catalog (`tests/catalog.py`) with dependency ordering. `scripts/ralph_wiggum.sh` automates the loop: get next subsystem, implement JAX code + differential tests, validate, commit.

Check progress: `tests/catalog_status.json` tracks which subsystems are passing.

### Differential Testing

Every test must compare JAX output against CybORG. Pattern:
1. Create CybORG env via `cyborg_env` fixture (conftest.py): SleepAgent blue, EnterpriseGreenAgent green, FiniteStateRedAgent red, seed=42, 500 steps
2. Build JAX const: `build_const_from_cyborg(cyborg_env)`
3. Create JAX state: `create_initial_state()`
4. Execute identical action sequences in both environments
5. Compare relevant state fields (host_compromised, red_sessions, red_privilege, etc.)

Test infrastructure lives in `tests/differential/` (harness, action translator, state comparator).

**Tests must exercise the training code path.** Differential tests should call the same functions used during training (`FsmRedCC4Env.step_env`, `apply_red_action`, `apply_green_agents`, `fsm_red_post_step_update`, etc.) rather than reimplementing logic in test code. Shared functions like `fsm_red_post_step_update` are extracted so both the training env and the differential harness use the same code. If a test needs custom orchestration (e.g. action duration tracking for CybORG parity), the JAX-side logic should still call into production functions.

### TDD + Differential Test Quality (Required)

For every fuzzer-found parity bug, follow strict red-green-refactor:
1. **Red**: add a failing regression test first.
2. **Green**: implement the smallest JAX fix that makes that exact test pass.
3. **Refactor**: clean up only after parity is restored and tests stay green.

Primary regression tests for parity bugs must be **explicit differential tests**:
- Run both CybORG and JAX in the same test (typically via `CC4DifferentialHarness`).
- Pin the exact reproduction context (seed, step, agent/action, hostnames/indices, field).
- Assert concrete before/after state on both sides, then assert no diff for the target field.
- Do not rely only on generic assertions like "no diffs at step N" without explicit context checks.

Unit tests are allowed as secondary guardrails, but they do **not** replace the required differential regression for parity bugs.

### Precomputed Randoms for Deterministic Testing

CybORG and JAX use independent RNG streams, making direct comparison of random-dependent behavior (green agents, detection rolls) impossible without synchronization. The solution is precomputed random arrays stored in `CC4State`:

- **Green agents**: `green_randoms` `(MAX_STEPS, GLOBAL_MAX_HOSTS, 7)` float array + `use_green_randoms` bool. The 7 fields encode action choice, service selection, reliability roll, FP roll, phishing roll, dest host, and access FP roll as `[0,1)` uniforms. `sample_green_random()` in `actions/rng.py` uses `jax.lax.cond` to read from the array or fall back to JAX RNG. `tests/differential/green_recorder.py` wraps CybORG's `np_random` to capture values, then the harness injects them into JAX state via `sync_green_rng=True`.
- **Detection**: `detection_randoms` flat sequence + `detection_random_index` counter. `sample_detection_random()` in `actions/rng.py`.

To run green parity tests: `uv run pytest tests/test_green_parity.py -v` (requires CybORG).
To run green unit tests (no CybORG): `uv run pytest tests/test_green_unit.py -v`.

## Fixing Differential Gaps

The differential fuzzer (`tests/differential/fuzzer.py`) systematically finds CybORG/JAX state divergences by running full episodes across many seeds. Run it:

```bash
uv run python -m tests.differential.fuzzer          # 20 seeds × 100 steps
```

It returns a `MismatchReport` on the first error: seed, step, field name, CybORG vs JAX values.

### Investigation workflow

1. **Reproduce (TDD red)**: write a failing **explicit differential regression** that runs CybORG and JAX to the failing seed/step and asserts the exact mismatch context.
   - Prefer `tests/differential/test_parity_gaps.py` for fuzzer-reported seed/step reproductions.
   - If adding to a subsystem file, keep it differential (pure CybORG + JAX in the same test), not JAX-only.
   - One gap at a time: stop on first mismatch, add test, fix, rerun.
2. Place the test in the appropriate file:
   - Red action bugs → `tests/subsystems/test_red_*.py` (exploit, discover, privesc, etc.)
   - Blue action bugs → `tests/subsystems/test_blue_*.py`
   - Green/phishing bugs → `tests/test_green_parity.py`
   - Session/topology bugs → `tests/subsystems/test_dynamic_topology.py`
   - FSM state machine bugs → `tests/subsystems/test_fsm_red_agent.py`
   - Reward/phase bugs → `tests/subsystems/test_rewards.py` or `tests/subsystems/test_phase_transitions.py`
   - Cross-cutting or unclear → `tests/test_fuzz_gaps.py` (catch-all for fuzzer-found gaps)
3. **Read CybORG source** at `.venv/lib/python3.11/site-packages/CybORG/` to understand the mechanic causing the divergence
4. **Fix the JAX code** in `src/jaxborg/` to match CybORG's behavior
5. **Verify targeted tests first** (new regression + closest subsystem tests), then run:
   - `uv run pytest tests/ -v --ignore=tests/test_env_smoke.py --ignore=tests/test_training_parity.py -x`
   - rerun differential fuzzing to find the next gap
6. **Lint**: `uv run ruff check --fix . && uv run ruff format .`
7. **Commit** with a message describing the gap and fix

### Common root causes

- JAX doesn't model a CybORG mechanic (action duration, session reassignment, observation-based tracking)
- JAX success conditions differ from CybORG (missing service check, wrong subnet logic, no decoy handling)
- Translator creates wrong CybORG action type or missing parameters (e.g. ExploitRemoteService needs a FixedExploitSelector)
- Timing: CybORG multi-tick actions (duration > 1) need deferred JAX execution via `_red_pending_jax` in the harness

### Key CybORG entry points for investigation

- `SimulationController.step()` — action submission, `actions_in_progress` duration tracking
- `ActionSpace.update()` — observation-based validity tracking (subnets, IPs)
- `replace_action_if_invalid()` — action type/parameter validation
- `ExploitRemoteService.execute()` — delegates to concrete exploit via selector
- `FiniteStateRedAgent` — CybORG's FSM, uses `ExploitRemoteService` (duration=4)
- `different_subnet_agent_reassignment()` — green phishing session transfers

## Linting

Run `uv run ruff check --fix . && uv run ruff format .` before committing.

## JAX Constraints

- `jax.lax.cond()` for branching (no Python if/else in JIT code)
- No Python loops over dynamic values in JIT code
- `flax.struct.dataclass` for PyTree-compatible state
- Use `numpy` for host indexing in tests; `jax.numpy` for JIT-compiled logic

## Reference

CC2 JAX port at `/home/paulhax/src/cyber/jaxmarl/integration/jaxmarl/environments/cage/` for patterns.

CybORG source installed at `.venv/lib/python3.11/site-packages/CybORG/`.

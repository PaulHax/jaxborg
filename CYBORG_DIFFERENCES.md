# CybORG CC4 JAX Port: Known Differences & Simplifications

Documented after a line-by-line code review of every JAX implementation file against
the CybORG source at `.venv/lib/python3.11/site-packages/CybORG/`.

## Detection Rates

### ~~AggressiveServiceDiscovery: always detected (CybORG: 75%)~~ RESOLVED

Now matches CybORG: 75% detection rate via `sample_detection_random()`. Uses precomputed
RNG sequences in differential tests for exact reproducibility.

### ~~StealthServiceDiscovery: never detected (CybORG: 25%)~~ RESOLVED

Now matches CybORG: 25% detection rate via `sample_detection_random()`. Sets
`ACTIVITY_SCAN` when detected, matching CybORG's behavior.

### ~~DiscoverDeception: deterministic (CybORG: probabilistic)~~ RESOLVED

Now matches CybORG: 50% true positive rate, 10% false positive rate. Two calls to
`sample_detection_random()` for TP and FP rolls.

## ~~Blue Remove: Malware Clearing~~ RESOLVED

Now matches CybORG: Remove only kills user-level sessions. Malware files persist until
Restore, so Analyse will still detect a compromised host after a successful Remove.

## ~~Green Agent: False Positive Observation Field~~ RESOLVED

Now matches CybORG: GreenLocalWork false positive sets `host_activity_detected` (which
maps to the `network_connections` observation field), not `host_has_malware`.

## ~~Service Reliability Degradation~~ RESOLVED

Now matches CybORG: `host_service_reliability` tracks per-service reliability (0-100,
default 100). `DegradeServices` decrements by 20 (clamped to 0). `GreenLocalWork` picks
a random active service and checks `randint(0, 100) < reliability` before executing.
`Restore` resets reliability to 100.

## ~~Privilege Escalation: No Sandbox Check~~ RESOLVED

Now matches CybORG: `apply_privesc` checks `red_session_sandboxed[agent_id, target_host]`.
If the session was created by exploiting a sandboxed decoy process, escalation fails and
the session is removed. None of the 4 CC4 decoy types currently use `SANDBOXING_EXPLOIT`,
but the check is in place for future decoy types.

## ~~FSM Red Agent: No U→F Subnet Guard~~ RESOLVED

Now matches CybORG: `fsm_red_update_state` checks if the target host is in the agent's
allowed subnets (via `red_agent_subnets` in CC4Const). If a host would transition to
state U but is outside the agent's subnets, it transitions to F instead.

## ~~Withdraw: Unconditional host_compromised Clear~~ RESOLVED

Now matches CybORG: `apply_withdraw` computes the max remaining privilege across all red
agents after clearing the withdrawing agent's session. `host_compromised` is set to that
max (which is `COMPROMISE_NONE` only if no other agent has privilege). `host_has_malware`
is only cleared if no other agent has a session on the host.

## Observation Layout: Fixed vs Variable Body Size

CybORG's `BlueFlatWrapper` builds observations with a variable-length body depending on how
many subnets the agent oversees. Agents 0-3 observe 1 subnet (body=60), agent 4 observes 3
subnets (body=178). Messages are appended right after the body, then the observation is
padded to 210 elements.

The JAX implementation always includes 3 subnet blocks (zero-filled for unassigned subnets),
giving a fixed body of 178 for all agents. Messages are always the last 32 elements.

For agents 0-3, this means the message section is at different indices:
- CybORG: `obs[60:92]` with padding at `obs[92:210]`
- JAX: `obs[178:210]` with zero blocks at `obs[60:178]`

The information content is identical — same subnet data, same messages — but at different
positions. This is fine for training (consistent structure is better), but means raw
observation vectors are not directly comparable between CybORG and JAX for agents 0-3.

- JAX: `observations.py` — always 3 blocks + messages
- CybORG: `BlueFlatWrapper.observation_change()` — variable blocks + messages + padding

## CC4Env: Red Agents Exposed as Controllable (CybORG: Scripted FSM)

CybORG CC4 is a **blue-only training** environment. Red agents always run the
`FiniteStateRedAgent` FSM internally — they are not controllable by an RL policy. The FSM
reads directly from state (`fsm_host_states`, `red_discovered_hosts`, `host_active`),
picks a random eligible host, samples an action from a hardcoded probability matrix, and
does not use observations at all.

The JAX `CC4Env` currently exposes red agents as controllable agents in the action dict
(6 red agents alongside 5 blue agents, 11 total). This was a design choice to support
potential future red-blue self-play, but diverges from CC4's intended use where only blue
agents are trained.

For standard CC4 training, red actions should be generated internally by `fsm_red_get_action()`
during `step_env()`, and only the 5 blue agents should appear in the action/observation/reward
dicts.

To make red agents genuinely trainable for self-play, additional work is needed:
- **Red observations**: `get_red_obs()` currently returns zeros — needs proper encoding of
  discovered hosts, session states, privilege levels, and FSM progress
- **Red action masking**: `get_avail_actions()` returns all-ones for red — needs masks to
  prevent invalid actions (e.g., scanning undiscovered hosts, exploiting without sessions)
- **Red obs space**: currently set to `BLUE_OBS_SIZE` as a placeholder

- JAX: `env.py` — red agents in `self.agents`, receive actions from caller
- CybORG: `EnterpriseScenarioGenerator` — red always `FiniteStateRedAgent`, not trainable

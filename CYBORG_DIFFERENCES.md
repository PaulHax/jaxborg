# CybORG CC4 JAX Port: Known Differences & Simplifications

Documented after a line-by-line code review of every JAX implementation file against
the CybORG source at `.venv/lib/python3.11/site-packages/CybORG/`.

## Detection Rates

### AggressiveServiceDiscovery: always detected (CybORG: 75%)
CybORG's `AggressiveServiceDiscovery` has `detection_rate = 0.75`, meaning 25% of
aggressive scans go undetected. The JAX implementation always sets
`red_activity_this_step = ACTIVITY_SCAN`, making every aggressive scan visible to
Monitor. This makes blue's job slightly easier than in CybORG.

- JAX: `red_aggressive_scan.py` — unconditionally sets activity flag
- CybORG: `AggressiveServiceDiscovery.py` — `self.detection_rate = 0.75`

### StealthServiceDiscovery: never detected (CybORG: 25%)
CybORG's `StealthServiceDiscovery` has `detection_rate = 0.25`, meaning 25% of stealth
scans ARE detected. The JAX implementation never sets the activity flag, making stealth
scans completely invisible. This makes blue's job slightly harder than in CybORG.

- JAX: `red_stealth_scan.py` — does not set activity flag
- CybORG: `StealthServiceDiscovery.py` — `self.detection_rate = 0.25`

### DiscoverDeception: deterministic (CybORG: probabilistic)
CybORG's `DiscoverDeception` has a 50% true positive rate (correctly identifies decoys)
and 10% false positive rate (incorrectly flags real services as decoys). The JAX
implementation deterministically detects all decoys with no false positives.

- JAX: `red_discover_deception.py` — checks `jnp.any(state.host_decoys[target_host])`
- CybORG: `DiscoverDeception.py` — `tp_rate=0.5`, `fp_rate=0.1`

## Blue Remove: Malware Clearing

CybORG's `Remove` action kills suspicious processes/sessions but does NOT delete malware
files from disk. This means `Analyse` (which checks for malware files via DensityScout +
SigCheck) would still detect the host as compromised after a successful Remove.

The JAX implementation clears `host_has_malware` when Remove successfully eliminates all
user-level red sessions on a host. This means Analyse would NOT detect the host after a
successful Remove.

- JAX: `blue_remove.py:33-37` — clears `host_has_malware` when `~any_compromised`
- CybORG: `Remove.py` — only calls `StopProcess`, files persist until `Restore`

This is tested behavior (test_blue_remove.py validates malware clearing).

## Green Agent: False Positive Observation Field

CybORG's `GreenLocalWork` false positive creates a `process_creation` event, which
Monitor picks up as a network_connections observation (the `_get_connections` path in
BlueFlatWrapper).

The JAX implementation sets `host_has_malware = True`, which maps to the
`malicious_processes` observation field instead.

Both result in the blue agent seeing suspicious activity on the host, but in different
observation indices.

- JAX: `green.py:128-132` — sets `host_has_malware`
- CybORG: `GreenLocalWork.py` — creates `process_creation` event → `network_connections` obs

## Service Reliability Degradation

CybORG's `DegradeServices` action calls `degrade_service_reliability()` on host services,
reducing their reliability score. `GreenLocalWork` checks service reliability before
executing (step 2), so degraded services can cause green actions to fail silently before
the FP/phishing checks run.

The JAX implementation does not model service reliability. `DegradeServices` only sets the
activity detection flag. Green agents always have a chance to trigger FP/phishing when
services exist on the host.

- JAX: `red_degrade.py` — sets activity flag only
- CybORG: `DegradeServices.py` → `Host.degrade_service_reliability()`
- JAX: `green.py` — checks `jnp.any(state.host_services[host_idx])` (binary)
- CybORG: `GreenLocalWork.py` step 2 — checks `service.active` and reliability

## Privilege Escalation: No Sandbox Check

CybORG's `PrivilegeEscalate` checks `is_escalate_sandbox` on red sessions during
escalation. If a session was created by exploiting a sandboxed decoy process, escalation
fails.

The JAX implementation skips this check. However, none of the 4 CC4 decoy types
(HarakaSMPT, Apache, Tomcat, Vsftpd) have `DecoyType.ESCALATE` (they all use
`DecoyType.EXPLOIT`), so sandboxed sessions never occur in CC4. No practical impact.

- JAX: `red_privesc.py` — no sandbox check
- CybORG: `PrivilegeEscalate.py` → checks `is_escalate_sandbox`

## FSM Red Agent: No U→F Subnet Guard

CybORG's `FiniteStateRedAgent` transitions a host from state U (user access) to F
(failed) if the host is outside the red agent's allowed subnets. This prevents the FSM
from attempting further actions on unreachable hosts.

The JAX implementation does not implement this guard. In practice, red agents can only
reach hosts they've discovered via `DiscoverRemoteSystems` on their assigned subnets, so
out-of-subnet hosts would never reach state U through normal gameplay.

- JAX: `fsm_red.py` — no subnet check on U state
- CybORG: `FiniteStateRedAgent.py` — `if subnet not in self.allowed_subnets: state = 'F'`

## Withdraw: Unconditional host_compromised Clear

`apply_withdraw` sets `host_compromised[target_host] = COMPROMISE_NONE` unconditionally
when a red agent withdraws. If multiple red agents have sessions on the same host, one
agent withdrawing clears the global compromised flag even though another agent still has
access.

In CybORG, `host_compromised` is implicit from active sessions/processes, so withdrawing
one agent doesn't affect the other's presence.

In CC4 practice, multiple red agents sharing the same host is rare since they start in
different zones and operate on separate subnets.

- JAX: `red_withdraw.py` — unconditionally sets `COMPROMISE_NONE`
- CybORG: compromise derived from session state, not a single flag

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

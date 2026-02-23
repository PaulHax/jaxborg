import numpy as np

from jaxborg.constants import (
    NUM_RED_AGENTS,
    NUM_SUBNETS,
)
from jaxborg.translate import CC4Mappings
from tests.differential.harness import StateDiff, StateSnapshot

_ERROR_FIELDS = {
    "host_compromised",
    "red_privilege",
    "red_sessions",
    "host_decoys",
    "ot_service_stopped",
    "blocked_zones",
    "mission_phase",
    "rewards",
}

_WARNING_FIELDS = {
    "red_discovered_hosts",
    "red_scanned_hosts",
    "host_activity_detected",
}


def extract_cyborg_snapshot(cyborg_env, mappings: CC4Mappings) -> StateSnapshot:
    state = cyborg_env.environment_controller.state
    sorted_hosts = sorted(state.hosts.keys())

    host_compromised = {}
    host_services = {}
    host_decoys = {}
    ot_service_stopped = {}
    for hostname in sorted_hosts:
        idx = mappings.hostname_to_idx[hostname]
        max_level = 0
        for agent_name, sessions in state.sessions.items():
            if not agent_name.startswith("red_agent_"):
                continue
            for sess in sessions.values():
                if sess.hostname == hostname:
                    if hasattr(sess, "username"):
                        if sess.username in ("root", "SYSTEM"):
                            max_level = max(max_level, 2)
                        else:
                            max_level = max(max_level, 1)
                    else:
                        max_level = max(max_level, 1)
        host_compromised[idx] = max_level

    red_privilege = {}
    red_sessions = {}
    for agent_name, sessions in state.sessions.items():
        if not agent_name.startswith("red_agent_"):
            continue
        red_idx = int(agent_name.split("_")[-1])
        if red_idx >= NUM_RED_AGENTS:
            continue
        session_hosts = set()
        priv = {}
        for sess in sessions.values():
            if sess.hostname in mappings.hostname_to_idx:
                hidx = mappings.hostname_to_idx[sess.hostname]
                session_hosts.add(hidx)
                level = 1
                if hasattr(sess, "username") and sess.username in ("root", "SYSTEM"):
                    level = 2
                priv[hidx] = max(priv.get(hidx, 0), level)
        red_sessions[red_idx] = session_hosts
        red_privilege[red_idx] = priv

    blocked = set()
    for to_subnet, from_list in getattr(state, "blocks", {}).items():
        for from_subnet in from_list:
            if from_subnet in mappings.cidr_to_subnet_idx or from_subnet in {v for v in mappings.subnet_names.values()}:
                pass
            blocked.add((str(from_subnet), str(to_subnet)))

    mission_phase = getattr(state, "mission_phase", 0)

    return StateSnapshot(
        time=getattr(state, "step_count", 0),
        mission_phase=mission_phase,
        host_compromised=host_compromised,
        red_privilege=red_privilege,
        red_sessions=red_sessions,
        host_services=host_services,
        host_decoys=host_decoys,
        ot_service_stopped=ot_service_stopped,
        blocked_zones=blocked,
    )


def extract_jax_snapshot(state, const, mappings: CC4Mappings) -> StateSnapshot:
    n = mappings.num_hosts
    host_compromised = {}
    for h in range(n):
        host_compromised[h] = int(state.host_compromised[h])

    red_privilege = {}
    red_sessions = {}
    for r in range(NUM_RED_AGENTS):
        sessions_set = set()
        priv = {}
        for h in range(n):
            if bool(state.red_sessions[r, h]):
                sessions_set.add(h)
            p = int(state.red_privilege[r, h])
            if p > 0:
                priv[h] = p
        red_sessions[r] = sessions_set
        red_privilege[r] = priv

    host_decoys = {}
    for h in range(n):
        decoys = np.array(state.host_decoys[h])
        if np.any(decoys):
            host_decoys[h] = tuple(bool(d) for d in decoys)

    ot_service_stopped = {}
    for h in range(n):
        if bool(state.ot_service_stopped[h]):
            ot_service_stopped[h] = True

    blocked = set()
    blocked_arr = np.array(state.blocked_zones)
    for src in range(NUM_SUBNETS):
        for dst in range(NUM_SUBNETS):
            if blocked_arr[src, dst]:
                src_name = mappings.subnet_names.get(src, str(src))
                dst_name = mappings.subnet_names.get(dst, str(dst))
                blocked.add((src_name, dst_name))

    return StateSnapshot(
        time=int(state.time),
        mission_phase=int(state.mission_phase),
        host_compromised=host_compromised,
        red_privilege=red_privilege,
        red_sessions=red_sessions,
        host_decoys=host_decoys,
        ot_service_stopped=ot_service_stopped,
        blocked_zones=blocked,
    )


def compare_snapshots(cyborg: StateSnapshot, jax: StateSnapshot) -> list[StateDiff]:
    diffs = []

    if cyborg.mission_phase != jax.mission_phase:
        diffs.append(StateDiff("mission_phase", cyborg.mission_phase, jax.mission_phase))

    for h in set(cyborg.host_compromised) | set(jax.host_compromised):
        cv = cyborg.host_compromised.get(h, 0)
        jv = jax.host_compromised.get(h, 0)
        if cv != jv:
            diffs.append(StateDiff("host_compromised", cv, jv, f"host_{h}"))

    for r in set(cyborg.red_sessions) | set(jax.red_sessions):
        cs = cyborg.red_sessions.get(r, set())
        js = jax.red_sessions.get(r, set())
        if cs != js:
            diffs.append(StateDiff("red_sessions", cs, js, f"red_agent_{r}"))

    for r in set(cyborg.red_privilege) | set(jax.red_privilege):
        cp = cyborg.red_privilege.get(r, {})
        jp = jax.red_privilege.get(r, {})
        for h in set(cp) | set(jp):
            cv = cp.get(h, 0)
            jv = jp.get(h, 0)
            if cv != jv:
                diffs.append(StateDiff("red_privilege", cv, jv, f"red_{r}_host_{h}"))

    for h in set(cyborg.host_decoys) | set(jax.host_decoys):
        cd = cyborg.host_decoys.get(h)
        jd = jax.host_decoys.get(h)
        if cd != jd:
            diffs.append(StateDiff("host_decoys", cd, jd, f"host_{h}"))

    for h in set(cyborg.ot_service_stopped) | set(jax.ot_service_stopped):
        cv = cyborg.ot_service_stopped.get(h, False)
        jv = jax.ot_service_stopped.get(h, False)
        if cv != jv:
            diffs.append(StateDiff("ot_service_stopped", cv, jv, f"host_{h}"))

    if cyborg.blocked_zones != jax.blocked_zones:
        diffs.append(StateDiff("blocked_zones", cyborg.blocked_zones, jax.blocked_zones))

    if cyborg.rewards != jax.rewards:
        diffs.append(StateDiff("rewards", cyborg.rewards, jax.rewards))

    return diffs


def snapshots_match(cyborg: StateSnapshot, jax: StateSnapshot) -> bool:
    return len(compare_snapshots(cyborg, jax)) == 0


def compare_fast(cyborg_env, jax_state, jax_const, mappings) -> list[StateDiff]:
    """Array-based comparison — bulk-transfers JAX arrays to numpy instead of per-element access."""
    n = mappings.num_hosts
    diffs = []

    # JAX arrays → numpy in bulk (one transfer per array)
    jax_compromised = np.asarray(jax_state.host_compromised[:n])
    jax_sessions = np.asarray(jax_state.red_sessions[:NUM_RED_AGENTS, :n])
    jax_priv = np.asarray(jax_state.red_privilege[:NUM_RED_AGENTS, :n])
    jax_decoys = np.asarray(jax_state.host_decoys[:n])
    jax_ot_stopped = np.asarray(jax_state.ot_service_stopped[:n])
    jax_blocked = np.asarray(jax_state.blocked_zones)
    jax_phase = int(jax_state.mission_phase)

    # CybORG → numpy arrays
    cyborg_state = cyborg_env.environment_controller.state
    cyborg_compromised = np.zeros(n, dtype=np.int32)
    cyborg_sessions = np.zeros((NUM_RED_AGENTS, n), dtype=np.bool_)
    cyborg_priv = np.zeros((NUM_RED_AGENTS, n), dtype=np.int32)

    for agent_name, sessions in cyborg_state.sessions.items():
        if not agent_name.startswith("red_agent_"):
            continue
        red_idx = int(agent_name.split("_")[-1])
        if red_idx >= NUM_RED_AGENTS:
            continue
        for sess in sessions.values():
            if sess.hostname not in mappings.hostname_to_idx:
                continue
            hidx = mappings.hostname_to_idx[sess.hostname]
            cyborg_sessions[red_idx, hidx] = True
            level = 2 if (hasattr(sess, "username") and sess.username in ("root", "SYSTEM")) else 1
            cyborg_priv[red_idx, hidx] = max(cyborg_priv[red_idx, hidx], level)
            cyborg_compromised[hidx] = max(cyborg_compromised[hidx], level)

    cyborg_phase = getattr(cyborg_state, "mission_phase", 0)

    # Compare with vectorized ops
    if cyborg_phase != jax_phase:
        diffs.append(StateDiff("mission_phase", cyborg_phase, jax_phase))

    comp_mismatch = np.where(cyborg_compromised != jax_compromised)[0]
    for h in comp_mismatch:
        diffs.append(StateDiff("host_compromised", int(cyborg_compromised[h]), int(jax_compromised[h]), f"host_{h}"))

    for r in range(NUM_RED_AGENTS):
        if not np.array_equal(cyborg_sessions[r], jax_sessions[r]):
            cs = set(np.where(cyborg_sessions[r])[0].tolist())
            js = set(np.where(jax_sessions[r])[0].tolist())
            diffs.append(StateDiff("red_sessions", cs, js, f"red_agent_{r}"))

    for r in range(NUM_RED_AGENTS):
        priv_mismatch = np.where(cyborg_priv[r] != jax_priv[r])[0]
        for h in priv_mismatch:
            diffs.append(StateDiff("red_privilege", int(cyborg_priv[r, h]), int(jax_priv[r, h]), f"red_{r}_host_{h}"))

    if np.any(jax_decoys):
        for h in range(n):
            if np.any(jax_decoys[h]):
                diffs.append(StateDiff("host_decoys", None, tuple(bool(d) for d in jax_decoys[h]), f"host_{h}"))

    if np.any(jax_ot_stopped):
        for h in np.where(jax_ot_stopped)[0]:
            diffs.append(StateDiff("ot_service_stopped", False, True, f"host_{h}"))

    cyborg_blocked = set()
    for to_subnet, from_list in getattr(cyborg_state, "blocks", {}).items():
        for from_subnet in from_list:
            cyborg_blocked.add((str(from_subnet), str(to_subnet)))
    jax_blocked_set = set()
    for src in range(NUM_SUBNETS):
        for dst in range(NUM_SUBNETS):
            if jax_blocked[src, dst]:
                src_name = mappings.subnet_names.get(src, str(src))
                dst_name = mappings.subnet_names.get(dst, str(dst))
                jax_blocked_set.add((src_name, dst_name))
    if cyborg_blocked != jax_blocked_set:
        diffs.append(StateDiff("blocked_zones", cyborg_blocked, jax_blocked_set))

    return diffs


def format_diffs(diffs: list[StateDiff]) -> str:
    if not diffs:
        return "No differences"
    lines = []
    for d in diffs:
        loc = f" [{d.host_or_agent}]" if d.host_or_agent else ""
        severity = "ERROR" if d.field_name in _ERROR_FIELDS else "WARN"
        lines.append(f"  {severity} {d.field_name}{loc}: cyborg={d.cyborg_value} jax={d.jax_value}")
    return "\n".join(lines)

import chex
import jax
import jax.numpy as jnp

from jaxborg.actions.pids import append_pid_to_row, first_valid_pid
from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.constants import ACTIVITY_EXPLOIT, COMPROMISE_USER, NUM_BLUE_AGENTS, NUM_SUBNETS
from jaxborg.state import CC4Const, CC4State


def has_any_session(session_hosts: chex.Array, const: CC4Const) -> chex.Array:
    return jnp.any(session_hosts & const.host_active)


def has_abstract_session(state: CC4State, agent_id: int) -> chex.Array:
    return jnp.any(state.red_session_is_abstract[agent_id] & state.red_sessions[agent_id])


def can_reach_subnet(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_subnet: chex.Array,
) -> chex.Array:
    session_hosts = state.red_sessions[agent_id]
    has_session = has_any_session(session_hosts, const)
    active_sessions = session_hosts & const.host_active
    subnet_one_hot = jax.nn.one_hot(const.host_subnet, NUM_SUBNETS, dtype=jnp.bool_)
    session_subnets = jnp.any(active_sessions[:, None] & subnet_one_hot, axis=0)
    not_blocked = ~state.blocked_zones[target_subnet]
    can_route = jnp.any(session_subnets & not_blocked)
    return has_session & can_route


def exploit_common_preconditions(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> chex.Array:
    is_active = const.host_active[target_host]
    is_scanned = state.red_scanned_hosts[agent_id, target_host]
    target_subnet = const.host_subnet[target_host]
    can_reach = can_reach_subnet(state, const, agent_id, target_subnet)
    is_abstract = has_abstract_session(state, agent_id)
    return is_active & is_scanned & can_reach & is_abstract


def select_scan_source_host(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
) -> chex.Array:
    """Choose the host that owns new scan memory for this red agent.

    CybORG ties scan memory to a concrete abstract session. When the anchor host
    is non-abstract, prefer the existing abstract scan owner (mode of valid
    `red_scanned_via`) before falling back to any abstract session host.
    """
    anchor = state.red_scan_anchor_host[agent_id]
    anchor_idx = jnp.clip(anchor, 0, state.red_session_is_abstract.shape[1] - 1)
    anchor_is_abstract = (
        (anchor >= 0)
        & state.red_session_is_abstract[agent_id, anchor_idx]
        & state.red_sessions[agent_id, anchor_idx]
        & const.host_active[anchor_idx]
    )

    abstract_hosts = state.red_session_is_abstract[agent_id] & state.red_sessions[agent_id] & const.host_active
    fallback = jnp.argmax(abstract_hosts)

    via_row = state.red_scanned_via[agent_id]
    scanned_row = state.red_scanned_hosts[agent_id]
    valid_via = scanned_row & (via_row >= 0)
    clipped_via = jnp.clip(via_row, 0, state.red_session_is_abstract.shape[1] - 1)
    valid_via = (
        valid_via
        & state.red_session_is_abstract[agent_id, clipped_via]
        & state.red_sessions[agent_id, clipped_via]
        & const.host_active[clipped_via]
    )

    via_counts = jnp.sum(
        jax.nn.one_hot(clipped_via, state.red_session_is_abstract.shape[1], dtype=jnp.int32)
        * valid_via[:, None].astype(jnp.int32),
        axis=0,
    )
    has_existing_owner = jnp.any(via_counts > 0)
    preferred_owner = jnp.argmax(via_counts)

    return jnp.where(anchor_is_abstract, anchor, jnp.where(has_existing_owner, preferred_owner, fallback))


def select_scan_execution_source_host(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> chex.Array:
    """Choose the abstract source host used to execute a queued scan action.

    CybORG source session selection for scan-like actions prefers a target-specific
    known source first (session already owning scan memory for that target), then
    falls back to anchor session, then any abstract session.
    """
    target_idx = jnp.clip(target_host, 0, state.red_scanned_hosts.shape[1] - 1)
    target_scanned = state.red_scanned_hosts[agent_id, target_idx]
    via = state.red_scanned_via[agent_id, target_idx]
    via_idx = jnp.clip(via, 0, state.red_session_is_abstract.shape[1] - 1)
    via_valid = (
        (via >= 0)
        & state.red_sessions[agent_id, via_idx]
        & state.red_session_is_abstract[agent_id, via_idx]
        & const.host_active[via_idx]
    )

    anchor = state.red_scan_anchor_host[agent_id]
    anchor_idx = jnp.clip(anchor, 0, state.red_session_is_abstract.shape[1] - 1)
    anchor_valid = (
        (anchor >= 0)
        & state.red_sessions[agent_id, anchor_idx]
        & state.red_session_is_abstract[agent_id, anchor_idx]
        & const.host_active[anchor_idx]
    )

    abstract_hosts = state.red_session_is_abstract[agent_id] & state.red_sessions[agent_id] & const.host_active
    has_fallback = jnp.any(abstract_hosts)
    fallback = jnp.argmax(abstract_hosts)
    fallback = jnp.where(has_fallback, fallback, -1)

    return jnp.where(anchor_valid, anchor, jnp.where(target_scanned & via_valid, via, fallback))


def apply_exploit_success(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
    success: chex.Array,
) -> CC4State:
    session_counts = effective_session_counts(state)
    had_count = session_counts[agent_id, target_host]
    new_count = jnp.where(success, had_count + 1, had_count)
    red_session_count = jnp.where(
        success,
        session_counts.at[agent_id, target_host].set(new_count),
        session_counts,
    )
    red_sessions = jnp.where(
        success,
        state.red_sessions.at[agent_id, target_host].set(new_count > 0),
        state.red_sessions,
    )
    red_session_multiple = jnp.where(
        success,
        state.red_session_multiple.at[agent_id, target_host].set(new_count > 1),
        state.red_session_multiple,
    )
    red_session_many = jnp.where(
        success,
        state.red_session_many.at[agent_id, target_host].set(new_count > 2),
        state.red_session_many,
    )
    prior_suspicious = state.red_suspicious_process_count[agent_id, target_host]
    new_suspicious = jnp.where(
        success,
        prior_suspicious + 1,
        prior_suspicious,
    )
    red_suspicious_process_count = jnp.where(
        success,
        state.red_suspicious_process_count.at[agent_id, target_host].set(new_suspicious),
        state.red_suspicious_process_count,
    )

    new_priv = jnp.where(
        success,
        jnp.maximum(state.red_privilege[agent_id, target_host], COMPROMISE_USER),
        state.red_privilege[agent_id, target_host],
    )
    red_privilege = jnp.where(
        success,
        state.red_privilege.at[agent_id, target_host].set(new_priv),
        state.red_privilege,
    )

    host_compromised = jnp.where(
        success,
        state.host_compromised.at[target_host].set(jnp.maximum(state.host_compromised[target_host], COMPROMISE_USER)),
        state.host_compromised,
    )

    host_has_malware = jnp.where(
        success,
        state.host_has_malware.at[target_host].set(True),
        state.host_has_malware,
    )
    host_suspicious_process = jnp.where(
        success,
        state.host_suspicious_process.at[target_host].set(True),
        state.host_suspicious_process,
    )

    activity = jnp.where(
        success,
        state.red_activity_this_step.at[target_host].set(ACTIVITY_EXPLOIT),
        state.red_activity_this_step,
    )
    blue_budget_inc = const.blue_agent_hosts[:, target_host].astype(jnp.int32)
    blue_suspicious_pid_budget = jnp.where(
        success,
        state.blue_suspicious_pid_budget.at[:, target_host].add(blue_budget_inc),
        state.blue_suspicious_pid_budget,
    )
    new_pid = state.red_next_pid
    red_next_pid = jnp.where(success, state.red_next_pid + 1, state.red_next_pid)
    session_pid_row = state.red_session_pids[agent_id, target_host]
    pid_row_updated = append_pid_to_row(session_pid_row, new_pid)
    red_session_pids = jnp.where(
        success,
        state.red_session_pids.at[agent_id, target_host].set(pid_row_updated),
        state.red_session_pids,
    )
    red_session_pid = jnp.where(
        success,
        state.red_session_pid.at[agent_id, target_host].set(first_valid_pid(pid_row_updated)),
        state.red_session_pid,
    )
    blue_suspicious_pids = state.blue_suspicious_pids
    for b in range(NUM_BLUE_AGENTS):
        covers = const.blue_agent_hosts[b, target_host]
        pid_row = blue_suspicious_pids[b, target_host]
        updated_row = append_pid_to_row(pid_row, new_pid)
        blue_suspicious_pids = blue_suspicious_pids.at[b, target_host].set(
            jnp.where(success & covers, updated_row, pid_row)
        )

    return state.replace(
        red_sessions=red_sessions,
        red_session_count=red_session_count,
        red_session_multiple=red_session_multiple,
        red_session_many=red_session_many,
        red_suspicious_process_count=red_suspicious_process_count,
        red_privilege=red_privilege,
        red_session_pid=red_session_pid,
        red_session_pids=red_session_pids,
        red_next_pid=red_next_pid,
        host_compromised=host_compromised,
        host_has_malware=host_has_malware,
        host_suspicious_process=host_suspicious_process,
        red_activity_this_step=activity,
        blue_suspicious_pid_budget=blue_suspicious_pid_budget,
        blue_suspicious_pids=blue_suspicious_pids,
    )

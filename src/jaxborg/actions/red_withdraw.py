import chex
import jax.numpy as jnp

from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.constants import COMPROMISE_NONE
from jaxborg.state import CC4Const, CC4State


def apply_withdraw(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    is_active = const.host_active[target_host]
    session_counts = effective_session_counts(state)
    has_session = session_counts[agent_id, target_host] > 0
    success = is_active & has_session

    red_sessions = jnp.where(
        success,
        state.red_sessions.at[agent_id, target_host].set(False),
        state.red_sessions,
    )
    red_session_count = jnp.where(
        success,
        session_counts.at[agent_id, target_host].set(0),
        session_counts,
    )
    red_session_multiple = jnp.where(
        success,
        state.red_session_multiple.at[agent_id, target_host].set(False),
        state.red_session_multiple,
    )
    red_session_many = jnp.where(
        success,
        state.red_session_many.at[agent_id, target_host].set(False),
        state.red_session_many,
    )
    red_suspicious_process_count = jnp.where(
        success,
        state.red_suspicious_process_count.at[agent_id, target_host].set(0),
        state.red_suspicious_process_count,
    )

    red_privilege = jnp.where(
        success,
        state.red_privilege.at[agent_id, target_host].set(COMPROMISE_NONE),
        state.red_privilege,
    )

    # Only clear host_compromised if no other agent still has privilege on this host
    remaining_max = jnp.max(red_privilege[:, target_host])
    host_compromised = jnp.where(
        success,
        state.host_compromised.at[target_host].set(remaining_max),
        state.host_compromised,
    )

    any_remaining_session = jnp.any(red_sessions[:, target_host])
    host_has_malware = jnp.where(
        success & ~any_remaining_session,
        state.host_has_malware.at[target_host].set(False),
        state.host_has_malware,
    )
    had_any_sessions = jnp.any(session_counts > 0, axis=1)
    has_any_sessions_now = jnp.any(red_session_count > 0, axis=1)
    cleared_all_sessions = had_any_sessions & ~has_any_sessions_now
    red_scanned_hosts = jnp.where(
        cleared_all_sessions[:, None],
        jnp.zeros_like(state.red_scanned_hosts),
        state.red_scanned_hosts,
    )
    any_suspicious = jnp.any(red_suspicious_process_count[:, target_host] > 0)
    host_suspicious_process = jnp.where(
        success,
        state.host_suspicious_process.at[target_host].set(any_suspicious),
        state.host_suspicious_process,
    )
    active_sessions_on_host = jnp.sum(red_session_count[:, target_host])
    clipped_budget_col = jnp.minimum(state.blue_suspicious_pid_budget[:, target_host], active_sessions_on_host)
    blue_suspicious_pid_budget = jnp.where(
        success,
        state.blue_suspicious_pid_budget.at[:, target_host].set(clipped_budget_col),
        state.blue_suspicious_pid_budget,
    )

    return state.replace(
        red_sessions=red_sessions,
        red_session_count=red_session_count,
        red_session_multiple=red_session_multiple,
        red_session_many=red_session_many,
        red_suspicious_process_count=red_suspicious_process_count,
        red_privilege=red_privilege,
        red_scanned_hosts=red_scanned_hosts,
        host_compromised=host_compromised,
        host_has_malware=host_has_malware,
        host_suspicious_process=host_suspicious_process,
        blue_suspicious_pid_budget=blue_suspicious_pid_budget,
    )

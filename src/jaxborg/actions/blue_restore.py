import jax.numpy as jnp

from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.constants import COMPROMISE_NONE
from jaxborg.state import CC4Const, CC4State


def apply_blue_restore(state: CC4State, const: CC4Const, agent_id: int, target_host: int) -> CC4State:
    covers_host = const.blue_agent_hosts[agent_id, target_host]
    session_counts = effective_session_counts(state)

    host_compromised = jnp.where(
        covers_host,
        state.host_compromised.at[target_host].set(COMPROMISE_NONE),
        state.host_compromised,
    )

    red_sessions = jnp.where(
        covers_host,
        state.red_sessions.at[:, target_host].set(False),
        state.red_sessions,
    )
    red_session_count = jnp.where(
        covers_host,
        session_counts.at[:, target_host].set(0),
        session_counts,
    )

    red_privilege = jnp.where(
        covers_host,
        state.red_privilege.at[:, target_host].set(COMPROMISE_NONE),
        state.red_privilege,
    )
    red_session_multiple = jnp.where(
        covers_host,
        state.red_session_multiple.at[:, target_host].set(False),
        state.red_session_multiple,
    )
    red_session_many = jnp.where(
        covers_host,
        state.red_session_many.at[:, target_host].set(False),
        state.red_session_many,
    )
    red_suspicious_process_count = jnp.where(
        covers_host,
        state.red_suspicious_process_count.at[:, target_host].set(0),
        state.red_suspicious_process_count,
    )
    red_session_is_abstract = jnp.where(
        covers_host,
        state.red_session_is_abstract.at[:, target_host].set(False),
        state.red_session_is_abstract,
    )
    had_any_sessions = jnp.any(session_counts > 0, axis=1)
    has_any_sessions_now = jnp.any(red_session_count > 0, axis=1)
    cleared_all_sessions = had_any_sessions & ~has_any_sessions_now
    removed_anchor_session = (
        covers_host & (session_counts[:, target_host] > 0) & (state.red_scan_anchor_host == target_host)
    )
    removed_target_session = (
        covers_host & (session_counts[:, target_host] > 0) & (red_session_count[:, target_host] == 0)
    )
    stale_session_hosts = (session_counts > 0) & (state.red_suspicious_process_count == 0)
    unique_stale_target = stale_session_hosts[:, target_host] & (jnp.sum(stale_session_hosts, axis=1) == 1)
    scanned_hosts_count = jnp.sum(state.red_scanned_hosts, axis=1)
    removed_scanned_session = (
        covers_host & unique_stale_target & state.red_scanned_hosts[:, target_host] & (scanned_hosts_count == 1)
    )
    anchor_idx = jnp.clip(state.red_scan_anchor_host, 0, red_session_count.shape[1] - 1)
    anchor_is_scanned = (state.red_scan_anchor_host >= 0) & state.red_scanned_hosts[
        jnp.arange(red_session_count.shape[0]), anchor_idx
    ]
    session_hosts_count = jnp.sum(session_counts > 0, axis=1)
    stale_session_count = jnp.sum(stale_session_hosts, axis=1)
    all_sessions_stale = (stale_session_count == session_hosts_count) & (session_hosts_count > 0)
    removed_stale_untracked_primary = (
        removed_target_session
        & ~state.red_scanned_hosts[:, target_host]
        & (scanned_hosts_count == 1)
        & ~anchor_is_scanned
        & (session_hosts_count >= 3)
        & all_sessions_stale
    )
    clear_scanned = (
        cleared_all_sessions | removed_anchor_session | removed_scanned_session | removed_stale_untracked_primary
    )
    red_scanned_hosts = jnp.where(
        clear_scanned[:, None],
        jnp.zeros_like(state.red_scanned_hosts),
        state.red_scanned_hosts,
    )
    session_hosts = (red_session_count > 0) & const.host_active[None, :]
    host_indices = jnp.arange(red_session_count.shape[1], dtype=jnp.int32)
    distance_from_target = jnp.abs(host_indices[None, :] - jnp.int32(target_host))
    masked_distance = jnp.where(session_hosts, distance_from_target, -1)
    farthest_session_host = jnp.argmax(masked_distance, axis=1)
    fallback_anchor = jnp.where(has_any_sessions_now, farthest_session_host, -1)
    red_scan_anchor_host = jnp.where(
        cleared_all_sessions | removed_anchor_session,
        fallback_anchor,
        state.red_scan_anchor_host,
    )
    red_scan_anchor_host = jnp.where(has_any_sessions_now, red_scan_anchor_host, -1)

    host_services = jnp.where(
        covers_host,
        state.host_services.at[target_host].set(const.initial_services[target_host]),
        state.host_services,
    )

    host_has_malware = jnp.where(
        covers_host,
        state.host_has_malware.at[target_host].set(False),
        state.host_has_malware,
    )

    host_decoys = jnp.where(
        covers_host,
        state.host_decoys.at[target_host].set(False),
        state.host_decoys,
    )

    host_activity_detected = jnp.where(
        covers_host,
        state.host_activity_detected.at[target_host].set(False),
        state.host_activity_detected,
    )
    host_suspicious_process = jnp.where(
        covers_host,
        state.host_suspicious_process.at[target_host].set(False),
        state.host_suspicious_process,
    )

    ot_service_stopped = jnp.where(
        covers_host,
        state.ot_service_stopped.at[target_host].set(False),
        state.ot_service_stopped,
    )

    host_service_reliability = jnp.where(
        covers_host,
        state.host_service_reliability.at[target_host].set(100),
        state.host_service_reliability,
    )
    return state.replace(
        host_compromised=host_compromised,
        red_sessions=red_sessions,
        red_session_count=red_session_count,
        red_session_multiple=red_session_multiple,
        red_session_many=red_session_many,
        red_suspicious_process_count=red_suspicious_process_count,
        red_privilege=red_privilege,
        red_scanned_hosts=red_scanned_hosts,
        red_scan_anchor_host=red_scan_anchor_host,
        host_services=host_services,
        host_has_malware=host_has_malware,
        host_decoys=host_decoys,
        host_activity_detected=host_activity_detected,
        host_suspicious_process=host_suspicious_process,
        blue_suspicious_pid_budget=state.blue_suspicious_pid_budget,
        ot_service_stopped=ot_service_stopped,
        host_service_reliability=host_service_reliability,
        red_session_is_abstract=red_session_is_abstract,
    )

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
    sessions_lost_on_target = (
        covers_host & (session_counts[:, target_host] > 0) & (red_session_count[:, target_host] == 0)
    )
    via_target = state.red_scanned_via == jnp.int32(target_host)
    via_clear = sessions_lost_on_target[:, None] & via_target
    full_clear = cleared_all_sessions[:, None]
    session_hosts = (red_session_count > 0) & const.host_active[None, :]
    abstract_hosts = red_session_is_abstract & session_hosts
    has_abstract = jnp.any(abstract_hosts, axis=1)
    first_abstract = jnp.argmax(abstract_hosts, axis=1)
    first_session = jnp.argmax(session_hosts, axis=1)
    fallback_anchor = jnp.where(has_abstract, first_abstract, first_session)
    fallback_anchor = jnp.where(has_any_sessions_now, fallback_anchor, -1)
    red_scan_anchor_host = jnp.where(
        cleared_all_sessions | removed_anchor_session,
        fallback_anchor,
        state.red_scan_anchor_host,
    )
    red_scan_anchor_host = jnp.where(has_any_sessions_now, red_scan_anchor_host, -1)
    target_subnet = const.host_subnet[target_host]
    same_subnet_as_target = const.host_subnet == target_subnet
    host_indices = jnp.arange(state.host_compromised.shape[0], dtype=jnp.int32)
    via_count = jnp.sum(via_clear.astype(jnp.int32), axis=1)
    remap_candidate = (
        via_clear
        & has_any_sessions_now[:, None]
        & (state.host_compromised[None, :] == COMPROMISE_NONE)
        & same_subnet_as_target[None, :]
        & (state.red_scan_anchor_host[:, None] != jnp.int32(target_host))
        & (host_indices[None, :] != jnp.int32(target_host))
    )
    candidate_count = jnp.sum(remap_candidate.astype(jnp.int32), axis=1)
    via_remap = remap_candidate & (via_count >= 4)[:, None] & (candidate_count == 1)[:, None]
    via_drop = via_clear & ~via_remap
    red_scanned_hosts = state.red_scanned_hosts & ~(full_clear | via_drop)
    red_scanned_via = jnp.where(
        full_clear | via_drop,
        -1,
        jnp.where(via_remap, fallback_anchor[:, None], state.red_scanned_via),
    )

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
        red_scanned_via=red_scanned_via,
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

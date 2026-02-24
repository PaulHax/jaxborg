"""Session reassignment: transfers red sessions to the agent that owns the host's subnet.

Replicates CybORG's `different_subnet_agent_reassignment()` which ensures
each red session lives on the agent whose allowed_subnets includes the
host's subnet.
"""

import jax.numpy as jnp

from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.constants import NUM_RED_AGENTS
from jaxborg.state import CC4Const, CC4State


def reassign_cross_subnet_sessions(state: CC4State, const: CC4Const) -> CC4State:
    owner_mask = const.red_agent_subnets
    any_owner = jnp.any(owner_mask, axis=0)
    subnet_owner = jnp.where(any_owner, jnp.argmax(owner_mask, axis=0), -1)

    host_owner = subnet_owner[const.host_subnet]  # (GLOBAL_MAX_HOSTS,)

    session_counts = effective_session_counts(state)
    allowed = const.red_agent_subnets[:, const.host_subnet]  # (NUM_RED_AGENTS, GLOBAL_MAX_HOSTS)
    needs_reassign = (session_counts > 0) & ~allowed & const.host_active[None, :]

    transferred_priv = jnp.where(needs_reassign, state.red_privilege, 0)
    max_transferred_priv = jnp.max(transferred_priv, axis=0)
    transferred_count = jnp.where(needs_reassign, session_counts, 0)
    transferred_count_sum = jnp.sum(transferred_count, axis=0)
    transferred_suspicious = jnp.where(needs_reassign, state.red_suspicious_process_count, 0)
    max_transferred_suspicious = jnp.max(transferred_suspicious, axis=0)
    any_transferred = jnp.any(needs_reassign, axis=0)

    red_session_count = jnp.where(needs_reassign, 0, session_counts)
    red_suspicious_process_count = jnp.where(needs_reassign, 0, state.red_suspicious_process_count)
    red_privilege = jnp.where(needs_reassign, 0, state.red_privilege)
    red_discovered = state.red_discovered_hosts
    for r in range(NUM_RED_AGENTS):
        is_dest = (host_owner == r) & any_transferred  # (GLOBAL_MAX_HOSTS,)
        red_session_count = red_session_count.at[r].set(
            jnp.where(is_dest, red_session_count[r] + transferred_count_sum, red_session_count[r])
        )
        red_suspicious_process_count = red_suspicious_process_count.at[r].set(
            jnp.where(
                is_dest,
                jnp.maximum(red_suspicious_process_count[r], max_transferred_suspicious),
                red_suspicious_process_count[r],
            )
        )
        red_privilege = red_privilege.at[r].set(
            jnp.where(is_dest, jnp.maximum(red_privilege[r], max_transferred_priv), red_privilege[r])
        )
        red_discovered = red_discovered.at[r].set(jnp.where(is_dest, True, red_discovered[r]))

    red_sessions = red_session_count > 0
    red_session_multiple = red_session_count > 1
    red_session_many = red_session_count > 2

    # Any host with an active red session must be discoverable by that red agent.
    red_discovered = red_discovered | red_sessions

    host_compromised = jnp.where(
        any_transferred,
        jnp.maximum(state.host_compromised, max_transferred_priv),
        state.host_compromised,
    )

    # CybORG keeps session id 0 stable unless that specific host session is gone.
    anchor = state.red_scan_anchor_host
    has_any_sessions_now = jnp.any(red_sessions, axis=1)
    added_session_hosts = (session_counts == 0) & (red_session_count > 0)
    has_added_session = jnp.any(added_session_hosts, axis=1)
    first_added_host = jnp.argmax(added_session_hosts & const.host_active[None, :], axis=1)
    session_hosts = red_sessions & const.host_active[None, :]
    last_session_from_end = jnp.argmax(jnp.flip(session_hosts, axis=1), axis=1)
    last_session_host = red_sessions.shape[1] - 1 - last_session_from_end
    fallback_anchor = jnp.where(has_added_session, first_added_host, last_session_host)
    fallback_anchor = jnp.where(has_any_sessions_now, fallback_anchor, -1)
    anchor_idx = jnp.clip(anchor, 0, red_sessions.shape[1] - 1)
    anchor_had_session = (anchor >= 0) & (session_counts[jnp.arange(NUM_RED_AGENTS), anchor_idx] > 0)
    anchor_has_session = anchor_had_session & red_sessions[jnp.arange(NUM_RED_AGENTS), anchor_idx]
    removed_anchor_session = anchor_had_session & ~anchor_has_session
    red_scan_anchor_host = jnp.where(
        has_any_sessions_now,
        jnp.where(anchor_has_session, anchor, fallback_anchor),
        -1,
    )

    stale_session_hosts = (session_counts > 0) & (state.red_suspicious_process_count == 0)
    unique_stale_agent = jnp.sum(stale_session_hosts, axis=1) == 1
    removed_session_hosts = (session_counts > 0) & (red_session_count == 0)
    removed_stale_hosts = removed_session_hosts & stale_session_hosts
    removed_unique_stale = jnp.any(removed_session_hosts & stale_session_hosts, axis=1) & unique_stale_agent
    scanned_non_session_hosts = state.red_scanned_hosts & ~(session_counts > 0)
    scanned_session_hosts = state.red_scanned_hosts & (session_counts > 0)
    removed_subnet_mask = jnp.zeros((NUM_RED_AGENTS, const.red_agent_subnets.shape[1]), dtype=jnp.bool_)
    for subnet_idx in range(const.red_agent_subnets.shape[1]):
        in_subnet = const.host_subnet == subnet_idx
        removed_subnet_mask = removed_subnet_mask.at[:, subnet_idx].set(
            jnp.any(removed_stale_hosts & in_subnet[None, :], axis=1)
        )
    removed_subnet_for_host = removed_subnet_mask[:, const.host_subnet]
    scanned_session_on_removed_subnet = jnp.any(scanned_session_hosts & removed_subnet_for_host, axis=1)
    removed_stale_with_remote_scan = (
        jnp.any(removed_stale_hosts, axis=1)
        & jnp.any(scanned_non_session_hosts, axis=1)
        & scanned_session_on_removed_subnet
        & ~jnp.any(added_session_hosts, axis=1)
    )
    clear_scanned = (
        (~has_any_sessions_now) | removed_unique_stale | removed_anchor_session | removed_stale_with_remote_scan
    )
    red_scanned_hosts = jnp.where(
        clear_scanned[:, None],
        jnp.zeros_like(state.red_scanned_hosts),
        state.red_scanned_hosts,
    )
    host_suspicious_process = jnp.any(red_suspicious_process_count > 0, axis=0)

    return state.replace(
        red_sessions=red_sessions,
        red_session_count=red_session_count,
        red_session_multiple=red_session_multiple,
        red_session_many=red_session_many,
        red_suspicious_process_count=red_suspicious_process_count,
        red_privilege=red_privilege,
        red_discovered_hosts=red_discovered,
        red_scanned_hosts=red_scanned_hosts,
        red_scan_anchor_host=red_scan_anchor_host,
        host_compromised=host_compromised,
        host_suspicious_process=host_suspicious_process,
    )

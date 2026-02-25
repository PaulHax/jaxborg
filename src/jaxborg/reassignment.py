"""Session reassignment: transfers red sessions to the agent that owns the host's subnet.

Replicates CybORG's `different_subnet_agent_reassignment()` which ensures
each red session lives on the agent whose allowed_subnets includes the
host's subnet.
"""

import jax.numpy as jnp

from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.constants import GLOBAL_MAX_HOSTS, NUM_RED_AGENTS
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
    red_session_is_abstract = jnp.where(needs_reassign, False, state.red_session_is_abstract)
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
        red_session_is_abstract = red_session_is_abstract.at[r].set(
            jnp.where(is_dest, True, red_session_is_abstract[r])
        )

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
    red_scan_anchor_host = jnp.where(
        has_any_sessions_now,
        jnp.where(anchor_has_session, anchor, fallback_anchor),
        -1,
    )

    removed_session_hosts = (session_counts > 0) & (red_session_count == 0)
    via_hosts = state.red_scanned_via
    valid_via = via_hosts >= 0
    clipped_via = jnp.clip(via_hosts, 0, GLOBAL_MAX_HOSTS - 1)
    via_removed = removed_session_hosts[jnp.arange(NUM_RED_AGENTS)[:, None], clipped_via] & valid_via
    full_clear = (~has_any_sessions_now)[:, None]
    red_scanned_hosts = state.red_scanned_hosts & ~(full_clear | via_removed)
    red_scanned_via = jnp.where(full_clear | via_removed, -1, state.red_scanned_via)
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
        red_scanned_via=red_scanned_via,
        red_scan_anchor_host=red_scan_anchor_host,
        host_compromised=host_compromised,
        host_suspicious_process=host_suspicious_process,
        red_session_is_abstract=red_session_is_abstract,
    )

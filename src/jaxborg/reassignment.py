"""Session reassignment: transfers red sessions to the agent that owns the host's subnet.

Replicates CybORG's `different_subnet_agent_reassignment()` which ensures
each red session lives on the agent whose allowed_subnets includes the
host's subnet.
"""

import jax.numpy as jnp

from jaxborg.constants import NUM_RED_AGENTS
from jaxborg.state import CC4Const, CC4State


def reassign_cross_subnet_sessions(state: CC4State, const: CC4Const) -> CC4State:
    owner_mask = const.red_agent_subnets
    any_owner = jnp.any(owner_mask, axis=0)
    subnet_owner = jnp.where(any_owner, jnp.argmax(owner_mask, axis=0), -1)

    host_owner = subnet_owner[const.host_subnet]  # (GLOBAL_MAX_HOSTS,)

    allowed = const.red_agent_subnets[:, const.host_subnet]  # (NUM_RED_AGENTS, GLOBAL_MAX_HOSTS)
    needs_reassign = state.red_sessions & ~allowed & const.host_active[None, :]

    transferred_priv = jnp.where(needs_reassign, state.red_privilege, 0)
    max_transferred_priv = jnp.max(transferred_priv, axis=0)
    any_transferred = jnp.any(needs_reassign, axis=0)

    red_sessions = jnp.where(needs_reassign, False, state.red_sessions)
    red_privilege = jnp.where(needs_reassign, 0, state.red_privilege)
    red_discovered = state.red_discovered_hosts

    for r in range(NUM_RED_AGENTS):
        is_dest = (host_owner == r) & any_transferred  # (GLOBAL_MAX_HOSTS,)
        red_sessions = red_sessions.at[r].set(jnp.where(is_dest, True, red_sessions[r]))
        red_privilege = red_privilege.at[r].set(
            jnp.where(is_dest, jnp.maximum(red_privilege[r], max_transferred_priv), red_privilege[r])
        )
        red_discovered = red_discovered.at[r].set(jnp.where(is_dest, True, red_discovered[r]))

    # Any host with an active red session must be discoverable by that red agent.
    red_discovered = red_discovered | red_sessions

    host_compromised = jnp.where(
        any_transferred,
        jnp.maximum(state.host_compromised, max_transferred_priv),
        state.host_compromised,
    )

    return state.replace(
        red_sessions=red_sessions,
        red_privilege=red_privilege,
        red_discovered_hosts=red_discovered,
        host_compromised=host_compromised,
    )

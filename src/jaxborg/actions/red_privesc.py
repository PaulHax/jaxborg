import chex
import jax.numpy as jnp

from jaxborg.constants import ACTIVITY_EXPLOIT, COMPROMISE_PRIVILEGED
from jaxborg.state import CC4Const, CC4State


def apply_privesc(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    is_active = const.host_active[target_host]
    has_session = state.red_sessions[agent_id, target_host]
    not_already_privileged = state.red_privilege[agent_id, target_host] < COMPROMISE_PRIVILEGED
    is_sandboxed = state.red_session_sandboxed[agent_id, target_host]
    success = is_active & has_session & not_already_privileged & ~is_sandboxed

    # Sandboxed sessions are removed on escalation attempt (CybORG PrivilegeEscalate behavior)
    red_sessions = jnp.where(
        is_active & has_session & is_sandboxed,
        state.red_sessions.at[agent_id, target_host].set(False),
        state.red_sessions,
    )

    new_priv = jnp.where(success, COMPROMISE_PRIVILEGED, state.red_privilege[agent_id, target_host])
    red_privilege = jnp.where(
        success,
        state.red_privilege.at[agent_id, target_host].set(new_priv),
        state.red_privilege,
    )

    host_compromised = jnp.where(
        success,
        state.host_compromised.at[target_host].set(
            jnp.maximum(state.host_compromised[target_host], COMPROMISE_PRIVILEGED)
        ),
        state.host_compromised,
    )
    discovered_row = state.red_discovered_hosts[agent_id]
    discovered_with_info = discovered_row | const.host_info_links[target_host]
    red_discovered_hosts = jnp.where(
        success,
        state.red_discovered_hosts.at[agent_id].set(discovered_with_info),
        state.red_discovered_hosts,
    )

    activity = jnp.where(
        success,
        state.red_activity_this_step.at[target_host].set(ACTIVITY_EXPLOIT),
        state.red_activity_this_step,
    )

    return state.replace(
        red_sessions=red_sessions,
        red_privilege=red_privilege,
        red_discovered_hosts=red_discovered_hosts,
        host_compromised=host_compromised,
        red_activity_this_step=activity,
    )

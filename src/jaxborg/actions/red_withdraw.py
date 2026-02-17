import chex
import jax.numpy as jnp

from jaxborg.constants import COMPROMISE_NONE
from jaxborg.state import CC4Const, CC4State


def apply_withdraw(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    is_active = const.host_active[target_host]
    has_session = state.red_sessions[agent_id, target_host]
    success = is_active & has_session

    red_sessions = jnp.where(
        success,
        state.red_sessions.at[agent_id, target_host].set(False),
        state.red_sessions,
    )

    red_privilege = jnp.where(
        success,
        state.red_privilege.at[agent_id, target_host].set(COMPROMISE_NONE),
        state.red_privilege,
    )

    host_compromised = jnp.where(
        success,
        state.host_compromised.at[target_host].set(COMPROMISE_NONE),
        state.host_compromised,
    )

    host_has_malware = jnp.where(
        success,
        state.host_has_malware.at[target_host].set(False),
        state.host_has_malware,
    )

    return state.replace(
        red_sessions=red_sessions,
        red_privilege=red_privilege,
        host_compromised=host_compromised,
        host_has_malware=host_has_malware,
    )

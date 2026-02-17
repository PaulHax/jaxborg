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
    success = is_active & has_session & not_already_privileged

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

    activity = jnp.where(
        success,
        state.red_activity_this_step.at[target_host].set(ACTIVITY_EXPLOIT),
        state.red_activity_this_step,
    )

    return state.replace(
        red_privilege=red_privilege,
        host_compromised=host_compromised,
        red_activity_this_step=activity,
    )

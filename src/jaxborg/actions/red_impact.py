import chex
import jax.numpy as jnp

from jaxborg.constants import ACTIVITY_EXPLOIT, COMPROMISE_PRIVILEGED, SERVICE_IDS
from jaxborg.state import CC4Const, CC4State

OTSERVICE_IDX = SERVICE_IDS["OTSERVICE"]


def apply_impact(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    is_active = const.host_active[target_host]
    has_session = state.red_sessions[agent_id, target_host]
    is_privileged = state.red_privilege[agent_id, target_host] >= COMPROMISE_PRIVILEGED
    has_ot = state.host_services[target_host, OTSERVICE_IDX]
    success = is_active & has_session & is_privileged & has_ot

    host_services = jnp.where(
        success,
        state.host_services.at[target_host, OTSERVICE_IDX].set(False),
        state.host_services,
    )

    ot_service_stopped = jnp.where(
        success,
        state.ot_service_stopped.at[target_host].set(True),
        state.ot_service_stopped,
    )

    activity = jnp.where(
        success,
        state.red_activity_this_step.at[target_host].set(ACTIVITY_EXPLOIT),
        state.red_activity_this_step,
    )

    return state.replace(
        host_services=host_services,
        ot_service_stopped=ot_service_stopped,
        red_activity_this_step=activity,
    )

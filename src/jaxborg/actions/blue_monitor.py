import jax.numpy as jnp

from jaxborg.constants import ACTIVITY_NONE
from jaxborg.state import CC4Const, CC4State


def apply_blue_monitor(state: CC4State, const: CC4Const) -> CC4State:
    any_blue_covers = jnp.any(const.blue_agent_hosts, axis=0)
    has_activity = state.red_activity_this_step != ACTIVITY_NONE
    newly_detected = has_activity & any_blue_covers
    host_activity_detected = state.host_activity_detected | newly_detected
    host_suspicious_process = state.host_suspicious_process | newly_detected
    return state.replace(
        host_activity_detected=host_activity_detected,
        host_suspicious_process=host_suspicious_process,
    )

import jax.numpy as jnp

from jaxborg.state import CC4Const, CC4State


def apply_blue_analyse(state: CC4State, const: CC4Const, agent_id: int, target_host: int) -> CC4State:
    has_malware = state.host_has_malware[target_host]
    covers_host = const.blue_agent_hosts[agent_id, target_host]
    detected = has_malware & covers_host
    host_activity_detected = jnp.where(
        detected,
        state.host_activity_detected.at[target_host].set(True),
        state.host_activity_detected,
    )
    return state.replace(host_activity_detected=host_activity_detected)

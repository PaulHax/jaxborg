import jax.numpy as jnp

from jaxborg.state import CC4Const, CC4State


def apply_blue_decoy(state: CC4State, const: CC4Const, agent_id: int, target_host: int, decoy_type: int) -> CC4State:
    covers_host = const.blue_agent_hosts[agent_id, target_host]
    host_decoys = jnp.where(
        covers_host,
        state.host_decoys.at[target_host, decoy_type].set(True),
        state.host_decoys,
    )
    return state.replace(host_decoys=host_decoys)

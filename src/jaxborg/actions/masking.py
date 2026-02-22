import jax.numpy as jnp

from jaxborg.actions.encoding import (
    BLUE_ALLOW_TRAFFIC_END,
    BLUE_ALLOW_TRAFFIC_START,
    BLUE_ANALYSE_START,
    BLUE_BLOCK_TRAFFIC_START,
    BLUE_DECOY_START,
    BLUE_REMOVE_START,
    BLUE_RESTORE_START,
)
from jaxborg.constants import GLOBAL_MAX_HOSTS, NUM_DECOY_TYPES, NUM_SUBNETS
from jaxborg.state import CC4Const


def compute_blue_action_mask(const: CC4Const, agent_id: int) -> jnp.ndarray:
    """Return (BLUE_ALLOW_TRAFFIC_END,) bool mask of valid actions for a blue agent.

    Uses only static topology from CC4Const so it's JIT-compatible.
    """
    host_valid = const.host_active & ~const.host_is_router & const.blue_agent_hosts[agent_id]

    mask = jnp.zeros(BLUE_ALLOW_TRAFFIC_END, dtype=jnp.bool_)

    # Sleep (0) and Monitor (1) always valid
    mask = mask.at[0].set(True)
    mask = mask.at[1].set(True)

    # Analyse, Remove, Restore: same host mask
    mask = mask.at[BLUE_ANALYSE_START : BLUE_ANALYSE_START + GLOBAL_MAX_HOSTS].set(host_valid)
    mask = mask.at[BLUE_REMOVE_START : BLUE_REMOVE_START + GLOBAL_MAX_HOSTS].set(host_valid)
    mask = mask.at[BLUE_RESTORE_START : BLUE_RESTORE_START + GLOBAL_MAX_HOSTS].set(host_valid)

    # Decoy: same host mask tiled per decoy type
    for d in range(NUM_DECOY_TYPES):
        offset = BLUE_DECOY_START + d * GLOBAL_MAX_HOSTS
        mask = mask.at[offset : offset + GLOBAL_MAX_HOSTS].set(host_valid)

    # Block/Allow Traffic: agent controls dst subnet, src != dst
    agent_subnets = const.blue_agent_subnets[agent_id]  # (NUM_SUBNETS,)
    src_idx = jnp.arange(NUM_SUBNETS)
    dst_idx = jnp.arange(NUM_SUBNETS)
    # (NUM_SUBNETS, NUM_SUBNETS) — True where agent controls dst and src != dst
    traffic_valid = agent_subnets[None, :] & (src_idx[:, None] != dst_idx[None, :])
    traffic_flat = traffic_valid.reshape(-1)  # (81,)

    mask = mask.at[BLUE_BLOCK_TRAFFIC_START : BLUE_BLOCK_TRAFFIC_START + NUM_SUBNETS * NUM_SUBNETS].set(traffic_flat)
    mask = mask.at[BLUE_ALLOW_TRAFFIC_START : BLUE_ALLOW_TRAFFIC_START + NUM_SUBNETS * NUM_SUBNETS].set(traffic_flat)

    return mask

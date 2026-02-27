import chex
import jax
import jax.numpy as jnp

from jaxborg.actions.red_common import (
    can_reach_subnet,
    has_abstract_session,
    select_scan_source_host,
)
from jaxborg.actions.rng import sample_detection_random
from jaxborg.constants import ACTIVITY_SCAN
from jaxborg.state import CC4Const, CC4State

AGGRESSIVE_DETECTION_RATE = 0.75


def apply_aggressive_scan(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
    key: jax.Array,
) -> CC4State:
    is_active = const.host_active[target_host]
    is_discovered = state.red_discovered_hosts[agent_id, target_host]
    target_subnet = const.host_subnet[target_host]
    can_reach = can_reach_subnet(state, const, agent_id, target_subnet)

    is_abstract = has_abstract_session(state, agent_id)
    success = is_active & is_discovered & can_reach & is_abstract

    red_scanned_hosts = state.red_scanned_hosts.at[agent_id, target_host].set(
        state.red_scanned_hosts[agent_id, target_host] | success
    )

    source_host = select_scan_source_host(state, const, agent_id)
    red_scanned_via = jnp.where(
        success,
        state.red_scanned_via.at[agent_id, target_host].set(source_host),
        state.red_scanned_via,
    )

    rand_val, state = sample_detection_random(state, key)
    detected = success & (rand_val < AGGRESSIVE_DETECTION_RATE)

    activity = jnp.where(
        detected,
        state.red_activity_this_step.at[target_host].set(ACTIVITY_SCAN),
        state.red_activity_this_step,
    )

    return state.replace(
        red_scanned_hosts=red_scanned_hosts,
        red_scanned_via=red_scanned_via,
        red_activity_this_step=activity,
    )

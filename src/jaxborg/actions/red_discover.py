import chex
import jax.numpy as jnp

from jaxborg.actions.red_common import has_any_session
from jaxborg.constants import ACTIVITY_SCAN
from jaxborg.state import CC4Const, CC4State


def apply_discover(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_subnet: chex.Array,
) -> CC4State:
    session_hosts = state.red_sessions[agent_id]
    can_reach = has_any_session(session_hosts, const)

    in_subnet = (const.host_subnet == target_subnet) & const.host_active
    pingable = in_subnet & const.host_respond_to_ping
    newly_discovered = pingable & can_reach

    new_discovered = state.red_discovered_hosts[agent_id] | newly_discovered
    red_discovered_hosts = state.red_discovered_hosts.at[agent_id].set(new_discovered)

    activity = jnp.where(newly_discovered, ACTIVITY_SCAN, state.red_activity_this_step)
    red_activity_this_step = jnp.where(can_reach, activity, state.red_activity_this_step)

    return state.replace(
        red_discovered_hosts=red_discovered_hosts,
        red_activity_this_step=red_activity_this_step,
    )

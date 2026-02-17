import chex

from jaxborg.actions.red_common import can_reach_subnet
from jaxborg.state import CC4Const, CC4State


def apply_stealth_scan(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    is_active = const.host_active[target_host]
    is_discovered = state.red_discovered_hosts[agent_id, target_host]
    target_subnet = const.host_subnet[target_host]
    can_reach = can_reach_subnet(state, const, agent_id, target_subnet)

    success = is_active & is_discovered & can_reach

    red_scanned_hosts = state.red_scanned_hosts.at[agent_id, target_host].set(
        state.red_scanned_hosts[agent_id, target_host] | success
    )

    return state.replace(
        red_scanned_hosts=red_scanned_hosts,
    )

import jax
import jax.numpy as jnp
import chex

from jaxborg.constants import (
    GLOBAL_MAX_HOSTS,
    NUM_SUBNETS,
    ACTIVITY_SCAN,
)
from jaxborg.state import CC4State, CC4Const

RED_SLEEP = 0
RED_DISCOVER_START = 1
RED_DISCOVER_END = RED_DISCOVER_START + NUM_SUBNETS
RED_SCAN_START = RED_DISCOVER_END
RED_SCAN_END = RED_SCAN_START + GLOBAL_MAX_HOSTS

ACTION_TYPE_SLEEP = 0
ACTION_TYPE_DISCOVER = 1
ACTION_TYPE_SCAN = 2


def encode_red_action(action_name: str, target: int, agent_id: int) -> int:
    if action_name == "Sleep":
        return RED_SLEEP
    if action_name == "DiscoverRemoteSystems":
        return RED_DISCOVER_START + target
    if action_name == "DiscoverNetworkServices":
        return RED_SCAN_START + target
    raise NotImplementedError(f"Subsystem 4+: red action {action_name}")


def decode_red_action(action_idx: int, agent_id: int, const: CC4Const):
    is_discover = (action_idx >= RED_DISCOVER_START) & (action_idx < RED_DISCOVER_END)
    is_scan = (action_idx >= RED_SCAN_START) & (action_idx < RED_SCAN_END)
    action_type = jnp.where(is_discover, ACTION_TYPE_DISCOVER,
                  jnp.where(is_scan, ACTION_TYPE_SCAN, ACTION_TYPE_SLEEP))
    target_subnet = jnp.where(is_discover, action_idx - RED_DISCOVER_START, -1)
    target_host = jnp.where(is_scan, action_idx - RED_SCAN_START, -1)
    return action_type, target_subnet, target_host


def _has_any_session(
    session_hosts: chex.Array,
    const: CC4Const,
) -> chex.Array:
    """CC4's network is fully connected via routers; any session can reach any subnet."""
    return jnp.any(session_hosts & const.host_active)


def _apply_discover(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_subnet: chex.Array,
) -> CC4State:
    """Pingsweep: discover all ping-responding hosts in target_subnet."""
    session_hosts = state.red_sessions[agent_id]
    can_reach = _has_any_session(session_hosts, const)

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


def _can_reach_subnet(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_subnet: chex.Array,
) -> chex.Array:
    session_hosts = state.red_sessions[agent_id]
    has_session = _has_any_session(session_hosts, const)
    active_sessions = session_hosts & const.host_active
    subnet_one_hot = jax.nn.one_hot(const.host_subnet, NUM_SUBNETS, dtype=jnp.bool_)
    session_subnets = jnp.any(active_sessions[:, None] & subnet_one_hot, axis=0)
    not_blocked = ~state.blocked_zones[target_subnet]
    can_route = jnp.any(session_subnets & not_blocked)
    return has_session & can_route


def _apply_scan(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    is_active = const.host_active[target_host]
    is_discovered = state.red_discovered_hosts[agent_id, target_host]
    target_subnet = const.host_subnet[target_host]
    can_reach = _can_reach_subnet(state, const, agent_id, target_subnet)

    success = is_active & is_discovered & can_reach

    red_scanned_hosts = state.red_scanned_hosts.at[agent_id, target_host].set(
        state.red_scanned_hosts[agent_id, target_host] | success
    )

    activity = jnp.where(
        success,
        state.red_activity_this_step.at[target_host].set(ACTIVITY_SCAN),
        state.red_activity_this_step,
    )

    return state.replace(
        red_scanned_hosts=red_scanned_hosts,
        red_activity_this_step=activity,
    )


def apply_red_action(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    action_idx: int,
) -> CC4State:
    action_type, target_subnet, target_host = decode_red_action(action_idx, agent_id, const)

    state = jax.lax.cond(
        action_type == ACTION_TYPE_DISCOVER,
        lambda s: _apply_discover(s, const, agent_id, target_subnet),
        lambda s: s,
        state,
    )

    state = jax.lax.cond(
        action_type == ACTION_TYPE_SCAN,
        lambda s: _apply_scan(s, const, agent_id, target_host),
        lambda s: s,
        state,
    )

    return state


def encode_blue_action(action_name: str, target_host: int, agent_id: int) -> int:
    raise NotImplementedError("Subsystem 8+: blue action encoding")


def decode_blue_action(action_idx: int, agent_id: int, const: CC4Const):
    raise NotImplementedError("Subsystem 8+: blue action decoding")


def apply_blue_action(state: CC4State, const: CC4Const, agent_id: int, action_idx: int) -> CC4State:
    raise NotImplementedError("Subsystem 8+: blue action application")

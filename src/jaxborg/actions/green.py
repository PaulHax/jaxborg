import jax
import jax.numpy as jnp

from jaxborg.constants import (
    COMPROMISE_USER,
    GLOBAL_MAX_HOSTS,
    NUM_RED_AGENTS,
    NUM_SUBNETS,
)
from jaxborg.state import CC4Const, CC4State

FP_DETECTION_RATE = 0.01
PHISHING_ERROR_RATE = 0.01

GREEN_SLEEP = 0
GREEN_LOCAL_WORK = 1
GREEN_ACCESS_SERVICE = 2
NUM_GREEN_ACTIONS = 3


def _find_phishing_red_agent(
    state: CC4State,
    const: CC4Const,
    host_idx: jnp.int32,
) -> jnp.int32:
    host_subnet = const.host_subnet[host_idx]
    has_session_on_subnet = jnp.zeros(NUM_RED_AGENTS, dtype=jnp.bool_)

    for r in range(NUM_RED_AGENTS):
        agent_sessions = state.red_sessions[r] & const.host_active
        agent_subnets = jnp.zeros(NUM_SUBNETS, dtype=jnp.bool_)
        for s in range(NUM_SUBNETS):
            agent_subnets = agent_subnets.at[s].set(jnp.any(agent_sessions & (const.host_subnet == s)))
        not_blocked = ~state.blocked_zones[host_subnet]
        can_route = jnp.any(agent_subnets & not_blocked)
        has_session_on_subnet = has_session_on_subnet.at[r].set(jnp.any(agent_sessions) & can_route)

    candidates = has_session_on_subnet
    first_valid = jnp.argmax(candidates)
    any_valid = jnp.any(candidates)
    return jnp.where(any_valid, first_valid, jnp.int32(-1))


def _apply_single_green(
    state: CC4State,
    const: CC4Const,
    host_idx: jnp.int32,
    key: jax.Array,
) -> CC4State:
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    action = jax.random.randint(k1, (), 0, NUM_GREEN_ACTIONS)

    has_service = jnp.any(state.host_services[host_idx])

    # -- GreenLocalWork --
    fp_roll = jax.random.uniform(k2)
    fp_triggered = fp_roll < FP_DETECTION_RATE
    local_fp = (action == GREEN_LOCAL_WORK) & has_service & fp_triggered

    phish_roll = jax.random.uniform(k3)
    phish_triggered = phish_roll < PHISHING_ERROR_RATE
    do_phish = (action == GREEN_LOCAL_WORK) & has_service & phish_triggered

    red_agent = _find_phishing_red_agent(state, const, host_idx)
    any_red_on_host = jnp.any(state.red_sessions[:, host_idx])
    phish_creates_session = do_phish & (red_agent >= 0) & ~any_red_on_host

    red_sessions = jnp.where(
        phish_creates_session,
        state.red_sessions.at[red_agent, host_idx].set(True),
        state.red_sessions,
    )
    red_privilege = jnp.where(
        phish_creates_session,
        state.red_privilege.at[red_agent, host_idx].set(
            jnp.maximum(state.red_privilege[red_agent, host_idx], COMPROMISE_USER)
        ),
        state.red_privilege,
    )
    host_compromised = jnp.where(
        phish_creates_session,
        state.host_compromised.at[host_idx].set(jnp.maximum(state.host_compromised[host_idx], COMPROMISE_USER)),
        state.host_compromised,
    )

    # -- GreenAccessService --
    src_subnet = const.host_subnet[host_idx]
    phase = state.mission_phase
    allowed = const.allowed_subnet_pairs[phase]
    src_in_allowed = jnp.any(allowed[src_subnet])
    own_subnet_only = jnp.zeros(NUM_SUBNETS, dtype=jnp.bool_).at[src_subnet].set(True)
    reachable_subnets = jnp.where(src_in_allowed, allowed[src_subnet], own_subnet_only)

    is_reachable_server = (
        const.host_active
        & const.host_is_server
        & reachable_subnets[const.host_subnet]
        & (jnp.arange(GLOBAL_MAX_HOSTS) != host_idx)
    )
    num_reachable = jnp.sum(is_reachable_server)
    has_reachable = num_reachable > 0

    server_indices = jnp.where(is_reachable_server, jnp.arange(GLOBAL_MAX_HOSTS), GLOBAL_MAX_HOSTS)
    sorted_servers = jnp.sort(server_indices)
    rand_idx = jax.random.randint(k4, (), 0, jnp.maximum(num_reachable, 1))
    dest_host = sorted_servers[rand_idx]
    dest_host = jnp.where(has_reachable, dest_host, jnp.int32(0))

    dest_subnet = const.host_subnet[dest_host]
    blocked_src_to_dst = state.blocked_zones[src_subnet, dest_subnet]
    blocked_dst_to_src = state.blocked_zones[dest_subnet, src_subnet]
    is_blocked = blocked_src_to_dst | blocked_dst_to_src

    do_access = (action == GREEN_ACCESS_SERVICE) & has_reachable

    access_blocked = do_access & is_blocked
    access_fp_roll = jax.random.uniform(k5)
    access_fp = do_access & ~is_blocked & (access_fp_roll < FP_DETECTION_RATE)

    host_activity_detected = state.host_activity_detected
    host_activity_detected = jnp.where(
        access_blocked | access_fp,
        host_activity_detected.at[dest_host].set(True),
        host_activity_detected,
    )

    host_has_malware = jnp.where(
        local_fp,
        state.host_has_malware.at[host_idx].set(True),
        state.host_has_malware,
    )

    return state.replace(
        red_sessions=red_sessions,
        red_privilege=red_privilege,
        host_compromised=host_compromised,
        host_has_malware=host_has_malware,
        host_activity_detected=host_activity_detected,
    )


def apply_green_agents(state: CC4State, const: CC4Const, key: jax.Array) -> CC4State:
    keys = jax.random.split(key, GLOBAL_MAX_HOSTS)

    def step_fn(carry_state, idx):
        is_active = const.green_agent_active[idx]
        new_state = _apply_single_green(carry_state, const, idx, keys[idx])
        out_state = jax.tree.map(
            lambda new, old: jnp.where(is_active, new, old),
            new_state,
            carry_state,
        )
        return out_state, None

    final_state, _ = jax.lax.scan(
        step_fn,
        state,
        jnp.arange(GLOBAL_MAX_HOSTS),
    )
    return final_state

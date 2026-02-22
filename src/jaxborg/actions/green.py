import jax
import jax.numpy as jnp

from jaxborg.actions.rng import sample_green_random
from jaxborg.constants import (
    COMPROMISE_USER,
    GLOBAL_MAX_HOSTS,
    NUM_RED_AGENTS,
    NUM_SERVICES,
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
    k1, k2, k3, k4, k5, k_svc, k_rel = jax.random.split(key, 7)
    t = state.time

    action = sample_green_random(state, t, host_idx, 0, k1, int_range=NUM_GREEN_ACTIONS)

    has_service = jnp.any(state.host_services[host_idx])

    active_services = state.host_services[host_idx]
    num_active = jnp.sum(active_services)
    svc_indices = jnp.where(active_services, jnp.arange(NUM_SERVICES), NUM_SERVICES)
    sorted_svcs = jnp.sort(svc_indices)
    svc_rand_idx = sample_green_random(state, t, host_idx, 1, k_svc, int_range=num_active)
    chosen_svc = sorted_svcs[svc_rand_idx]
    chosen_svc = jnp.where(has_service, chosen_svc, jnp.int32(0))
    svc_reliability = state.host_service_reliability[host_idx, chosen_svc]
    rel_roll = sample_green_random(state, t, host_idx, 2, k_rel, int_range=100)
    work_succeeds = has_service & (rel_roll < svc_reliability)

    # -- GreenLocalWork --
    local_work_failed = (action == GREEN_LOCAL_WORK) & has_service & ~work_succeeds
    green_lwf_this_step = jnp.where(
        local_work_failed,
        state.green_lwf_this_step.at[host_idx].set(True),
        state.green_lwf_this_step,
    )

    fp_roll = sample_green_random(state, t, host_idx, 3, k2)
    fp_triggered = fp_roll < FP_DETECTION_RATE
    local_fp = (action == GREEN_LOCAL_WORK) & work_succeeds & fp_triggered

    phish_roll = sample_green_random(state, t, host_idx, 4, k3)
    phish_triggered = phish_roll < PHISHING_ERROR_RATE
    do_phish = (action == GREEN_LOCAL_WORK) & work_succeeds & phish_triggered

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
    own_subnet = jnp.zeros(NUM_SUBNETS, dtype=jnp.bool_).at[src_subnet].set(True)
    src_in_allowed = jnp.any(allowed[src_subnet])
    reachable_subnets = jnp.where(src_in_allowed, allowed[src_subnet] | own_subnet, own_subnet)

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
    rand_idx = sample_green_random(state, t, host_idx, 5, k4, int_range=num_reachable)
    dest_host = sorted_servers[rand_idx]
    dest_host = jnp.where(has_reachable, dest_host, jnp.int32(0))

    dest_subnet = const.host_subnet[dest_host]
    blocked_src_to_dst = state.blocked_zones[src_subnet, dest_subnet]
    blocked_dst_to_src = state.blocked_zones[dest_subnet, src_subnet]
    is_blocked = blocked_src_to_dst | blocked_dst_to_src

    do_access = (action == GREEN_ACCESS_SERVICE) & has_reachable

    access_blocked = do_access & is_blocked
    access_fp_roll = sample_green_random(state, t, host_idx, 6, k5)
    access_fp = do_access & ~is_blocked & (access_fp_roll < FP_DETECTION_RATE)

    green_asf_this_step = jnp.where(
        access_blocked,
        state.green_asf_this_step.at[dest_host].set(True),
        state.green_asf_this_step,
    )

    host_activity_detected = state.host_activity_detected
    host_activity_detected = jnp.where(
        access_blocked | access_fp,
        host_activity_detected.at[dest_host].set(True),
        host_activity_detected,
    )

    host_activity_detected = jnp.where(
        local_fp,
        host_activity_detected.at[host_idx].set(True),
        host_activity_detected,
    )

    return state.replace(
        red_sessions=red_sessions,
        red_privilege=red_privilege,
        host_compromised=host_compromised,
        host_activity_detected=host_activity_detected,
        green_lwf_this_step=green_lwf_this_step,
        green_asf_this_step=green_asf_this_step,
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

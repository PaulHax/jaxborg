import jax
import jax.numpy as jnp

from jaxborg.actions.encoding import (
    RED_AGGRESSIVE_SCAN_START,
    RED_DEGRADE_START,
    RED_DISCOVER_DECEPTION_START,
    RED_DISCOVER_START,
    RED_EXPLOIT_SSH_START,
    RED_IMPACT_START,
    RED_PRIVESC_START,
    RED_SLEEP,
    RED_STEALTH_SCAN_START,
    RED_WITHDRAW_START,
)
from jaxborg.constants import (
    GLOBAL_MAX_HOSTS,
)
from jaxborg.state import CC4Const, CC4State

FSM_K = 0
FSM_KD = 1
FSM_S = 2
FSM_SD = 3
FSM_U = 4
FSM_UD = 5
FSM_R = 6
FSM_RD = 7
FSM_F = 8
NUM_FSM_STATES = 9

FSM_ACT_DISCOVER = 0
FSM_ACT_AGGRESSIVE_SCAN = 1
FSM_ACT_STEALTH_SCAN = 2
FSM_ACT_DISCOVER_DECEPTION = 3
FSM_ACT_EXPLOIT = 4
FSM_ACT_PRIVESC = 5
FSM_ACT_IMPACT = 6
FSM_ACT_DEGRADE = 7
FSM_ACT_WITHDRAW = 8
NUM_FSM_ACTIONS = 9

_SENTINEL = -1

TRANSITION_SUCCESS = jnp.array(
    [
        [FSM_KD, FSM_S, FSM_S, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL],
        [FSM_KD, FSM_SD, FSM_SD, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL],
        [FSM_SD, _SENTINEL, _SENTINEL, FSM_S, FSM_U, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL],
        [FSM_SD, _SENTINEL, _SENTINEL, FSM_SD, FSM_UD, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL],
        [FSM_UD, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, FSM_R, _SENTINEL, _SENTINEL, FSM_S],
        [FSM_UD, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, FSM_RD, _SENTINEL, _SENTINEL, FSM_SD],
        [FSM_RD, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, FSM_R, FSM_R, FSM_S],
        [FSM_RD, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, FSM_RD, FSM_RD, FSM_SD],
        [FSM_F, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL],
    ],
    dtype=jnp.int32,
)

TRANSITION_FAILURE = jnp.array(
    [
        [FSM_K, FSM_K, FSM_K, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL],
        [FSM_KD, FSM_KD, FSM_KD, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL],
        [FSM_S, _SENTINEL, _SENTINEL, FSM_S, FSM_S, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL],
        [FSM_SD, _SENTINEL, _SENTINEL, FSM_SD, FSM_SD, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL],
        [FSM_U, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, FSM_U, _SENTINEL, _SENTINEL, FSM_U],
        [FSM_UD, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, FSM_UD, _SENTINEL, _SENTINEL, FSM_UD],
        [FSM_R, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, FSM_R, FSM_R, FSM_R],
        [FSM_RD, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, FSM_RD, FSM_RD, FSM_RD],
        [FSM_F, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL, _SENTINEL],
    ],
    dtype=jnp.int32,
)

_prob_none = -1.0

PROBABILITY_MATRIX = jnp.array(
    [
        [0.5, 0.25, 0.25, _prob_none, _prob_none, _prob_none, _prob_none, _prob_none, _prob_none],
        [_prob_none, 0.5, 0.5, _prob_none, _prob_none, _prob_none, _prob_none, _prob_none, _prob_none],
        [0.25, _prob_none, _prob_none, 0.25, 0.5, _prob_none, _prob_none, _prob_none, _prob_none],
        [_prob_none, _prob_none, _prob_none, 0.25, 0.75, _prob_none, _prob_none, _prob_none, _prob_none],
        [0.5, _prob_none, _prob_none, _prob_none, _prob_none, 0.5, _prob_none, _prob_none, 0.0],
        [_prob_none, _prob_none, _prob_none, _prob_none, _prob_none, 1.0, _prob_none, _prob_none, 0.0],
        [0.5, _prob_none, _prob_none, _prob_none, _prob_none, _prob_none, 0.25, 0.25, 0.0],
        [_prob_none, _prob_none, _prob_none, _prob_none, _prob_none, _prob_none, 0.5, 0.5, 0.0],
    ],
    dtype=jnp.float32,
)

ACTION_VALID_MASK = PROBABILITY_MATRIX >= 0.0


def _fsm_action_to_jax_action(fsm_action, target_host, target_subnet):
    return jax.lax.switch(
        fsm_action,
        [
            lambda: RED_DISCOVER_START + target_subnet,
            lambda: RED_AGGRESSIVE_SCAN_START + target_host,
            lambda: RED_STEALTH_SCAN_START + target_host,
            lambda: RED_DISCOVER_DECEPTION_START + target_host,
            lambda: RED_EXPLOIT_SSH_START + target_host,
            lambda: RED_PRIVESC_START + target_host,
            lambda: RED_IMPACT_START + target_host,
            lambda: RED_DEGRADE_START + target_host,
            lambda: RED_WITHDRAW_START + target_host,
        ],
    )


def fsm_red_get_action(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    key: jax.Array,
) -> int:
    fsm_states = state.fsm_host_states[agent_id]
    discovered = state.red_discovered_hosts[agent_id]
    active = const.host_active

    eligible = discovered & active & (fsm_states != FSM_F)

    key1, key2 = jax.random.split(key)

    any_eligible = jnp.any(eligible)

    host_probs = jnp.where(eligible, 1.0, 0.0)
    host_total = jnp.sum(host_probs)
    host_probs = host_probs / jnp.maximum(host_total, 1e-8)
    chosen_host = jax.random.choice(key1, GLOBAL_MAX_HOSTS, p=host_probs)

    host_state = fsm_states[chosen_host]
    host_state_clamped = jnp.clip(host_state, 0, NUM_FSM_STATES - 2)

    action_probs_raw = PROBABILITY_MATRIX[host_state_clamped]
    valid_mask = ACTION_VALID_MASK[host_state_clamped]
    action_probs = jnp.where(valid_mask, jnp.maximum(action_probs_raw, 0.0), 0.0)
    action_total = jnp.sum(action_probs)
    action_probs = action_probs / jnp.maximum(action_total, 1e-8)
    chosen_fsm_action = jax.random.choice(key2, NUM_FSM_ACTIONS, p=action_probs)

    target_subnet = const.host_subnet[chosen_host]
    jax_action = _fsm_action_to_jax_action(chosen_fsm_action, chosen_host, target_subnet)

    return jnp.where(any_eligible, jax_action, RED_SLEEP)


def fsm_red_update_state(
    fsm_states: jnp.ndarray,
    agent_id: int,
    target_host: jnp.ndarray,
    fsm_action: int,
    success: jnp.ndarray,
) -> jnp.ndarray:
    cur = fsm_states[agent_id, target_host]

    next_success = TRANSITION_SUCCESS[cur, fsm_action]
    next_failure = TRANSITION_FAILURE[cur, fsm_action]
    next_state = jnp.where(success, next_success, next_failure)

    valid = next_state != _SENTINEL
    new_state = jnp.where(valid, next_state, cur)

    return fsm_states.at[agent_id, target_host].set(new_state)


def fsm_red_process_session_removal(
    state: CC4State,
    agent_id: int,
) -> jnp.ndarray:
    fsm_states = state.fsm_host_states[agent_id]
    has_session = state.red_sessions[agent_id]
    was_compromised = (fsm_states == FSM_U) | (fsm_states == FSM_UD) | (fsm_states == FSM_R) | (fsm_states == FSM_RD)
    lost_session = was_compromised & ~has_session

    new_states = jnp.where(lost_session, FSM_KD, fsm_states)
    return state.fsm_host_states.at[agent_id].set(new_states)


def fsm_red_init_states(
    const: CC4Const,
    agent_id: int,
) -> jnp.ndarray:
    start_host = const.red_start_hosts[agent_id]
    fsm = jnp.full(GLOBAL_MAX_HOSTS, FSM_K, dtype=jnp.int32)
    fsm = fsm.at[start_host].set(FSM_U)
    return fsm

import jax
import jax.numpy as jnp

from jaxborg.actions import apply_blue_action, apply_red_action
from jaxborg.actions.encoding import (
    BLUE_ACTION_DURATIONS,
    RED_ACTION_DURATIONS,
    decode_blue_action,
    decode_red_action,
)
from jaxborg.state import CC4Const, CC4State


def process_red_with_duration(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    action_idx: int,
    key: jax.Array,
) -> CC4State:
    is_busy = state.red_pending_ticks[agent_id] > 0

    effective_action = jnp.where(is_busy, state.red_pending_action[agent_id], action_idx)
    effective_key = jnp.where(
        is_busy,
        state.red_pending_key[agent_id],
        jnp.asarray(key, dtype=jnp.uint32),
    )

    action_type, _, _ = decode_red_action(effective_action, agent_id, const)
    duration = RED_ACTION_DURATIONS[action_type]
    current_ticks = jnp.where(is_busy, state.red_pending_ticks[agent_id], duration)

    new_ticks = current_ticks - 1
    should_execute = new_ticks <= 0

    new_state = jax.lax.cond(
        should_execute,
        lambda s: apply_red_action(s, const, agent_id, effective_action, effective_key),
        lambda s: s,
        state,
    )

    final_ticks = jnp.where(should_execute, jnp.int32(0), new_ticks)
    new_state = new_state.replace(
        red_pending_ticks=new_state.red_pending_ticks.at[agent_id].set(final_ticks),
        red_pending_action=new_state.red_pending_action.at[agent_id].set(effective_action),
        red_pending_key=new_state.red_pending_key.at[agent_id].set(effective_key),
    )

    return new_state


def process_blue_with_duration(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    action_idx: int,
) -> CC4State:
    is_busy = state.blue_pending_ticks[agent_id] > 0

    effective_action = jnp.where(is_busy, state.blue_pending_action[agent_id], action_idx)

    action_type, _, _, _, _ = decode_blue_action(effective_action, agent_id, const)
    duration = BLUE_ACTION_DURATIONS[action_type]
    current_ticks = jnp.where(is_busy, state.blue_pending_ticks[agent_id], duration)

    new_ticks = current_ticks - 1
    should_execute = new_ticks <= 0

    new_state = jax.lax.cond(
        should_execute,
        lambda s: apply_blue_action(s, const, agent_id, effective_action),
        lambda s: s,
        state,
    )

    final_ticks = jnp.where(should_execute, jnp.int32(0), new_ticks)
    new_state = new_state.replace(
        blue_pending_ticks=new_state.blue_pending_ticks.at[agent_id].set(final_ticks),
        blue_pending_action=new_state.blue_pending_action.at[agent_id].set(effective_action),
    )

    return new_state

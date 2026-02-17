import chex
import jax.numpy as jnp

from jaxborg.actions.red_common import can_reach_subnet
from jaxborg.state import CC4Const, CC4State


def apply_discover_deception(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    is_active = const.host_active[target_host]
    is_scanned = state.red_scanned_hosts[agent_id, target_host]
    target_subnet = const.host_subnet[target_host]
    can_reach = can_reach_subnet(state, const, agent_id, target_subnet)

    success = is_active & is_scanned & can_reach
    has_decoys = jnp.any(state.host_decoys[target_host])
    detected = success & has_decoys

    return state.replace(
        fsm_host_states=jnp.where(
            detected,
            _apply_decoy_detection(state.fsm_host_states, agent_id, target_host),
            state.fsm_host_states,
        ),
    )


def _apply_decoy_detection(fsm_host_states, agent_id, target_host):
    from jaxborg.agents.fsm_red import FSM_K, FSM_KD, FSM_R, FSM_RD, FSM_S, FSM_SD, FSM_U, FSM_UD

    cur = fsm_host_states[agent_id, target_host]
    new_state = cur
    new_state = jnp.where(cur == FSM_K, FSM_KD, new_state)
    new_state = jnp.where(cur == FSM_S, FSM_SD, new_state)
    new_state = jnp.where(cur == FSM_U, FSM_UD, new_state)
    new_state = jnp.where(cur == FSM_R, FSM_RD, new_state)
    return fsm_host_states.at[agent_id, target_host].set(new_state)

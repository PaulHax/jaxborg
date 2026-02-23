import jax.numpy as jnp

from jaxborg.constants import ACTIVITY_NONE, COMPROMISE_NONE, COMPROMISE_USER, NUM_RED_AGENTS
from jaxborg.state import CC4Const, CC4State


def apply_blue_remove(state: CC4State, const: CC4Const, agent_id: int, target_host: int) -> CC4State:
    covers_host = const.blue_agent_hosts[agent_id, target_host]
    is_detected = state.host_activity_detected[target_host]
    has_malware = state.host_has_malware[target_host]
    has_fresh_activity = state.red_activity_this_step[target_host] != ACTIVITY_NONE

    new_sessions = state.red_sessions
    new_privilege = state.red_privilege
    for r in range(NUM_RED_AGENTS):
        is_user = state.red_privilege[r, target_host] == COMPROMISE_USER
        should_clear = covers_host & is_detected & has_malware & has_fresh_activity & is_user
        new_sessions = jnp.where(
            should_clear,
            new_sessions.at[r, target_host].set(False),
            new_sessions,
        )
        new_privilege = jnp.where(
            should_clear,
            new_privilege.at[r, target_host].set(COMPROMISE_NONE),
            new_privilege,
        )

    any_compromised = jnp.any(new_privilege[:, target_host] > 0)
    new_host_compromised = jnp.where(
        covers_host & is_detected & has_malware & has_fresh_activity & ~any_compromised,
        state.host_compromised.at[target_host].set(COMPROMISE_NONE),
        state.host_compromised,
    )

    return state.replace(
        red_sessions=new_sessions,
        red_privilege=new_privilege,
        host_compromised=new_host_compromised,
    )

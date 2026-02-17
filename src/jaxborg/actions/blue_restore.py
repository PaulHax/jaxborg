import jax.numpy as jnp

from jaxborg.constants import COMPROMISE_NONE
from jaxborg.state import CC4Const, CC4State


def apply_blue_restore(state: CC4State, const: CC4Const, agent_id: int, target_host: int) -> CC4State:
    covers_host = const.blue_agent_hosts[agent_id, target_host]

    host_compromised = jnp.where(
        covers_host,
        state.host_compromised.at[target_host].set(COMPROMISE_NONE),
        state.host_compromised,
    )

    red_sessions = jnp.where(
        covers_host,
        state.red_sessions.at[:, target_host].set(False),
        state.red_sessions,
    )

    red_privilege = jnp.where(
        covers_host,
        state.red_privilege.at[:, target_host].set(COMPROMISE_NONE),
        state.red_privilege,
    )

    host_services = jnp.where(
        covers_host,
        state.host_services.at[target_host].set(const.initial_services[target_host]),
        state.host_services,
    )

    host_has_malware = jnp.where(
        covers_host,
        state.host_has_malware.at[target_host].set(False),
        state.host_has_malware,
    )

    host_decoys = jnp.where(
        covers_host,
        state.host_decoys.at[target_host].set(False),
        state.host_decoys,
    )

    host_activity_detected = jnp.where(
        covers_host,
        state.host_activity_detected.at[target_host].set(False),
        state.host_activity_detected,
    )

    ot_service_stopped = jnp.where(
        covers_host,
        state.ot_service_stopped.at[target_host].set(False),
        state.ot_service_stopped,
    )

    host_service_reliability = jnp.where(
        covers_host,
        state.host_service_reliability.at[target_host].set(100),
        state.host_service_reliability,
    )

    return state.replace(
        host_compromised=host_compromised,
        red_sessions=red_sessions,
        red_privilege=red_privilege,
        host_services=host_services,
        host_has_malware=host_has_malware,
        host_decoys=host_decoys,
        host_activity_detected=host_activity_detected,
        ot_service_stopped=ot_service_stopped,
        host_service_reliability=host_service_reliability,
    )

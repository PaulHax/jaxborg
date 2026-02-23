import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.constants import COMPROMISE_USER
from jaxborg.reassignment import reassign_cross_subnet_sessions
from jaxborg.state import create_initial_state
from jaxborg.topology import build_topology


@pytest.fixture
def jax_const():
    return build_topology(jnp.array([42]), num_steps=500)


def _base_state(const):
    state = create_initial_state()
    return state.replace(host_services=jnp.array(const.initial_services))


def _find_cross_subnet_transfer_case(const):
    host_subnets = np.array(const.host_subnet)
    active = np.array(const.host_active)
    allowed = np.array(const.red_agent_subnets)
    for h in range(int(const.num_hosts)):
        if not active[h]:
            continue
        subnet = int(host_subnets[h])
        if allowed[0, subnet]:
            continue
        owners = np.where(allowed[:, subnet])[0]
        if owners.size == 0:
            continue
        owner = int(owners[0])
        if owner != 0:
            return h, owner
    return None, None


def test_reassignment_marks_existing_session_host_as_discovered(jax_const):
    state = _base_state(jax_const)
    host = int(jax_const.red_start_hosts[0])

    state = state.replace(
        red_sessions=state.red_sessions.at[0, host].set(True),
    )
    assert not bool(state.red_discovered_hosts[0, host])

    new_state = reassign_cross_subnet_sessions(state, jax_const)

    assert bool(new_state.red_sessions[0, host])
    assert bool(new_state.red_discovered_hosts[0, host])


def test_reassignment_transfers_session_and_discovery_to_owner(jax_const):
    host, owner = _find_cross_subnet_transfer_case(jax_const)
    if host is None:
        pytest.skip("No cross-subnet reassignment case found")

    state = _base_state(jax_const)
    state = state.replace(
        red_sessions=state.red_sessions.at[0, host].set(True),
        red_privilege=state.red_privilege.at[0, host].set(COMPROMISE_USER),
    )
    assert not bool(state.red_discovered_hosts[0, host])

    new_state = reassign_cross_subnet_sessions(state, jax_const)

    assert not bool(new_state.red_sessions[0, host])
    assert bool(new_state.red_sessions[owner, host])
    assert bool(new_state.red_discovered_hosts[owner, host])
    assert not bool(new_state.red_discovered_hosts[0, host])

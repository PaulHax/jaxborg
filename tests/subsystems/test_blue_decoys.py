import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions import apply_blue_action
from jaxborg.actions.blue_decoys import apply_blue_decoy
from jaxborg.actions.encoding import (
    BLUE_ACTION_TYPE_DECOY,
    BLUE_DECOY_START,
    decode_blue_action,
    encode_blue_action,
)
from jaxborg.constants import (
    DECOY_IDS,
    GLOBAL_MAX_HOSTS,
    NUM_BLUE_AGENTS,
    NUM_DECOY_TYPES,
)
from jaxborg.state import create_initial_state
from jaxborg.topology import build_const_from_cyborg


def _make_cyborg_env():
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=SleepAgent,
        red_agent_class=SleepAgent,
        steps=500,
    )
    return CybORG(scenario_generator=sg, seed=42)


@pytest.fixture
def jax_const():
    return build_const_from_cyborg(_make_cyborg_env())


def _make_jax_state(const):
    state = create_initial_state()
    state = state.replace(host_services=jnp.array(const.initial_services))
    return state


def _find_host_in_subnet(const, subnet_name, exclude_router=True):
    from jaxborg.constants import SUBNET_IDS

    sid = SUBNET_IDS[subnet_name]
    for h in range(int(const.num_hosts)):
        if not bool(const.host_active[h]):
            continue
        if int(const.host_subnet[h]) != sid:
            continue
        if exclude_router and bool(const.host_is_router[h]):
            continue
        return h
    return None


def _find_blue_for_host(const, host):
    for b in range(NUM_BLUE_AGENTS):
        if bool(const.blue_agent_hosts[b, host]):
            return b
    return None


HARAKA_IDX = DECOY_IDS["HarakaSMPT"]
APACHE_IDX = DECOY_IDS["Apache"]
TOMCAT_IDX = DECOY_IDS["Tomcat"]
VSFTPD_IDX = DECOY_IDS["Vsftpd"]


class TestBlueDecoyEncoding:
    def test_encode_decoy_haraka(self):
        idx = encode_blue_action("DeployDecoy_HarakaSMPT", 5, 0)
        assert idx == BLUE_DECOY_START + HARAKA_IDX * GLOBAL_MAX_HOSTS + 5

    def test_encode_decoy_apache(self):
        idx = encode_blue_action("DeployDecoy_Apache", 10, 0)
        assert idx == BLUE_DECOY_START + APACHE_IDX * GLOBAL_MAX_HOSTS + 10

    def test_decode_decoy(self, jax_const):
        action_idx = BLUE_DECOY_START + TOMCAT_IDX * GLOBAL_MAX_HOSTS + 7
        action_type, target_host, decoy_type, *_ = decode_blue_action(action_idx, 0, jax_const)
        assert int(action_type) == BLUE_ACTION_TYPE_DECOY
        assert int(target_host) == 7
        assert int(decoy_type) == TOMCAT_IDX

    def test_decode_all_decoy_types(self, jax_const):
        for dtype in range(NUM_DECOY_TYPES):
            action_idx = BLUE_DECOY_START + dtype * GLOBAL_MAX_HOSTS + 3
            action_type, target_host, decoy_type, *_ = decode_blue_action(action_idx, 0, jax_const)
            assert int(action_type) == BLUE_ACTION_TYPE_DECOY
            assert int(target_host) == 3
            assert int(decoy_type) == dtype


class TestApplyBlueDecoy:
    def test_deploy_decoy_sets_flag(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        new_state = apply_blue_decoy(state, jax_const, blue_idx, target, APACHE_IDX)
        assert bool(new_state.host_decoys[target, APACHE_IDX])
        assert not bool(new_state.host_decoys[target, HARAKA_IDX])

    def test_deploy_multiple_decoys(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        state = apply_blue_decoy(state, jax_const, blue_idx, target, APACHE_IDX)
        state = apply_blue_decoy(state, jax_const, blue_idx, target, HARAKA_IDX)
        assert bool(state.host_decoys[target, APACHE_IDX])
        assert bool(state.host_decoys[target, HARAKA_IDX])

    def test_deploy_on_uncovered_host_is_noop(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        uncovering_blue = None
        for b in range(NUM_BLUE_AGENTS):
            if not bool(jax_const.blue_agent_hosts[b, target]):
                uncovering_blue = b
                break
        assert uncovering_blue is not None

        new_state = apply_blue_decoy(state, jax_const, uncovering_blue, target, APACHE_IDX)
        assert not bool(new_state.host_decoys[target, APACHE_IDX])

    def test_does_not_affect_other_hosts(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        other = _find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        assert target is not None and other is not None

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        new_state = apply_blue_decoy(state, jax_const, blue_idx, target, APACHE_IDX)
        np.testing.assert_array_equal(
            np.array(new_state.host_decoys[other]),
            np.array(state.host_decoys[other]),
        )

    def test_jit_compatible(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        jitted = jax.jit(apply_blue_decoy, static_argnums=(2, 3, 4))
        new_state = jitted(state, jax_const, blue_idx, target, APACHE_IDX)
        assert bool(new_state.host_decoys[target, APACHE_IDX])


class TestDecoyViaDispatch:
    def test_decoy_dispatched(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        action_idx = encode_blue_action("DeployDecoy_Tomcat", target, blue_idx)
        new_state = apply_blue_action(state, jax_const, blue_idx, action_idx)
        assert bool(new_state.host_decoys[target, TOMCAT_IDX])

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.actions import apply_blue_action
from jaxborg.actions.blue_restore import apply_blue_restore
from jaxborg.actions.encoding import (
    BLUE_ACTION_TYPE_RESTORE,
    BLUE_RESTORE_START,
    decode_blue_action,
    encode_blue_action,
)
from jaxborg.constants import (
    COMPROMISE_NONE,
    COMPROMISE_PRIVILEGED,
    COMPROMISE_USER,
    NUM_BLUE_AGENTS,
    NUM_DECOY_TYPES,
)
from jaxborg.state import create_initial_state
from jaxborg.topology import build_const_from_cyborg

try:
    from CybORG import CybORG
    from CybORG.Agents import SleepAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    HAS_CYBORG = True
except ImportError:
    HAS_CYBORG = False

cyborg_required = pytest.mark.skipif(not HAS_CYBORG, reason="CybORG not installed")


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


class TestBlueRestoreEncoding:
    def test_encode_restore(self):
        assert encode_blue_action("Restore", 5, 0) == BLUE_RESTORE_START + 5

    def test_decode_restore(self, jax_const):
        action_idx = BLUE_RESTORE_START + 10
        action_type, target_host, *_ = decode_blue_action(action_idx, 0, jax_const)
        assert int(action_type) == BLUE_ACTION_TYPE_RESTORE
        assert int(target_host) == 10


class TestApplyBlueRestore:
    def test_restore_clears_all_red_sessions(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True).at[1, target].set(True),
            red_privilege=state.red_privilege.at[0, target]
            .set(COMPROMISE_USER)
            .at[1, target]
            .set(COMPROMISE_PRIVILEGED),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_PRIVILEGED),
            host_has_malware=state.host_has_malware.at[target].set(True),
        )

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        new_state = apply_blue_restore(state, jax_const, blue_idx, target)
        assert not bool(new_state.red_sessions[0, target])
        assert not bool(new_state.red_sessions[1, target])
        assert int(new_state.red_privilege[0, target]) == COMPROMISE_NONE
        assert int(new_state.red_privilege[1, target]) == COMPROMISE_NONE
        assert int(new_state.host_compromised[target]) == COMPROMISE_NONE
        assert not bool(new_state.host_has_malware[target])

    def test_restore_clears_decoys(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(host_decoys=state.host_decoys.at[target].set(jnp.ones(NUM_DECOY_TYPES, dtype=jnp.bool_)))

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        new_state = apply_blue_restore(state, jax_const, blue_idx, target)
        assert not np.any(np.array(new_state.host_decoys[target]))

    def test_restore_resets_services_to_initial(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(host_services=state.host_services.at[target].set(False))

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        new_state = apply_blue_restore(state, jax_const, blue_idx, target)
        np.testing.assert_array_equal(
            np.array(new_state.host_services[target]),
            np.array(jax_const.initial_services[target]),
        )

    def test_restore_clears_activity_detected(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(host_activity_detected=state.host_activity_detected.at[target].set(True))

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        new_state = apply_blue_restore(state, jax_const, blue_idx, target)
        assert not bool(new_state.host_activity_detected[target])

    def test_restore_clears_ot_service_stopped(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(ot_service_stopped=state.ot_service_stopped.at[target].set(True))

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        new_state = apply_blue_restore(state, jax_const, blue_idx, target)
        assert not bool(new_state.ot_service_stopped[target])

    def test_restore_does_not_affect_other_hosts(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        other = _find_host_in_subnet(jax_const, "OPERATIONAL_ZONE_A")
        assert target is not None and other is not None

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True).at[0, other].set(True),
            red_privilege=state.red_privilege.at[0, target]
            .set(COMPROMISE_PRIVILEGED)
            .at[0, other]
            .set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target]
            .set(COMPROMISE_PRIVILEGED)
            .at[other]
            .set(COMPROMISE_USER),
        )

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        new_state = apply_blue_restore(state, jax_const, blue_idx, target)
        assert not bool(new_state.red_sessions[0, target])
        assert bool(new_state.red_sessions[0, other])
        assert int(new_state.red_privilege[0, other]) == COMPROMISE_USER

    def test_jit_compatible(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_PRIVILEGED),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_PRIVILEGED),
        )

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        jitted = jax.jit(apply_blue_restore, static_argnums=(2, 3))
        new_state = jitted(state, jax_const, blue_idx, target)
        assert not bool(new_state.red_sessions[0, target])
        assert int(new_state.host_compromised[target]) == COMPROMISE_NONE


class TestRestoreViaDispatch:
    def test_restore_dispatched(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_PRIVILEGED),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_PRIVILEGED),
            host_has_malware=state.host_has_malware.at[target].set(True),
        )

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        action_idx = encode_blue_action("Restore", target, blue_idx)
        new_state = apply_blue_action(state, jax_const, blue_idx, action_idx)
        assert not bool(new_state.red_sessions[0, target])
        assert int(new_state.host_compromised[target]) == COMPROMISE_NONE
        assert not bool(new_state.host_has_malware[target])

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Shared.Session import RedAbstractSession
from CybORG.Simulator.Actions import Remove
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions import apply_blue_action, apply_red_action
from jaxborg.actions.blue_remove import apply_blue_remove
from jaxborg.actions.encoding import (
    BLUE_ACTION_TYPE_REMOVE,
    BLUE_REMOVE_START,
    decode_blue_action,
    encode_blue_action,
    encode_red_action,
)
from jaxborg.constants import (
    ACTIVITY_EXPLOIT,
    COMPROMISE_NONE,
    COMPROMISE_PRIVILEGED,
    COMPROMISE_USER,
    GLOBAL_MAX_HOSTS,
    NUM_BLUE_AGENTS,
    SERVICE_IDS,
)
from jaxborg.state import create_initial_state
from jaxborg.topology import build_const_from_cyborg

SSH_SVC = SERVICE_IDS["SSHD"]


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
    start_host = int(const.red_start_hosts[0])
    red_sessions = state.red_sessions.at[0, start_host].set(True)
    return state.replace(red_sessions=red_sessions)


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


def _setup_exploit(state, const, target_h):
    target_subnet = int(const.host_subnet[target_h])
    discover_idx = encode_red_action("DiscoverRemoteSystems", target_subnet, 0)
    state = apply_red_action(state, const, 0, discover_idx, jax.random.PRNGKey(0))
    state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))
    scan_idx = encode_red_action("DiscoverNetworkServices", target_h, 0)
    state = apply_red_action(state, const, 0, scan_idx, jax.random.PRNGKey(0))
    state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))
    exploit_idx = encode_red_action("ExploitRemoteService_cc4SSHBruteForce", target_h, 0)
    state = apply_red_action(state, const, 0, exploit_idx, jax.random.PRNGKey(0))
    return state


class TestBlueRemoveEncoding:
    def test_encode_remove(self):
        assert encode_blue_action("Remove", 5, 0) == BLUE_REMOVE_START + 5

    def test_decode_remove(self, jax_const):
        action_idx = BLUE_REMOVE_START + 10
        action_type, target_host, *_ = decode_blue_action(action_idx, 0, jax_const)
        assert int(action_type) == BLUE_ACTION_TYPE_REMOVE
        assert int(target_host) == 10


class TestApplyBlueRemove:
    def test_remove_clears_user_session(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
            red_activity_this_step=state.red_activity_this_step.at[target].set(ACTIVITY_EXPLOIT),
        )

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        new_state = apply_blue_remove(state, jax_const, blue_idx, target)
        assert not bool(new_state.red_sessions[0, target])
        assert int(new_state.red_privilege[0, target]) == COMPROMISE_NONE
        assert int(new_state.host_compromised[target]) == COMPROMISE_NONE
        assert bool(new_state.host_has_malware[target])

    def test_remove_does_not_clear_privileged_session(self, jax_const):
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

        new_state = apply_blue_remove(state, jax_const, blue_idx, target)
        assert bool(new_state.red_sessions[0, target])
        assert int(new_state.red_privilege[0, target]) == COMPROMISE_PRIVILEGED
        assert bool(new_state.host_has_malware[target])

    def test_remove_on_clean_host_is_noop(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        new_state = apply_blue_remove(state, jax_const, blue_idx, target)
        np.testing.assert_array_equal(np.array(new_state.red_sessions), np.array(state.red_sessions))

    def test_remove_clears_user_but_leaves_other_red_agents_privileged(self, jax_const):
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
            host_activity_detected=state.host_activity_detected.at[target].set(True),
            red_activity_this_step=state.red_activity_this_step.at[target].set(ACTIVITY_EXPLOIT),
        )

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        new_state = apply_blue_remove(state, jax_const, blue_idx, target)
        assert not bool(new_state.red_sessions[0, target])
        assert int(new_state.red_privilege[0, target]) == COMPROMISE_NONE
        assert bool(new_state.red_sessions[1, target])
        assert int(new_state.red_privilege[1, target]) == COMPROMISE_PRIVILEGED
        assert bool(new_state.host_has_malware[target])

    def test_jit_compatible(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
            red_activity_this_step=state.red_activity_this_step.at[target].set(ACTIVITY_EXPLOIT),
        )

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        jitted = jax.jit(apply_blue_remove, static_argnums=(2, 3))
        new_state = jitted(state, jax_const, blue_idx, target)
        assert not bool(new_state.red_sessions[0, target])
        assert int(new_state.red_privilege[0, target]) == COMPROMISE_NONE


class TestRemoveViaDispatch:
    def test_remove_dispatched(self, jax_const):
        state = _make_jax_state(jax_const)
        target = _find_host_in_subnet(jax_const, "RESTRICTED_ZONE_A")
        assert target is not None

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
            red_activity_this_step=state.red_activity_this_step.at[target].set(ACTIVITY_EXPLOIT),
        )

        blue_idx = _find_blue_for_host(jax_const, target)
        assert blue_idx is not None

        action_idx = encode_blue_action("Remove", target, blue_idx)
        new_state = apply_blue_action(state, jax_const, blue_idx, action_idx)
        assert not bool(new_state.red_sessions[0, target])
        assert int(new_state.red_privilege[0, target]) == COMPROMISE_NONE


class TestDifferentialWithCybORG:
    @pytest.fixture
    def cyborg_and_jax(self):
        cyborg_env = _make_cyborg_env()
        const = build_const_from_cyborg(cyborg_env)
        state = _make_jax_state(const)
        return cyborg_env, const, state

    def test_remove_without_suspicious_process_does_not_clear_user_session_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        assert target is not None
        target_hostname = sorted_hosts[target]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        red_session = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent="red_agent_0",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cyborg_state.add_session(red_session)

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
        )

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx)
        new_state = apply_blue_action(state, const, blue_idx, action_idx)

        cyborg_red_sessions = [
            s for s in cyborg_state.sessions["red_agent_0"].values() if s.hostname == target_hostname
        ]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_red_sessions)
        cyborg_host_compromised = 1 if cyborg_has_user_session else 0

        assert bool(new_state.red_sessions[0, target]) == cyborg_has_user_session
        expected_priv = COMPROMISE_USER if cyborg_has_user_session else COMPROMISE_NONE
        assert int(new_state.red_privilege[0, target]) == expected_priv
        assert int(new_state.host_compromised[target]) == cyborg_host_compromised

    def test_remove_with_stale_suspicious_pid_does_not_clear_user_session_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        assert target is not None
        target_hostname = sorted_hosts[target]

        blue_idx = _find_blue_for_host(const, target)
        assert blue_idx is not None

        red_session = RedAbstractSession(
            ident=None,
            hostname=target_hostname,
            username="user",
            agent="red_agent_0",
            parent=0,
            session_type="shell",
            pid=None,
        )
        cyborg_state.add_session(red_session)

        blue_parent = cyborg_state.sessions[f"blue_agent_{blue_idx}"][0]
        stale_pid = 999999
        assert cyborg_state.hosts[target_hostname].get_process(stale_pid) is None
        blue_parent.add_sus_pids(hostname=target_hostname, pid=stale_pid)

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
            host_has_malware=state.host_has_malware.at[target].set(True),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
        )

        remove_action = Remove(session=0, agent=f"blue_agent_{blue_idx}", hostname=target_hostname)
        remove_action.duration = 1
        cyborg_obs = remove_action.execute(cyborg_state)
        assert cyborg_obs.success

        action_idx = encode_blue_action("Remove", target, blue_idx)
        new_state = apply_blue_action(state, const, blue_idx, action_idx)

        cyborg_red_sessions = [
            s for s in cyborg_state.sessions["red_agent_0"].values() if s.hostname == target_hostname
        ]
        cyborg_has_user_session = any(not s.has_privileged_access() for s in cyborg_red_sessions)

        assert cyborg_has_user_session
        assert bool(new_state.red_sessions[0, target]) == cyborg_has_user_session

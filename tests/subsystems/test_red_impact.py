import numpy as np
import jax
import jax.numpy as jnp
import pytest

from jaxborg.constants import (
    GLOBAL_MAX_HOSTS,
    NUM_RED_AGENTS,
    SERVICE_IDS,
    ACTIVITY_NONE,
    ACTIVITY_EXPLOIT,
    COMPROMISE_NONE,
    COMPROMISE_USER,
    COMPROMISE_PRIVILEGED,
)
from jaxborg.state import CC4State, CC4Const, create_initial_state
from jaxborg.topology import build_topology, CYBORG_SUFFIX_TO_ID
from jaxborg.actions import (
    encode_red_action,
    decode_red_action,
    apply_red_action,
    RED_IMPACT_START,
    RED_IMPACT_END,
    RED_PRIVESC_END,
    ACTION_TYPE_IMPACT,
)


try:
    from CybORG import CybORG
    from CybORG.Agents import SleepAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    from CybORG.Simulator.Actions import Impact
    from CybORG.Shared.Session import RedAbstractSession
    from CybORG.Shared.Enums import ProcessName
    HAS_CYBORG = True
except ImportError:
    HAS_CYBORG = False

cyborg_required = pytest.mark.skipif(not HAS_CYBORG, reason="CybORG not installed")

SSH_SVC = SERVICE_IDS["SSHD"]
OT_SVC = SERVICE_IDS["OTSERVICE"]


@pytest.fixture
def jax_const():
    return build_topology(jnp.array([42]), num_steps=500)


def _setup_privileged_state(jax_const, target_host):
    state = create_initial_state()
    state = state.replace(host_services=jnp.array(jax_const.initial_services))

    start_host = int(jax_const.red_start_hosts[0])
    red_sessions = state.red_sessions.at[0, start_host].set(True)
    state = state.replace(red_sessions=red_sessions)

    target_subnet = int(jax_const.host_subnet[target_host])
    discover_idx = encode_red_action("DiscoverRemoteSystems", target_subnet, 0)
    state = apply_red_action(state, jax_const, 0, discover_idx)
    state = state.replace(
        red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32)
    )

    scan_idx = encode_red_action("DiscoverNetworkServices", target_host, 0)
    state = apply_red_action(state, jax_const, 0, scan_idx)
    state = state.replace(
        red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32)
    )

    exploit_idx = encode_red_action("ExploitRemoteService_cc4SSHBruteForce", target_host, 0)
    state = apply_red_action(state, jax_const, 0, exploit_idx)
    state = state.replace(
        red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32)
    )

    privesc_idx = encode_red_action("PrivilegeEscalate", target_host, 0)
    state = apply_red_action(state, jax_const, 0, privesc_idx)
    state = state.replace(
        red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32)
    )

    return state


def _find_ot_host(jax_const):
    for h in range(jax_const.num_hosts):
        if (jax_const.host_active[h]
                and not jax_const.host_is_router[h]
                and jax_const.initial_services[h, OT_SVC]
                and jax_const.initial_services[h, SSH_SVC]
                and jax_const.host_has_bruteforceable_user[h]
                and h != int(jax_const.red_start_hosts[0])):
            return h
    return None


def _find_non_ot_host(jax_const):
    for h in range(jax_const.num_hosts):
        if (jax_const.host_active[h]
                and not jax_const.host_is_router[h]
                and not jax_const.initial_services[h, OT_SVC]
                and jax_const.initial_services[h, SSH_SVC]
                and jax_const.host_has_bruteforceable_user[h]
                and h != int(jax_const.red_start_hosts[0])):
            return h
    return None


class TestImpactEncoding:

    def test_impact_range_starts_after_privesc(self):
        assert RED_IMPACT_START == RED_PRIVESC_END

    def test_impact_range_is_max_hosts_wide(self):
        assert RED_IMPACT_END - RED_IMPACT_START == GLOBAL_MAX_HOSTS

    def test_encode_per_host(self):
        for h in [0, 5, 50, GLOBAL_MAX_HOSTS - 1]:
            code = encode_red_action("Impact", h, 0)
            assert code == RED_IMPACT_START + h

    def test_decode_roundtrip(self, jax_const):
        for h in [0, 5, 50, GLOBAL_MAX_HOSTS - 1]:
            code = encode_red_action("Impact", h, 0)
            action_type, target_subnet, target_host = decode_red_action(code, 0, jax_const)
            assert int(action_type) == ACTION_TYPE_IMPACT
            assert int(target_subnet) == -1
            assert int(target_host) == h


class TestApplyImpact:

    def test_impact_stops_ot_service(self, jax_const):
        target = _find_ot_host(jax_const)
        if target is None:
            pytest.skip("No OT host found")

        state = _setup_privileged_state(jax_const, target)
        assert int(state.red_privilege[0, target]) == COMPROMISE_PRIVILEGED
        assert bool(state.host_services[target, OT_SVC])

        action_idx = encode_red_action("Impact", target, 0)
        new_state = apply_red_action(state, jax_const, 0, action_idx)

        assert not bool(new_state.host_services[target, OT_SVC])
        assert bool(new_state.ot_service_stopped[target])

    def test_impact_sets_activity(self, jax_const):
        target = _find_ot_host(jax_const)
        if target is None:
            pytest.skip("No OT host found")

        state = _setup_privileged_state(jax_const, target)
        action_idx = encode_red_action("Impact", target, 0)
        new_state = apply_red_action(state, jax_const, 0, action_idx)
        assert int(new_state.red_activity_this_step[target]) == ACTIVITY_EXPLOIT

    def test_impact_fails_without_privileged_access(self, jax_const):
        target = _find_ot_host(jax_const)
        if target is None:
            pytest.skip("No OT host found")

        state = create_initial_state()
        state = state.replace(host_services=jnp.array(jax_const.initial_services))
        start_host = int(jax_const.red_start_hosts[0])
        red_sessions = state.red_sessions.at[0, start_host].set(True)
        red_sessions = red_sessions.at[0, target].set(True)
        red_privilege = state.red_privilege.at[0, target].set(COMPROMISE_USER)
        state = state.replace(red_sessions=red_sessions, red_privilege=red_privilege)

        action_idx = encode_red_action("Impact", target, 0)
        new_state = apply_red_action(state, jax_const, 0, action_idx)

        assert bool(new_state.host_services[target, OT_SVC])
        assert not bool(new_state.ot_service_stopped[target])

    def test_impact_fails_without_session(self, jax_const):
        target = _find_ot_host(jax_const)
        if target is None:
            pytest.skip("No OT host found")

        state = create_initial_state()
        state = state.replace(host_services=jnp.array(jax_const.initial_services))

        action_idx = encode_red_action("Impact", target, 0)
        new_state = apply_red_action(state, jax_const, 0, action_idx)

        assert bool(new_state.host_services[target, OT_SVC])
        assert not bool(new_state.ot_service_stopped[target])

    def test_impact_fails_without_ot_service(self, jax_const):
        target = _find_non_ot_host(jax_const)
        if target is None:
            pytest.skip("No non-OT host found")

        state = _setup_privileged_state(jax_const, target)
        assert int(state.red_privilege[0, target]) == COMPROMISE_PRIVILEGED
        assert not bool(state.host_services[target, OT_SVC])

        action_idx = encode_red_action("Impact", target, 0)
        new_state = apply_red_action(state, jax_const, 0, action_idx)

        assert not bool(new_state.ot_service_stopped[target])
        assert int(new_state.red_activity_this_step[target]) == ACTIVITY_NONE

    def test_impact_noop_when_already_stopped(self, jax_const):
        target = _find_ot_host(jax_const)
        if target is None:
            pytest.skip("No OT host found")

        state = _setup_privileged_state(jax_const, target)
        action_idx = encode_red_action("Impact", target, 0)
        state1 = apply_red_action(state, jax_const, 0, action_idx)
        assert bool(state1.ot_service_stopped[target])
        assert not bool(state1.host_services[target, OT_SVC])

        state1 = state1.replace(
            red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32)
        )
        state2 = apply_red_action(state1, jax_const, 0, action_idx)
        assert int(state2.red_activity_this_step[target]) == ACTIVITY_NONE

    def test_impact_does_not_affect_other_agents(self, jax_const):
        target = _find_ot_host(jax_const)
        if target is None:
            pytest.skip("No OT host found")

        state = _setup_privileged_state(jax_const, target)
        action_idx = encode_red_action("Impact", target, 0)
        new_state = apply_red_action(state, jax_const, 0, action_idx)

        for agent in range(1, NUM_RED_AGENTS):
            np.testing.assert_array_equal(
                np.array(new_state.red_sessions[agent]),
                np.array(state.red_sessions[agent]),
            )
            np.testing.assert_array_equal(
                np.array(new_state.red_privilege[agent]),
                np.array(state.red_privilege[agent]),
            )

    def test_impact_only_affects_target(self, jax_const):
        target = _find_ot_host(jax_const)
        if target is None:
            pytest.skip("No OT host found")

        state = _setup_privileged_state(jax_const, target)
        action_idx = encode_red_action("Impact", target, 0)
        new_state = apply_red_action(state, jax_const, 0, action_idx)

        for h in range(jax_const.num_hosts):
            if h == target:
                continue
            assert bool(new_state.host_services[h, OT_SVC]) == bool(state.host_services[h, OT_SVC])
            assert bool(new_state.ot_service_stopped[h]) == bool(state.ot_service_stopped[h])

    def test_jit_compatible(self, jax_const):
        target = _find_ot_host(jax_const)
        if target is None:
            pytest.skip("No OT host found")

        state = _setup_privileged_state(jax_const, target)
        action_idx = encode_red_action("Impact", target, 0)
        jitted = jax.jit(apply_red_action, static_argnums=(2,))
        new_state = jitted(state, jax_const, 0, action_idx)
        assert not bool(new_state.host_services[target, OT_SVC])
        assert bool(new_state.ot_service_stopped[target])


class TestImpactChain:

    def test_full_killchain_with_impact(self, jax_const):
        target = _find_ot_host(jax_const)
        if target is None:
            pytest.skip("No OT host found")

        state = _setup_privileged_state(jax_const, target)
        assert int(state.red_privilege[0, target]) == COMPROMISE_PRIVILEGED
        assert bool(state.red_sessions[0, target])
        assert bool(state.host_services[target, OT_SVC])

        impact_idx = encode_red_action("Impact", target, 0)
        state = apply_red_action(state, jax_const, 0, impact_idx)

        assert not bool(state.host_services[target, OT_SVC])
        assert bool(state.ot_service_stopped[target])
        assert int(state.red_activity_this_step[target]) == ACTIVITY_EXPLOIT
        assert bool(state.red_sessions[0, target])
        assert int(state.red_privilege[0, target]) == COMPROMISE_PRIVILEGED


@cyborg_required
class TestDifferentialWithCybORG:

    @pytest.fixture
    def cyborg_env(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=SleepAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        return CybORG(scenario_generator=sg, seed=42)

    @pytest.fixture
    def cyborg_and_jax(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg
        const = build_const_from_cyborg(cyborg_env)
        state = create_initial_state()
        state = state.replace(host_services=jnp.array(const.initial_services))
        start_host = int(const.red_start_hosts[0])
        red_sessions = state.red_sessions.at[0, start_host].set(True)
        state = state.replace(red_sessions=red_sessions)
        return cyborg_env, const, state

    def _inject_privileged_session(self, cyborg_env, const, state, target_h):
        """Inject a privileged red session on target_h in both CybORG and JAX."""
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())
        target_hostname = sorted_hosts[target_h]

        session = RedAbstractSession(
            hostname=target_hostname, username="root", agent="red_agent_0",
            parent=0, session_type=None, ident=None, pid=None,
        )
        cyborg_state.add_session(session)

        red_sessions = state.red_sessions.at[0, target_h].set(True)
        red_privilege = state.red_privilege.at[0, target_h].set(COMPROMISE_PRIVILEGED)
        host_compromised = state.host_compromised.at[target_h].set(COMPROMISE_PRIVILEGED)
        state = state.replace(
            red_sessions=red_sessions,
            red_privilege=red_privilege,
            host_compromised=host_compromised,
        )
        return state, target_hostname

    def _find_ot_host_idx(self, const):
        for h in range(const.num_hosts):
            if (const.host_active[h]
                    and not const.host_is_router[h]
                    and const.initial_services[h, OT_SVC]):
                return h
        return None

    def _find_non_ot_host_idx(self, const):
        for h in range(const.num_hosts):
            if (const.host_active[h]
                    and not const.host_is_router[h]
                    and not const.initial_services[h, OT_SVC]
                    and h != int(const.red_start_hosts[0])):
                return h
        return None

    def test_impact_success_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state

        target_h = self._find_ot_host_idx(const)
        assert target_h is not None, "No OT host found"

        state, target_hostname = self._inject_privileged_session(
            cyborg_env, const, state, target_h
        )

        impact = Impact(hostname=target_hostname, session=0, agent="red_agent_0")
        cyborg_obs = impact.execute(cyborg_state)
        cyborg_success = cyborg_obs.success

        impact_idx = encode_red_action("Impact", target_h, 0)
        new_state = apply_red_action(state, const, 0, impact_idx)

        jax_stopped = bool(new_state.ot_service_stopped[target_h])
        jax_svc_off = not bool(new_state.host_services[target_h, OT_SVC])

        assert jax_stopped == cyborg_success, (
            f"JAX ot_stopped={jax_stopped} but CybORG success={cyborg_success} "
            f"for host {target_h} ({target_hostname})"
        )
        assert jax_svc_off == cyborg_success

    def test_impact_ot_service_state_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state

        target_h = self._find_ot_host_idx(const)
        assert target_h is not None, "No OT host found"

        state, target_hostname = self._inject_privileged_session(
            cyborg_env, const, state, target_h
        )

        cyborg_host = cyborg_state.hosts[target_hostname]
        cyborg_ot_before = any(
            svc.active for sname, svc in cyborg_host.services.items()
            if sname == ProcessName.OTSERVICE
        )
        jax_ot_before = bool(state.host_services[target_h, OT_SVC])
        assert jax_ot_before == cyborg_ot_before

        impact = Impact(hostname=target_hostname, session=0, agent="red_agent_0")
        impact.execute(cyborg_state)

        impact_idx = encode_red_action("Impact", target_h, 0)
        new_state = apply_red_action(state, const, 0, impact_idx)

        cyborg_ot_after = any(
            svc.active for sname, svc in cyborg_host.services.items()
            if sname == ProcessName.OTSERVICE
        )
        jax_ot_after = bool(new_state.host_services[target_h, OT_SVC])

        assert jax_ot_after == cyborg_ot_after, (
            f"JAX OT active={jax_ot_after} but CybORG OT active={cyborg_ot_after}"
        )

    def test_impact_fails_without_session_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target_h = self._find_ot_host_idx(const)
        assert target_h is not None
        target_hostname = sorted_hosts[target_h]

        impact = Impact(hostname=target_hostname, session=0, agent="red_agent_0")
        cyborg_obs = impact.execute(cyborg_state)
        cyborg_success = cyborg_obs.success

        impact_idx = encode_red_action("Impact", target_h, 0)
        new_state = apply_red_action(state, const, 0, impact_idx)
        jax_stopped = bool(new_state.ot_service_stopped[target_h])

        assert jax_stopped == False
        assert cyborg_success == False

    def test_impact_fails_on_non_ot_host_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state

        target_h = self._find_non_ot_host_idx(const)
        if target_h is None:
            pytest.skip("No non-OT host found")

        state, target_hostname = self._inject_privileged_session(
            cyborg_env, const, state, target_h
        )

        impact = Impact(hostname=target_hostname, session=0, agent="red_agent_0")
        cyborg_obs = impact.execute(cyborg_state)
        cyborg_success = cyborg_obs.success

        impact_idx = encode_red_action("Impact", target_h, 0)
        new_state = apply_red_action(state, const, 0, impact_idx)
        jax_stopped = bool(new_state.ot_service_stopped[target_h])

        assert jax_stopped == cyborg_success == False

    def test_impact_fails_with_user_priv_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        target_h = self._find_ot_host_idx(const)
        assert target_h is not None
        target_hostname = sorted_hosts[target_h]

        user_session = RedAbstractSession(
            hostname=target_hostname, username="ubuntu", agent="red_agent_0",
            parent=0, session_type=None, ident=None, pid=None,
        )
        cyborg_state.add_session(user_session)

        red_sessions = state.red_sessions.at[0, target_h].set(True)
        red_privilege = state.red_privilege.at[0, target_h].set(COMPROMISE_USER)
        state = state.replace(red_sessions=red_sessions, red_privilege=red_privilege)

        impact = Impact(hostname=target_hostname, session=0, agent="red_agent_0")
        cyborg_obs = impact.execute(cyborg_state)
        cyborg_success = cyborg_obs.success

        impact_idx = encode_red_action("Impact", target_h, 0)
        new_state = apply_red_action(state, const, 0, impact_idx)
        jax_stopped = bool(new_state.ot_service_stopped[target_h])
        jax_svc_still_on = bool(new_state.host_services[target_h, OT_SVC])

        assert jax_stopped == cyborg_success == False
        assert jax_svc_still_on == True

    def test_repeated_impact_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state

        target_h = self._find_ot_host_idx(const)
        assert target_h is not None

        state, target_hostname = self._inject_privileged_session(
            cyborg_env, const, state, target_h
        )

        impact = Impact(hostname=target_hostname, session=0, agent="red_agent_0")
        cyborg_obs1 = impact.execute(cyborg_state)
        assert cyborg_obs1.success == True

        impact_idx = encode_red_action("Impact", target_h, 0)
        state = apply_red_action(state, const, 0, impact_idx)
        assert bool(state.ot_service_stopped[target_h])
        assert not bool(state.host_services[target_h, OT_SVC])

        state = state.replace(
            red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32)
        )

        cyborg_obs2 = impact.execute(cyborg_state)
        cyborg_success2 = cyborg_obs2.success

        state2 = apply_red_action(state, const, 0, impact_idx)
        jax_changed = (
            bool(state2.host_services[target_h, OT_SVC])
            != bool(state.host_services[target_h, OT_SVC])
        )

        cyborg_host = cyborg_state.hosts[target_hostname]
        cyborg_ot_after2 = any(
            svc.active for sname, svc in cyborg_host.services.items()
            if sname == ProcessName.OTSERVICE
        )
        jax_ot_after2 = bool(state2.host_services[target_h, OT_SVC])

        assert jax_ot_after2 == cyborg_ot_after2

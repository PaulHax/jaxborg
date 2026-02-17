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
    RED_PRIVESC_START,
    RED_PRIVESC_END,
    RED_EXPLOIT_BLUEKEEP_END,
    ACTION_TYPE_PRIVESC,
)


try:
    from CybORG import CybORG
    from CybORG.Agents import SleepAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
    from CybORG.Simulator.Actions import DiscoverRemoteSystems, PrivilegeEscalate
    from CybORG.Simulator.Actions.AbstractActions.DiscoverNetworkServices import (
        AggressiveServiceDiscovery,
    )
    from CybORG.Simulator.Actions.AbstractActions.ExploitRemoteService import (
        ExploitRemoteService,
    )
    HAS_CYBORG = True
except ImportError:
    HAS_CYBORG = False

cyborg_required = pytest.mark.skipif(not HAS_CYBORG, reason="CybORG not installed")

SSH_SVC = SERVICE_IDS["SSHD"]


@pytest.fixture
def jax_const():
    return build_topology(jnp.array([42]), num_steps=500)


def _setup_exploited_state(jax_const, target_host):
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
    return state


def _find_exploitable_host(jax_const, exclude_start=True):
    for h in range(jax_const.num_hosts):
        if (jax_const.host_active[h]
                and not jax_const.host_is_router[h]
                and jax_const.initial_services[h, SSH_SVC]
                and jax_const.host_has_bruteforceable_user[h]):
            if exclude_start and h == int(jax_const.red_start_hosts[0]):
                continue
            return h
    return None


class TestPrivescEncoding:

    def test_privesc_range_starts_after_bluekeep(self):
        assert RED_PRIVESC_START == RED_EXPLOIT_BLUEKEEP_END

    def test_privesc_range_is_max_hosts_wide(self):
        assert RED_PRIVESC_END - RED_PRIVESC_START == GLOBAL_MAX_HOSTS

    def test_encode_per_host(self):
        for h in [0, 5, 50, GLOBAL_MAX_HOSTS - 1]:
            code = encode_red_action("PrivilegeEscalate", h, 0)
            assert code == RED_PRIVESC_START + h

    def test_decode_roundtrip(self, jax_const):
        for h in [0, 5, 50, GLOBAL_MAX_HOSTS - 1]:
            code = encode_red_action("PrivilegeEscalate", h, 0)
            action_type, target_subnet, target_host = decode_red_action(code, 0, jax_const)
            assert int(action_type) == ACTION_TYPE_PRIVESC
            assert int(target_subnet) == -1
            assert int(target_host) == h


class TestApplyPrivesc:

    def test_privesc_upgrades_to_privileged(self, jax_const):
        target = _find_exploitable_host(jax_const)
        if target is None:
            pytest.skip("No exploitable host found")

        state = _setup_exploited_state(jax_const, target)
        assert int(state.red_privilege[0, target]) == COMPROMISE_USER

        action_idx = encode_red_action("PrivilegeEscalate", target, 0)
        new_state = apply_red_action(state, jax_const, 0, action_idx)

        assert int(new_state.red_privilege[0, target]) == COMPROMISE_PRIVILEGED
        assert int(new_state.host_compromised[target]) == COMPROMISE_PRIVILEGED

    def test_privesc_sets_activity(self, jax_const):
        target = _find_exploitable_host(jax_const)
        if target is None:
            pytest.skip("No exploitable host found")

        state = _setup_exploited_state(jax_const, target)
        action_idx = encode_red_action("PrivilegeEscalate", target, 0)
        new_state = apply_red_action(state, jax_const, 0, action_idx)
        assert int(new_state.red_activity_this_step[target]) == ACTIVITY_EXPLOIT

    def test_privesc_fails_without_session(self, jax_const):
        target = _find_exploitable_host(jax_const)
        if target is None:
            pytest.skip("No exploitable host found")

        state = create_initial_state()
        state = state.replace(host_services=jnp.array(jax_const.initial_services))
        start_host = int(jax_const.red_start_hosts[0])
        red_sessions = state.red_sessions.at[0, start_host].set(True)
        state = state.replace(red_sessions=red_sessions)

        action_idx = encode_red_action("PrivilegeEscalate", target, 0)
        new_state = apply_red_action(state, jax_const, 0, action_idx)
        assert int(new_state.red_privilege[0, target]) == COMPROMISE_NONE
        assert int(new_state.host_compromised[target]) == COMPROMISE_NONE

    def test_privesc_noop_when_already_privileged(self, jax_const):
        target = _find_exploitable_host(jax_const)
        if target is None:
            pytest.skip("No exploitable host found")

        state = _setup_exploited_state(jax_const, target)
        action_idx = encode_red_action("PrivilegeEscalate", target, 0)
        state1 = apply_red_action(state, jax_const, 0, action_idx)
        assert int(state1.red_privilege[0, target]) == COMPROMISE_PRIVILEGED

        state1 = state1.replace(
            red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32)
        )
        state2 = apply_red_action(state1, jax_const, 0, action_idx)
        assert int(state2.red_activity_this_step[target]) == ACTIVITY_NONE

    def test_privesc_preserves_session(self, jax_const):
        target = _find_exploitable_host(jax_const)
        if target is None:
            pytest.skip("No exploitable host found")

        state = _setup_exploited_state(jax_const, target)
        action_idx = encode_red_action("PrivilegeEscalate", target, 0)
        new_state = apply_red_action(state, jax_const, 0, action_idx)
        assert bool(new_state.red_sessions[0, target])

    def test_privesc_does_not_affect_other_agents(self, jax_const):
        target = _find_exploitable_host(jax_const)
        if target is None:
            pytest.skip("No exploitable host found")

        state = _setup_exploited_state(jax_const, target)
        action_idx = encode_red_action("PrivilegeEscalate", target, 0)
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

    def test_privesc_only_affects_target(self, jax_const):
        target = _find_exploitable_host(jax_const)
        if target is None:
            pytest.skip("No exploitable host found")

        state = _setup_exploited_state(jax_const, target)
        action_idx = encode_red_action("PrivilegeEscalate", target, 0)
        new_state = apply_red_action(state, jax_const, 0, action_idx)

        for h in range(jax_const.num_hosts):
            if h == target:
                continue
            assert int(new_state.red_privilege[0, h]) == int(state.red_privilege[0, h])
            assert int(new_state.host_compromised[h]) == int(state.host_compromised[h])

    def test_jit_compatible(self, jax_const):
        target = _find_exploitable_host(jax_const)
        if target is None:
            pytest.skip("No exploitable host found")

        state = _setup_exploited_state(jax_const, target)
        action_idx = encode_red_action("PrivilegeEscalate", target, 0)
        jitted = jax.jit(apply_red_action, static_argnums=(2,))
        new_state = jitted(state, jax_const, 0, action_idx)
        assert int(new_state.red_privilege[0, target]) == COMPROMISE_PRIVILEGED


class TestPrivescChain:

    def test_discover_scan_exploit_privesc_chain(self, jax_const):
        target = _find_exploitable_host(jax_const)
        if target is None:
            pytest.skip("No exploitable host found")

        state = _setup_exploited_state(jax_const, target)
        assert int(state.red_privilege[0, target]) == COMPROMISE_USER
        assert bool(state.red_sessions[0, target])

        privesc_idx = encode_red_action("PrivilegeEscalate", target, 0)
        state = apply_red_action(state, jax_const, 0, privesc_idx)

        assert int(state.red_privilege[0, target]) == COMPROMISE_PRIVILEGED
        assert int(state.host_compromised[target]) == COMPROMISE_PRIVILEGED
        assert bool(state.red_sessions[0, target])
        assert int(state.red_activity_this_step[target]) == ACTIVITY_EXPLOIT


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

    def _find_and_exploit_host(self, cyborg_env, const, state):
        cyborg_state = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state.hosts.keys())

        subnet_name = "contractor_network_subnet"
        subnet_cidr = cyborg_state.subnet_name_to_cidr[subnet_name]
        sid = CYBORG_SUFFIX_TO_ID[subnet_name]

        discover_action = DiscoverRemoteSystems(
            subnet=subnet_cidr, session=0, agent="red_agent_0"
        )
        discover_action.duration = 1
        cyborg_env.step(agent="red_agent_0", action=discover_action)

        discover_idx = encode_red_action("DiscoverRemoteSystems", sid, 0)
        state = apply_red_action(state, const, 0, discover_idx)
        state = state.replace(
            red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32)
        )

        discovered_jax = np.array(state.red_discovered_hosts[0])
        exploitable = [
            h for h in range(const.num_hosts)
            if (discovered_jax[h]
                and not const.host_is_router[h]
                and const.initial_services[h, SSH_SVC]
                and const.host_has_bruteforceable_user[h]
                and h != int(const.red_start_hosts[0]))
        ]
        assert len(exploitable) > 0, "No exploitable hosts found"
        target_h = exploitable[0]
        target_hostname = sorted_hosts[target_h]

        target_ip = None
        for ip, hostname in cyborg_state.ip_addresses.items():
            if hostname == target_hostname:
                target_ip = ip
                break
        assert target_ip is not None

        scan_action = AggressiveServiceDiscovery(
            session=0, agent="red_agent_0", ip_address=target_ip
        )
        scan_action.duration = 1
        cyborg_env.step(agent="red_agent_0", action=scan_action)

        scan_idx = encode_red_action("DiscoverNetworkServices", target_h, 0)
        state = apply_red_action(state, const, 0, scan_idx)
        state = state.replace(
            red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32)
        )

        exploit_action = ExploitRemoteService(
            ip_address=target_ip, session=0, agent="red_agent_0"
        )
        exploit_action.duration = 1
        cyborg_result = cyborg_env.step(agent="red_agent_0", action=exploit_action)
        cyborg_exploit_success = cyborg_result.observation.get("success") == True

        exploit_idx = encode_red_action(
            "ExploitRemoteService_cc4SSHBruteForce", target_h, 0
        )
        state = apply_red_action(state, const, 0, exploit_idx)
        state = state.replace(
            red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32)
        )

        jax_exploit_success = bool(state.red_sessions[0, target_h])
        assert jax_exploit_success == cyborg_exploit_success
        assert cyborg_exploit_success, "CybORG exploit failed unexpectedly"

        return state, target_h, target_hostname, target_ip

    def test_privesc_success_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        state, target_h, target_hostname, target_ip = self._find_and_exploit_host(
            cyborg_env, const, state
        )

        privesc_action = PrivilegeEscalate(
            hostname=target_hostname, session=0, agent="red_agent_0"
        )
        privesc_action.duration = 1
        cyborg_result = cyborg_env.step(agent="red_agent_0", action=privesc_action)
        cyborg_success = cyborg_result.observation.get("success") == True

        privesc_idx = encode_red_action("PrivilegeEscalate", target_h, 0)
        new_state = apply_red_action(state, const, 0, privesc_idx)

        jax_success = int(new_state.red_privilege[0, target_h]) == COMPROMISE_PRIVILEGED

        assert jax_success == cyborg_success, (
            f"JAX privesc success={jax_success} but CybORG={cyborg_success} "
            f"for host {target_h} ({target_hostname})"
        )

        if cyborg_success:
            assert int(new_state.host_compromised[target_h]) == COMPROMISE_PRIVILEGED
            assert bool(new_state.red_sessions[0, target_h])

    def test_privesc_privilege_level_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        state, target_h, target_hostname, target_ip = self._find_and_exploit_host(
            cyborg_env, const, state
        )

        cyborg_state_obj = cyborg_env.environment_controller.state
        cyborg_host = cyborg_state_obj.hosts[target_hostname]
        sessions_before = [
            s for s in cyborg_state_obj.sessions["red_agent_0"].values()
            if s.hostname == target_hostname
        ]
        assert len(sessions_before) > 0
        cyborg_priv_before = any(s.has_privileged_access() for s in sessions_before)
        jax_priv_before = int(state.red_privilege[0, target_h])
        assert (jax_priv_before >= COMPROMISE_PRIVILEGED) == cyborg_priv_before

        privesc_action = PrivilegeEscalate(
            hostname=target_hostname, session=0, agent="red_agent_0"
        )
        privesc_action.duration = 1
        cyborg_env.step(agent="red_agent_0", action=privesc_action)

        privesc_idx = encode_red_action("PrivilegeEscalate", target_h, 0)
        new_state = apply_red_action(state, const, 0, privesc_idx)

        sessions_after = [
            s for s in cyborg_state_obj.sessions["red_agent_0"].values()
            if s.hostname == target_hostname
        ]
        cyborg_priv_after = any(s.has_privileged_access() for s in sessions_after)
        jax_priv_after = int(new_state.red_privilege[0, target_h]) >= COMPROMISE_PRIVILEGED

        assert jax_priv_after == cyborg_priv_after, (
            f"JAX privileged={jax_priv_after} but CybORG privileged={cyborg_priv_after} "
            f"for host {target_h} ({target_hostname})"
        )

    def test_privesc_fails_without_session_matches_cyborg(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state_obj = cyborg_env.environment_controller.state
        sorted_hosts = sorted(cyborg_state_obj.hosts.keys())

        start_host = int(const.red_start_hosts[0])
        target_h = None
        for h in range(const.num_hosts):
            if (const.host_active[h]
                    and not const.host_is_router[h]
                    and h != start_host
                    and not state.red_sessions[0, h]):
                target_h = h
                break
        assert target_h is not None
        target_hostname = sorted_hosts[target_h]

        privesc_action = PrivilegeEscalate(
            hostname=target_hostname, session=0, agent="red_agent_0"
        )
        privesc_action.duration = 1
        cyborg_result = cyborg_env.step(agent="red_agent_0", action=privesc_action)
        cyborg_success = cyborg_result.observation.get("success") == True

        privesc_idx = encode_red_action("PrivilegeEscalate", target_h, 0)
        new_state = apply_red_action(state, const, 0, privesc_idx)
        jax_success = int(new_state.red_privilege[0, target_h]) >= COMPROMISE_PRIVILEGED

        assert jax_success == cyborg_success == False

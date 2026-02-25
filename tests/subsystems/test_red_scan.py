import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Simulator.Actions import DiscoverRemoteSystems
from CybORG.Simulator.Actions.AbstractActions.DiscoverNetworkServices import (
    AggressiveServiceDiscovery,
)
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions import apply_red_action
from jaxborg.actions.encoding import (
    ACTION_TYPE_SCAN,
    RED_SCAN_START,
    decode_red_action,
    encode_red_action,
)
from jaxborg.actions.red_common import can_reach_subnet
from jaxborg.constants import (
    ACTIVITY_SCAN,
    GLOBAL_MAX_HOSTS,
    NUM_RED_AGENTS,
    NUM_SUBNETS,
)
from jaxborg.state import create_initial_state
from jaxborg.topology import CYBORG_SUFFIX_TO_ID, build_topology


@pytest.fixture
def jax_const():
    return build_topology(jnp.array([42]), num_steps=500)


@pytest.fixture
def jax_state_with_discovered(jax_const):
    state = create_initial_state()
    start_host = int(jax_const.red_start_hosts[0])
    state = state.replace(
        red_sessions=state.red_sessions.at[0, start_host].set(True),
        red_session_is_abstract=state.red_session_is_abstract.at[0, start_host].set(True),
    )

    start_subnet = int(jax_const.host_subnet[start_host])
    discover_idx = encode_red_action("DiscoverRemoteSystems", start_subnet, 0)
    state = apply_red_action(state, jax_const, 0, discover_idx, jax.random.PRNGKey(0))
    state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))
    return state


def _first_discovered_non_router(jax_const, state, agent_id=0):
    discovered = np.array(state.red_discovered_hosts[agent_id])
    for h in range(jax_const.num_hosts):
        if discovered[h] and not jax_const.host_is_router[h]:
            return h
    return None


class TestScanEncoding:
    def test_scan_encodes_per_host(self):
        for h in range(10):
            code = encode_red_action("DiscoverNetworkServices", h, 0)
            assert code == RED_SCAN_START + h

    def test_decode_scan_roundtrip(self, jax_const):
        for h in [0, 5, 50, GLOBAL_MAX_HOSTS - 1]:
            code = encode_red_action("DiscoverNetworkServices", h, 0)
            action_type, target_subnet, target_host = decode_red_action(code, 0, jax_const)
            assert int(action_type) == ACTION_TYPE_SCAN
            assert int(target_subnet) == -1
            assert int(target_host) == h

    def test_scan_range_does_not_overlap_discover(self):
        from jaxborg.actions.encoding import RED_DISCOVER_END

        assert RED_SCAN_START == RED_DISCOVER_END


class TestCanReachSubnet:
    def test_can_reach_own_subnet(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        start_host = int(jax_const.red_start_hosts[0])
        start_subnet = int(jax_const.host_subnet[start_host])
        assert bool(can_reach_subnet(state, jax_const, 0, start_subnet))

    def test_cannot_reach_when_no_session(self, jax_const):
        state = create_initial_state()
        assert not bool(can_reach_subnet(state, jax_const, 0, 0))

    def test_blocked_zone_prevents_reach(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        start_host = int(jax_const.red_start_hosts[0])
        start_subnet = int(jax_const.host_subnet[start_host])

        target_subnet = (start_subnet + 1) % NUM_SUBNETS
        blocked = state.blocked_zones.at[target_subnet, start_subnet].set(True)
        blocked_state = state.replace(blocked_zones=blocked)

        can_reach_unblocked = bool(can_reach_subnet(state, jax_const, 0, target_subnet))
        can_reach_blocked = bool(can_reach_subnet(blocked_state, jax_const, 0, target_subnet))

        if can_reach_unblocked:
            assert not can_reach_blocked


class TestApplyScan:
    def test_scan_discovered_host_marks_scanned(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        action_idx = encode_red_action("DiscoverNetworkServices", target, 0)
        new_state = apply_red_action(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        assert bool(new_state.red_scanned_hosts[0, target])

    def test_scan_sets_activity(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        action_idx = encode_red_action("DiscoverNetworkServices", target, 0)
        new_state = apply_red_action(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        assert int(new_state.red_activity_this_step[target]) == ACTIVITY_SCAN

    def test_scan_only_affects_target(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        action_idx = encode_red_action("DiscoverNetworkServices", target, 0)
        new_state = apply_red_action(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        for h in range(jax_const.num_hosts):
            if h != target:
                assert not bool(new_state.red_scanned_hosts[0, h])

    def test_scan_undiscovered_host_no_change(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        discovered = np.array(state.red_discovered_hosts[0])
        undiscovered = None
        for h in range(jax_const.num_hosts):
            if jax_const.host_active[h] and not discovered[h]:
                undiscovered = h
                break
        if undiscovered is None:
            pytest.skip("All hosts discovered")

        action_idx = encode_red_action("DiscoverNetworkServices", undiscovered, 0)
        new_state = apply_red_action(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        assert not bool(new_state.red_scanned_hosts[0, undiscovered])

    def test_scan_without_session_no_change(self, jax_const):
        state = create_initial_state()
        discovered = state.red_discovered_hosts.at[0, 5].set(True)
        state = state.replace(red_discovered_hosts=discovered)

        action_idx = encode_red_action("DiscoverNetworkServices", 5, 0)
        new_state = apply_red_action(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        assert not bool(new_state.red_scanned_hosts[0, 5])

    def test_scan_idempotent(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        action_idx = encode_red_action("DiscoverNetworkServices", target, 0)
        state1 = apply_red_action(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))
        state2 = apply_red_action(state1, jax_const, 0, action_idx, jax.random.PRNGKey(0))
        np.testing.assert_array_equal(
            np.array(state1.red_scanned_hosts),
            np.array(state2.red_scanned_hosts),
        )

    def test_scan_does_not_affect_other_agents(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        action_idx = encode_red_action("DiscoverNetworkServices", target, 0)
        new_state = apply_red_action(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        for agent in range(1, NUM_RED_AGENTS):
            np.testing.assert_array_equal(
                np.array(new_state.red_scanned_hosts[agent]),
                np.array(state.red_scanned_hosts[agent]),
            )

    def test_scan_does_not_change_discovered(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        action_idx = encode_red_action("DiscoverNetworkServices", target, 0)
        new_state = apply_red_action(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        np.testing.assert_array_equal(
            np.array(new_state.red_discovered_hosts),
            np.array(state.red_discovered_hosts),
        )

    def test_scan_blocked_zone_no_change(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        start_host = int(jax_const.red_start_hosts[0])
        int(jax_const.host_subnet[start_host])
        int(jax_const.host_subnet[target])

        blocked = jnp.ones((NUM_SUBNETS, NUM_SUBNETS), dtype=jnp.bool_)
        state_blocked = state.replace(blocked_zones=blocked)

        action_idx = encode_red_action("DiscoverNetworkServices", target, 0)
        new_state = apply_red_action(state_blocked, jax_const, 0, action_idx, jax.random.PRNGKey(0))

        assert not bool(new_state.red_scanned_hosts[0, target])

    def test_jit_compatible(self, jax_const, jax_state_with_discovered):
        state = jax_state_with_discovered
        target = _first_discovered_non_router(jax_const, state)
        assert target is not None

        action_idx = encode_red_action("DiscoverNetworkServices", target, 0)
        jitted = jax.jit(apply_red_action, static_argnums=(2,))
        new_state = jitted(state, jax_const, 0, action_idx, jax.random.PRNGKey(0))
        assert bool(new_state.red_scanned_hosts[0, target])


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
        start_host = int(const.red_start_hosts[0])
        state = state.replace(
            red_sessions=state.red_sessions.at[0, start_host].set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[0, start_host].set(True),
        )
        return cyborg_env, const, state

    def test_scan_host_services_detected(self, cyborg_and_jax):
        cyborg_env, const, state = cyborg_and_jax
        cyborg_state = cyborg_env.environment_controller.state

        subnet_name = "contractor_network_subnet"
        subnet_cidr = cyborg_state.subnet_name_to_cidr[subnet_name]
        sid = CYBORG_SUFFIX_TO_ID[subnet_name]

        discover_action = DiscoverRemoteSystems(subnet=subnet_cidr, session=0, agent="red_agent_0")
        discover_action.duration = 1
        cyborg_env.step(agent="red_agent_0", action=discover_action)

        discover_idx = encode_red_action("DiscoverRemoteSystems", sid, 0)
        state = apply_red_action(state, const, 0, discover_idx, jax.random.PRNGKey(0))

        sorted_hosts = sorted(cyborg_state.hosts.keys())
        discovered_jax = np.array(state.red_discovered_hosts[0])
        discovered_hosts = [h for h in range(const.num_hosts) if discovered_jax[h] and not const.host_is_router[h]]
        assert len(discovered_hosts) > 0

        target_h = discovered_hosts[0]
        target_hostname = sorted_hosts[target_h]
        target_ip = None
        for ip, hostname in cyborg_state.ip_addresses.items():
            if hostname == target_hostname:
                target_ip = ip
                break
        assert target_ip is not None

        scan_action = AggressiveServiceDiscovery(session=0, agent="red_agent_0", ip_address=target_ip)
        scan_action.duration = 1
        cyborg_env.step(agent="red_agent_0", action=scan_action)

        scan_idx = encode_red_action("DiscoverNetworkServices", target_h, 0)
        state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))
        new_state = apply_red_action(state, const, 0, scan_idx, jax.random.PRNGKey(0))

        assert bool(new_state.red_scanned_hosts[0, target_h]), (
            f"JAX should mark host {target_h} ({target_hostname}) as scanned"
        )


class TestScanRequiresAbstractSession:
    """CybORG gates DiscoverNetworkServices on RedAbstractSession.

    Sessions from green phishing reassignment are plain Sessions that cannot scan.
    JAX must replicate this by tracking red_session_is_abstract.
    """

    def test_scan_fails_without_abstract_session(self):
        """Scan must fail when agent only has non-abstract sessions (from phishing)."""
        from jaxborg.topology import build_topology

        const = build_topology(jnp.array([42]), num_steps=500)
        state = create_initial_state()

        agent_id = 0
        start_host = int(const.red_start_hosts[agent_id])
        target_subnet = int(const.host_subnet[start_host])

        # Give agent a session but NOT an abstract one (simulating phishing reassignment)
        state = state.replace(
            red_sessions=state.red_sessions.at[agent_id, start_host].set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[agent_id, start_host].set(False),
        )

        # Discover hosts first
        discover_idx = encode_red_action("DiscoverRemoteSystems", target_subnet, agent_id)
        state = apply_red_action(state, const, agent_id, discover_idx, jax.random.PRNGKey(0))
        state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))

        target = _first_discovered_non_router(const, state, agent_id)
        assert target is not None, "Need at least one discovered host to scan"

        scan_idx = encode_red_action("DiscoverNetworkServices", target, agent_id)
        new_state = apply_red_action(state, const, agent_id, scan_idx, jax.random.PRNGKey(1))

        assert not bool(new_state.red_scanned_hosts[agent_id, target]), (
            "Scan must fail when agent has no abstract session (CybORG RedAbstractSession check)"
        )

    def test_scan_succeeds_with_abstract_session(self):
        """Scan succeeds when agent has an abstract session (from exploit)."""
        from jaxborg.topology import build_topology

        const = build_topology(jnp.array([42]), num_steps=500)
        state = create_initial_state()

        agent_id = 0
        start_host = int(const.red_start_hosts[agent_id])
        target_subnet = int(const.host_subnet[start_host])

        # Give agent an abstract session (from exploit)
        state = state.replace(
            red_sessions=state.red_sessions.at[agent_id, start_host].set(True),
            red_session_is_abstract=state.red_session_is_abstract.at[agent_id, start_host].set(True),
        )

        # Discover hosts
        discover_idx = encode_red_action("DiscoverRemoteSystems", target_subnet, agent_id)
        state = apply_red_action(state, const, agent_id, discover_idx, jax.random.PRNGKey(0))
        state = state.replace(red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32))

        target = _first_discovered_non_router(const, state, agent_id)
        assert target is not None

        scan_idx = encode_red_action("DiscoverNetworkServices", target, agent_id)
        new_state = apply_red_action(state, const, agent_id, scan_idx, jax.random.PRNGKey(1))

        assert bool(new_state.red_scanned_hosts[agent_id, target]), (
            "Scan should succeed when agent has an abstract session"
        )

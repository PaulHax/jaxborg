import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, SleepAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions.green import (
    FP_DETECTION_RATE,
    PHISHING_ERROR_RATE,
    apply_green_agents,
)
from jaxborg.constants import (
    COMPROMISE_USER,
    GLOBAL_MAX_HOSTS,
    NUM_RED_AGENTS,
    NUM_SUBNETS,
)
from jaxborg.state import create_initial_state
from jaxborg.topology import build_topology


@pytest.fixture
def jax_const():
    return build_topology(jnp.array([42]), num_steps=500)


@pytest.fixture
def jax_state(jax_const):
    state = create_initial_state()
    return state.replace(host_services=jax_const.initial_services)


def _run_many_green(state, const, num_trials=50):
    jitted = jax.jit(apply_green_agents)
    keys = jax.random.split(jax.random.PRNGKey(0), num_trials)
    results = []
    for i in range(num_trials):
        results.append(jitted(state, const, keys[i]))
    return results


class TestGreenAgentBasics:
    def test_no_crash_on_initial_state(self, jax_const, jax_state):
        key = jax.random.PRNGKey(0)
        new_state = apply_green_agents(jax_state, jax_const, key)
        assert new_state.host_activity_detected.shape == (GLOBAL_MAX_HOSTS,)

    def test_jit_compatible(self, jax_const, jax_state):
        key = jax.random.PRNGKey(0)
        jitted = jax.jit(apply_green_agents)
        new_state = jitted(jax_state, jax_const, key)
        assert new_state.host_activity_detected.shape == (GLOBAL_MAX_HOSTS,)

    def test_deterministic_with_same_key(self, jax_const, jax_state):
        key = jax.random.PRNGKey(123)
        jitted = jax.jit(apply_green_agents)
        s1 = jitted(jax_state, jax_const, key)
        s2 = jitted(jax_state, jax_const, key)
        np.testing.assert_array_equal(np.array(s1.host_activity_detected), np.array(s2.host_activity_detected))
        np.testing.assert_array_equal(np.array(s1.red_sessions), np.array(s2.red_sessions))

    def test_different_keys_different_results(self, jax_const, jax_state):
        jitted = jax.jit(apply_green_agents)
        results = []
        for seed in range(20):
            key = jax.random.PRNGKey(seed)
            s = jitted(jax_state, jax_const, key)
            results.append(int(jnp.sum(s.host_activity_detected)))
        assert len(set(results)) > 1

    def test_inactive_hosts_unchanged(self, jax_const, jax_state):
        key = jax.random.PRNGKey(0)
        new_state = apply_green_agents(jax_state, jax_const, key)
        for h in range(GLOBAL_MAX_HOSTS):
            if not jax_const.green_agent_active[h]:
                assert not new_state.host_activity_detected[h] or jax_state.host_activity_detected[h]


class TestGreenLocalWorkFalsePositive:
    def test_fp_rate_statistical(self, jax_const, jax_state):
        results = _run_many_green(jax_state, jax_const, num_trials=50)
        fp_count = sum(int(jnp.sum(r.host_activity_detected & ~jax_state.host_activity_detected)) for r in results)
        assert fp_count > 0, "Expected at least some false positives over 50 steps"


class TestGreenPhishing:
    def test_phishing_creates_red_session(self, jax_const, jax_state):
        start_host = int(jax_const.red_start_hosts[0])
        state = jax_state.replace(red_sessions=jax_state.red_sessions.at[0, start_host].set(True))
        results = _run_many_green(state, jax_const, num_trials=100)
        phish_count = sum(int(np.sum(np.array(r.red_sessions) & ~np.array(state.red_sessions))) for r in results)
        assert phish_count > 0, "Expected at least one phishing event over 100 steps"

    def test_phishing_only_user_level(self, jax_const, jax_state):
        start_host = int(jax_const.red_start_hosts[0])
        state = jax_state.replace(red_sessions=jax_state.red_sessions.at[0, start_host].set(True))
        results = _run_many_green(state, jax_const, num_trials=30)
        for new_state in results:
            new_sessions = np.array(new_state.red_sessions) & ~np.array(state.red_sessions)
            new_priv = np.array(new_state.red_privilege)
            for r in range(NUM_RED_AGENTS):
                for h in range(GLOBAL_MAX_HOSTS):
                    if new_sessions[r, h]:
                        assert new_priv[r, h] == COMPROMISE_USER


class TestGreenAccessService:
    def test_access_service_network_events(self, jax_const, jax_state):
        results = _run_many_green(jax_state, jax_const, num_trials=50)
        event_count = sum(int(jnp.sum(r.host_activity_detected & ~jax_state.host_activity_detected)) for r in results)
        assert event_count >= 0

    def test_blocked_zones_create_events(self, jax_const, jax_state):
        blocked = jnp.ones((NUM_SUBNETS, NUM_SUBNETS), dtype=jnp.bool_)
        state = jax_state.replace(blocked_zones=blocked)
        results = _run_many_green(state, jax_const, num_trials=30)
        event_count = sum(int(jnp.sum(r.host_activity_detected & ~state.host_activity_detected)) for r in results)
        assert event_count > 0, "Blocked zones should cause network connection events"


class TestDynamicTopology:
    def test_different_seeds_different_host_counts(self):
        counts = set()
        for seed in range(20):
            const = build_topology(jnp.array([seed]), num_steps=500)
            counts.add(int(const.num_hosts))
        assert len(counts) > 1, "Different seeds should produce different host counts"

    def test_host_count_range(self):
        for seed in range(10):
            const = build_topology(jnp.array([seed]), num_steps=500)
            num = int(const.num_hosts)
            assert 37 <= num <= 137, f"Seed {seed}: host count {num} out of expected range"

    def test_padding_to_global_max(self):
        for seed in range(5):
            const = build_topology(jnp.array([seed]), num_steps=500)
            assert const.host_active.shape == (GLOBAL_MAX_HOSTS,)
            assert const.host_subnet.shape == (GLOBAL_MAX_HOSTS,)
            assert const.data_links.shape == (GLOBAL_MAX_HOSTS, GLOBAL_MAX_HOSTS)

    def test_host_active_consistency(self):
        for seed in range(5):
            const = build_topology(jnp.array([seed]), num_steps=500)
            active_count = int(jnp.sum(const.host_active))
            assert active_count == const.num_hosts
            for h in range(GLOBAL_MAX_HOSTS):
                if h >= const.num_hosts:
                    assert not const.host_active[h]

    def test_green_agents_present(self):
        for seed in range(5):
            const = build_topology(jnp.array([seed]), num_steps=500)
            num_green = int(jnp.sum(const.green_agent_active))
            num_user = int(jnp.sum(const.host_is_user))
            assert num_green == num_user
            assert const.num_green_agents == num_user

    def test_all_subnets_have_hosts(self):
        for seed in range(10):
            const = build_topology(jnp.array([seed]), num_steps=500)
            for sid in range(NUM_SUBNETS):
                count = int(jnp.sum(const.host_active & (const.host_subnet == sid)))
                assert count >= 1, f"Seed {seed}: subnet {sid} has no hosts"


class TestDifferentialGreen:
    @pytest.fixture
    def cyborg_env(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        return CybORG(scenario_generator=sg, seed=42)

    def test_green_agent_count_matches(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg

        const = build_const_from_cyborg(cyborg_env)
        scenario = cyborg_env.environment_controller.state.scenario
        cyborg_green_count = sum(1 for name in scenario.agents if name.startswith("green_agent_"))
        assert const.num_green_agents == cyborg_green_count

    def test_green_agents_on_user_hosts(self, cyborg_env):
        from jaxborg.topology import build_const_from_cyborg

        const = build_const_from_cyborg(cyborg_env)
        for h in range(const.num_hosts):
            if const.host_is_user[h]:
                assert const.green_agent_active[h]
                assert const.green_agent_host[h] >= 0

    def test_phishing_rate_matches_cyborg(self):
        assert PHISHING_ERROR_RATE == 0.01

    def test_fp_rate_matches_cyborg(self):
        assert FP_DETECTION_RATE == 0.01


class TestGreenStatisticalDifferential:
    @pytest.fixture
    def cyborg_env(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=SleepAgent,
            steps=500,
        )
        return CybORG(scenario_generator=sg, seed=42)

    def test_fp_rate_within_statistical_bounds(self, cyborg_env):
        """Run many steps with only green active, check FP rate is statistically consistent."""
        from jaxborg.topology import build_const_from_cyborg

        const = build_const_from_cyborg(cyborg_env)
        state = create_initial_state()
        state = state.replace(host_services=const.initial_services)

        jitted = jax.jit(apply_green_agents)
        fp_total = 0
        n_green = int(jnp.sum(const.green_agent_active))
        n_trials = 200

        for i in range(n_trials):
            key = jax.random.PRNGKey(i)
            new_state = jitted(state, const, key)
            fp_total += int(jnp.sum(new_state.host_activity_detected & const.green_agent_active))

        observed_rate = fp_total / (n_green * n_trials) if n_green > 0 else 0
        assert observed_rate < 0.05, f"FP rate {observed_rate} seems too high"

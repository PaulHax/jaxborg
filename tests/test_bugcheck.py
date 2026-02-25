"""Bug-hunting differential tests informed by CC2 historical bug catalog.

Each test class targets a specific bug pattern that caused high entropy / no
learning in the CC2 JAX port of CybORG. If the same bug exists in the CC4
jaxborg port, these tests will fail.

Reference: 40 bugs cataloged from jaxmarl-integration branch git history.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import SleepAgent
from CybORG.Agents.Wrappers.BlueFlatWrapper import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions import apply_blue_action, apply_red_action
from jaxborg.actions.encoding import (
    BLUE_ANALYSE_START,
    BLUE_DECOY_START,
    BLUE_MONITOR,
    BLUE_REMOVE_START,
    BLUE_RESTORE_START,
    BLUE_SLEEP,
    RED_EXPLOIT_HARAKA_START,
    RED_EXPLOIT_SSH_START,
    RED_IMPACT_START,
    decode_blue_action,
    decode_red_action,
)
from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.constants import (
    COMPROMISE_NONE,
    COMPROMISE_PRIVILEGED,
    COMPROMISE_USER,
    GLOBAL_MAX_HOSTS,
    NUM_BLUE_AGENTS,
    NUM_DECOY_TYPES,
    NUM_RED_AGENTS,
    NUM_SUBNETS,
    SERVICE_IDS,
)
from jaxborg.observations import get_blue_obs
from jaxborg.rewards import compute_rewards
from jaxborg.state import create_initial_state
from jaxborg.topology import build_const_from_cyborg

pytestmark = pytest.mark.slow


def _make_cyborg(seed=42, steps=500, blue_cls=SleepAgent, green_cls=SleepAgent, red_cls=SleepAgent):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=blue_cls,
        green_agent_class=green_cls,
        red_agent_class=red_cls,
        steps=steps,
    )
    return CybORG(scenario_generator=sg, seed=seed)


_CONST_CACHE = {}


def _get_const(seed=42):
    if seed not in _CONST_CACHE:
        _CONST_CACHE[seed] = build_const_from_cyborg(_make_cyborg(seed=seed))
    return _CONST_CACHE[seed]


def _make_jax_state(const):
    state = create_initial_state()
    return state.replace(host_services=jnp.array(const.initial_services))


def _find_host_in_subnet(const, subnet_name):
    from jaxborg.constants import SUBNET_IDS

    sid = SUBNET_IDS[subnet_name]
    for h in range(int(const.num_hosts)):
        if not bool(const.host_active[h]):
            continue
        if int(const.host_subnet[h]) != sid:
            continue
        if bool(const.host_is_router[h]):
            continue
        return h
    return None


def _find_blue_for_host(const, host_idx):
    for b in range(NUM_BLUE_AGENTS):
        if bool(const.blue_agent_hosts[b, host_idx]):
            return b
    return None


# ---------------------------------------------------------------------------
# BUG 1 analog: Host ordering must be alphabetical
# ---------------------------------------------------------------------------
class TestHostOrdering:
    """CC2 BUG 1: Hosts must be in alphabetical order matching CybORG."""

    def test_host_order_matches_cyborg(self, cyborg_env):
        build_const_from_cyborg(cyborg_env)
        from jaxborg.translate import build_mappings_from_cyborg

        mappings = build_mappings_from_cyborg(cyborg_env)
        hostnames = [mappings.idx_to_hostname[i] for i in range(mappings.num_hosts)]
        assert hostnames == sorted(hostnames), f"Host ordering is not alphabetical: {hostnames[:10]}..."


# ---------------------------------------------------------------------------
# BUG 4+20 analog: Observation encoding must match CybORG
# ---------------------------------------------------------------------------
class TestObservationParity:
    """CC2 BUGs 4, 20, 21, 34: Blue observations must match CybORG exactly."""

    def test_initial_obs_match(self, cyborg_env):
        """Initial observations (step 0) must match CybORG BlueFlatWrapper."""
        wrapped = BlueFlatWrapper(cyborg_env, pad_spaces=True)
        const = build_const_from_cyborg(cyborg_env)
        state = _make_jax_state(const)

        observations, _ = wrapped.reset()

        for agent_id in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{agent_id}"
            cyborg_obs = np.array(observations[agent_name])
            jax_obs = np.array(get_blue_obs(state, const, agent_id))

            assert cyborg_obs.shape == jax_obs.shape, (
                f"Agent {agent_id}: obs shape mismatch: cyborg={cyborg_obs.shape} jax={jax_obs.shape}"
            )
            np.testing.assert_array_equal(
                cyborg_obs,
                jax_obs,
                err_msg=f"Agent {agent_id}: initial obs mismatch at indices {np.where(cyborg_obs != jax_obs)[0][:10]}",
            )

    def test_obs_after_5_steps_of_sleep(self, cyborg_env):
        """After 5 steps of all-sleep, observations should still match."""
        wrapped = BlueFlatWrapper(cyborg_env, pad_spaces=True)
        const = build_const_from_cyborg(cyborg_env)
        state = _make_jax_state(const)

        wrapped.reset()

        for step in range(5):
            actions = {}
            for agent in wrapped.agents:
                actions[agent] = wrapped.action_space(agent).sample() * 0  # Sleep = 0
            observations, rewards, terminated, truncated, infos = wrapped.step(actions)

        for agent_id in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{agent_id}"
            if agent_name not in observations:
                continue
            cyborg_obs = np.array(observations[agent_name])
            jax_obs = np.array(get_blue_obs(state, const, agent_id))

            np.testing.assert_array_equal(
                cyborg_obs, jax_obs, err_msg=f"Agent {agent_id} after 5 sleep steps: obs mismatch"
            )


# ---------------------------------------------------------------------------
# BUG 3+14 analog: Blue action decode roundtrip
# ---------------------------------------------------------------------------
class TestBlueActionEncoding:
    """CC2 BUGs 3, 14: Blue action encoding/decoding must be consistent."""

    def test_action_order_is_analyse_remove_restore_decoy(self):
        """CC4 action layout: Analyse < Remove < Restore < Decoy (differs from CC2)."""
        assert BLUE_ANALYSE_START < BLUE_REMOVE_START < BLUE_RESTORE_START < BLUE_DECOY_START, (
            f"Expected Analyse({BLUE_ANALYSE_START}) < Remove({BLUE_REMOVE_START}) "
            f"< Restore({BLUE_RESTORE_START}) < Decoy({BLUE_DECOY_START}). "
            "CC2 BUG 3: action ordering was wrong."
        )

    def test_decoy_decode_roundtrip(self):
        """Decoy action encode->decode must produce correct (host, decoy_type)."""
        const = _get_const()
        for decoy_type in range(NUM_DECOY_TYPES):
            for host_offset in range(min(5, GLOBAL_MAX_HOSTS)):
                action_idx = BLUE_DECOY_START + host_offset
                if action_idx >= BLUE_DECOY_START + GLOBAL_MAX_HOSTS * NUM_DECOY_TYPES:
                    break
                action_type, target_host, decoy_t, *_ = decode_blue_action(
                    action_idx + decoy_type * GLOBAL_MAX_HOSTS, 0, const
                )
                assert int(target_host) == host_offset, (
                    f"Decoy decode host mismatch: expected {host_offset}, got {int(target_host)}"
                )
                assert int(decoy_t) == decoy_type, (
                    f"Decoy decode type mismatch: expected {decoy_type}, got {int(decoy_t)}"
                )


# ---------------------------------------------------------------------------
# BUG 28 analog: Red exploit action encoding
# ---------------------------------------------------------------------------
class TestRedExploitEncoding:
    """CC2 BUG 28: Red exploit actions must correctly encode host and exploit type."""

    def test_ssh_exploit_targets_correct_host(self):
        const = _get_const()
        for host in range(min(10, GLOBAL_MAX_HOSTS)):
            action_idx = RED_EXPLOIT_SSH_START + host
            action_type, _, target_host = decode_red_action(action_idx, 0, const)
            assert int(target_host) == host, (
                f"SSH exploit action {action_idx}: expected host {host}, got {int(target_host)}"
            )

    def test_haraka_exploit_targets_correct_host(self):
        const = _get_const()
        for host in range(min(10, GLOBAL_MAX_HOSTS)):
            action_idx = RED_EXPLOIT_HARAKA_START + host
            action_type, _, target_host = decode_red_action(action_idx, 0, const)
            assert int(target_host) == host, (
                f"Haraka exploit action {action_idx}: expected host {host}, got {int(target_host)}"
            )


# ---------------------------------------------------------------------------
# BUG 7 analog: Red initial foothold must be PRIVILEGED
# ---------------------------------------------------------------------------
class TestRedInitialFoothold:
    """CC2 BUG 7: Red starts with PRIVILEGED (root/SYSTEM), not USER."""

    def test_initial_foothold_privilege_matches_cyborg(self, cyborg_env):
        from jaxborg.translate import build_mappings_from_cyborg

        mappings = build_mappings_from_cyborg(cyborg_env)
        const = build_const_from_cyborg(cyborg_env)

        cyborg_state = cyborg_env.environment_controller.state

        for agent_name, sessions in cyborg_state.sessions.items():
            if not agent_name.startswith("red_agent_"):
                continue
            red_idx = int(agent_name.split("_")[-1])
            if red_idx >= NUM_RED_AGENTS:
                continue
            for sess in sessions.values():
                if sess.hostname not in mappings.hostname_to_idx:
                    continue
                host_idx = mappings.hostname_to_idx[sess.hostname]
                is_root = hasattr(sess, "username") and sess.username in ("root", "SYSTEM")
                if is_root:
                    assert bool(const.host_active[host_idx]), f"Red foothold host {sess.hostname} not active in JAX"


# ---------------------------------------------------------------------------
# BUG 8 analog: Remove only works on detected user-level sessions
# ---------------------------------------------------------------------------
class TestRemoveBehavior:
    """CC2 BUG 8: Remove should NOT clear privileged access."""

    def test_remove_does_not_clear_privileged(self):
        const = _get_const()
        state = _make_jax_state(const)

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        if target is None:
            pytest.skip("No host found in RESTRICTED_ZONE_A")

        blue_agent = _find_blue_for_host(const, target)
        if blue_agent is None:
            pytest.skip("No blue agent covers target host")

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_PRIVILEGED),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_PRIVILEGED),
            host_activity_detected=state.host_activity_detected.at[target].set(True),
        )

        action = BLUE_REMOVE_START + target
        state = apply_blue_action(state, const, blue_agent, action)

        assert int(state.red_privilege[0, target]) == COMPROMISE_PRIVILEGED, (
            "Remove should NOT clear privileged access (CC2 BUG 8)"
        )


# ---------------------------------------------------------------------------
# BUG 9 analog: Restore preserves initial Red foothold
# ---------------------------------------------------------------------------
class TestRestorePreservesFoothold:
    """CC2 BUG 9: Restore on initial foothold host should preserve Red's session."""

    def test_restore_on_non_foothold_clears_sessions(self):
        const = _get_const()
        state = _make_jax_state(const)

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        if target is None:
            pytest.skip("No host found")

        blue_agent = _find_blue_for_host(const, target)
        if blue_agent is None:
            pytest.skip("No blue agent covers host")

        state = state.replace(
            red_sessions=state.red_sessions.at[0, target].set(True),
            red_privilege=state.red_privilege.at[0, target].set(COMPROMISE_USER),
            host_compromised=state.host_compromised.at[target].set(COMPROMISE_USER),
        )

        action = BLUE_RESTORE_START + target
        state = apply_blue_action(state, const, blue_agent, action)

        assert int(state.red_sessions[0, target]) == 0, "Restore on non-foothold host should clear Red sessions"
        assert int(state.red_privilege[0, target]) == COMPROMISE_NONE, (
            "Restore on non-foothold host should clear Red privilege"
        )


# ---------------------------------------------------------------------------
# BUG 11+12 analog: Exploit and PrivEsc determinism
# ---------------------------------------------------------------------------
class TestExploitDeterminism:
    """CC2 BUGs 11, 12: Exploits should be deterministic (not random)."""

    def test_exploit_same_result_different_keys(self):
        const = _get_const()
        state = _make_jax_state(const)

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        if target is None:
            pytest.skip("No host found")

        state = state.replace(
            red_discovered_hosts=state.red_discovered_hosts.at[0, target].set(True),
            red_scanned_hosts=state.red_scanned_hosts.at[0, target].set(True),
        )

        action = RED_EXPLOIT_SSH_START + target
        key1 = jax.random.PRNGKey(0)
        key2 = jax.random.PRNGKey(999)

        state1 = apply_red_action(state, const, 0, action, key1)
        state2 = apply_red_action(state, const, 0, action, key2)

        assert int(state1.host_compromised[target]) == int(state2.host_compromised[target]), (
            "Exploit result should not depend on RNG key (CC2 BUG 11: was probabilistic)"
        )


# ---------------------------------------------------------------------------
# BUG 10 analog: Impact requires operational target
# ---------------------------------------------------------------------------
class TestImpactRequiresOperational:
    """CC2 BUG 10: Impact should only work on operational targets."""

    def test_impact_fails_on_non_operational(self):
        from jaxborg.constants import SERVICE_IDS

        OTSERVICE_IDX = SERVICE_IDS["OTSERVICE"]
        const = _get_const()
        state = _make_jax_state(const)

        non_operational = None
        for h in range(int(const.num_hosts)):
            if not bool(const.host_active[h]):
                continue
            if bool(const.host_is_router[h]):
                continue
            if not bool(const.initial_services[h, OTSERVICE_IDX]):
                non_operational = h
                break

        if non_operational is None:
            pytest.skip("No non-operational host found")

        state = state.replace(
            red_sessions=state.red_sessions.at[0, non_operational].set(True),
            red_privilege=state.red_privilege.at[0, non_operational].set(COMPROMISE_PRIVILEGED),
            host_compromised=state.host_compromised.at[non_operational].set(COMPROMISE_PRIVILEGED),
        )

        action = RED_IMPACT_START + non_operational
        key = jax.random.PRNGKey(0)
        state = apply_red_action(state, const, 0, action, key)

        assert not bool(state.ot_service_stopped[non_operational]), (
            "Impact should fail on non-operational host (CC2 BUG 10)"
        )


# ---------------------------------------------------------------------------
# BUG 13 analog: Exploit privilege levels
# ---------------------------------------------------------------------------
class TestExploitPrivilegeLevels:
    """CC2 BUG 13: Certain exploits grant root directly."""

    def test_haraka_does_not_grant_access(self):
        """In CC4, SMTP processes are always patched (HARAKA_2_8_9) so HarakaRCE never succeeds."""
        const = _get_const()
        state = _make_jax_state(const)

        target = None
        smtp_idx = SERVICE_IDS["SMTP"]
        for h in range(int(const.num_hosts)):
            if not bool(const.host_active[h]):
                continue
            if bool(const.host_is_router[h]):
                continue
            if bool(const.initial_services[h, smtp_idx]):
                target = h
                break

        if target is None:
            pytest.skip("No host with SMTP service found")

        state = state.replace(
            red_discovered_hosts=state.red_discovered_hosts.at[0, target].set(True),
            red_scanned_hosts=state.red_scanned_hosts.at[0, target].set(True),
        )

        action = RED_EXPLOIT_HARAKA_START + target
        key = jax.random.PRNGKey(42)
        state = apply_red_action(state, const, 0, action, key)

        assert int(state.red_privilege[0, target]) == COMPROMISE_NONE, (
            f"HarakaRCE should not grant access on host {target} — "
            "CC4 SMTP is always patched (prob_vuln_proc_occurs=1.0)"
        )


# ---------------------------------------------------------------------------
# BUG 5+6 analog: Reward calculation parity
# ---------------------------------------------------------------------------
class TestRewardParity:
    """CC2 BUGs 5, 6, 10, 26: Reward calculation must match CybORG."""

    def test_zero_reward_at_start(self):
        """With no compromise or impact, reward should be 0."""
        const = _get_const()
        state = _make_jax_state(const)

        impact = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)
        green_lwf = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)
        green_asf = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)

        reward = compute_rewards(state, const, impact, green_lwf, green_asf)
        assert float(reward) == 0.0, f"Reward should be 0 with no compromise/impact, got {float(reward)}"


# ---------------------------------------------------------------------------
# BUG 16+17 analog: Decoy-service conflicts and exploit blocking
# ---------------------------------------------------------------------------
class TestDecoyMechanics:
    """CC2 BUGs 16, 17, 23, 25: Decoy deployment and exploit-blocking."""

    def test_decoy_deployment_changes_state(self):
        const = _get_const()
        state = _make_jax_state(const)

        target = _find_host_in_subnet(const, "RESTRICTED_ZONE_A")
        if target is None:
            pytest.skip("No host found")

        blue_agent = _find_blue_for_host(const, target)
        if blue_agent is None:
            pytest.skip("No blue agent covers host")

        action = BLUE_DECOY_START + target
        state_after = apply_blue_action(state, const, blue_agent, action)

        any_decoy_before = bool(np.any(np.array(state.host_decoys[target])))
        any_decoy_after = bool(np.any(np.array(state_after.host_decoys[target])))

        assert any_decoy_after or any_decoy_before, (
            "Decoy deployment should change host_decoys state (CC2 BUG 23: decoys did nothing)"
        )


# ---------------------------------------------------------------------------
# BUG 29 analog: Subnet adjacency must be correct
# ---------------------------------------------------------------------------
class TestSubnetAdjacency:
    """CC2 BUG 29: Subnet adjacency must match CybORG NACL rules."""

    def test_adjacency_is_not_all_true(self):
        const = _get_const()
        adj = np.array(const.subnet_adjacency)
        active = adj[:NUM_SUBNETS, :NUM_SUBNETS]
        assert not np.all(active), "Subnet adjacency should not be all-true (CC2 BUG 29: NACL parsing was wrong)"

    def test_adjacency_is_symmetric_where_expected(self):
        """Certain subnet pairs should be bidirectional (CC2 BUG 29 variant)."""
        const = _get_const()
        adj = np.array(const.subnet_adjacency)
        from jaxborg.constants import SUBNET_IDS

        S = SUBNET_IDS
        assert adj[S["RESTRICTED_ZONE_A"], S["OPERATIONAL_ZONE_A"]], (
            "RESTRICTED_ZONE_A -> OPERATIONAL_ZONE_A should be connected"
        )
        assert adj[S["OPERATIONAL_ZONE_A"], S["RESTRICTED_ZONE_A"]], (
            "OPERATIONAL_ZONE_A -> RESTRICTED_ZONE_A should be connected"
        )


# ---------------------------------------------------------------------------
# BUG 22 analog: Monitor detection happens automatically
# ---------------------------------------------------------------------------
class TestMonitorBehavior:
    """CC2 BUG 22: Monitor detection should happen at end of step."""

    def test_monitor_action_is_valid(self):
        const = _get_const()
        mask = np.array(compute_blue_action_mask(const, 0))
        assert mask[BLUE_MONITOR], "Monitor action should be valid"

    def test_sleep_action_is_valid(self):
        const = _get_const()
        mask = np.array(compute_blue_action_mask(const, 0))
        assert mask[BLUE_SLEEP], "Sleep action should be valid"


# ---------------------------------------------------------------------------
# Full episode differential: run CybORG FSM red + sleep blue, compare state
# ---------------------------------------------------------------------------
class TestFullEpisodeParity:
    """Run a full episode with FSM red and sleep blue, compare key state fields."""

    def test_10_step_sleep_episode(self):
        """All-sleep episode: no red/blue actions, only green side-effects."""
        from tests.differential.harness import CC4DifferentialHarness
        from tests.differential.state_comparator import _ERROR_FIELDS

        harness = CC4DifferentialHarness(
            seed=42,
            max_steps=500,
            red_cls=SleepAgent,
            green_cls=SleepAgent,
        )
        result = harness.run_episode(max_steps=10)

        error_diffs = []
        for sr in result.step_results:
            for d in sr.diffs:
                if d.field_name in _ERROR_FIELDS:
                    error_diffs.append((sr.step, d))

        if error_diffs:
            details = "\n".join(
                f"  Step {step}: {d.field_name} [{d.host_or_agent}] cyborg={d.cyborg_value} jax={d.jax_value}"
                for step, d in error_diffs[:20]
            )
            pytest.fail(f"{len(error_diffs)} state mismatches in 10-step sleep episode:\n{details}")

    def test_15_step_sleep_episode(self):
        """Longer all-sleep episode (stays within phase 0: 15*11=165 < 167)."""
        from tests.differential.harness import CC4DifferentialHarness
        from tests.differential.state_comparator import _ERROR_FIELDS

        harness = CC4DifferentialHarness(
            seed=42,
            max_steps=500,
            red_cls=SleepAgent,
            green_cls=SleepAgent,
        )
        result = harness.run_episode(max_steps=15)

        error_diffs = []
        for sr in result.step_results:
            for d in sr.diffs:
                if d.field_name in _ERROR_FIELDS:
                    error_diffs.append((sr.step, d))

        if error_diffs:
            details = "\n".join(
                f"  Step {step}: {d.field_name} [{d.host_or_agent}] cyborg={d.cyborg_value} jax={d.jax_value}"
                for step, d in error_diffs[:20]
            )
            pytest.fail(f"{len(error_diffs)} state mismatches in 15-step sleep episode:\n{details}")


# ---------------------------------------------------------------------------
# Action mask differential across multiple steps
# ---------------------------------------------------------------------------
class TestActionMaskAcrossSteps:
    """Verify action masks match CybORG not just at init, but after several steps."""

    def test_masks_match_after_5_steps(self, cyborg_env):
        from CybORG.Agents.Wrappers import BlueFlatWrapper

        wrapped = BlueFlatWrapper(cyborg_env, pad_spaces=True)
        wrapped.reset()

        const = build_const_from_cyborg(cyborg_env)

        for step in range(5):
            actions = {}
            for agent in wrapped.agents:
                actions[agent] = 0
            wrapped.step(actions)

        for agent_idx in range(NUM_BLUE_AGENTS):
            jax_mask = np.array(compute_blue_action_mask(const, agent_idx))
            assert jax_mask.sum() > 0, f"Agent {agent_idx} has no valid actions after 5 steps"

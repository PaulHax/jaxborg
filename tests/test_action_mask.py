import jax
import jax.numpy as jnp
import numpy as np

from jaxborg.actions.encoding import (
    BLUE_ALLOW_TRAFFIC_END,
    BLUE_ALLOW_TRAFFIC_START,
    BLUE_ANALYSE_START,
    BLUE_BLOCK_TRAFFIC_START,
    BLUE_DECOY_START,
    BLUE_REMOVE_START,
    BLUE_RESTORE_START,
)
from jaxborg.actions.masking import compute_blue_action_mask
from jaxborg.constants import GLOBAL_MAX_HOSTS, NUM_DECOY_TYPES, NUM_SUBNETS
from jaxborg.state import create_initial_const


def _make_const(active_hosts, router_hosts, agent_hosts, agent_subnets):
    """Build a minimal CC4Const with specified host/subnet assignments."""
    const = create_initial_const()
    host_active = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
    host_active[active_hosts] = True
    host_is_router = np.zeros(GLOBAL_MAX_HOSTS, dtype=bool)
    host_is_router[router_hosts] = True
    blue_agent_hosts = np.zeros_like(np.array(const.blue_agent_hosts), dtype=bool)
    blue_agent_hosts[0, agent_hosts] = True
    blue_agent_subnets = np.zeros_like(np.array(const.blue_agent_subnets), dtype=bool)
    blue_agent_subnets[0, agent_subnets] = True
    return const.replace(
        host_active=jnp.array(host_active),
        host_is_router=jnp.array(host_is_router),
        blue_agent_hosts=jnp.array(blue_agent_hosts),
        blue_agent_subnets=jnp.array(blue_agent_subnets),
    )


class TestActionMaskShape:
    def test_mask_shape(self):
        const = _make_const([0, 1, 2], [], [0, 1, 2], [0])
        mask = compute_blue_action_mask(const, 0)
        assert mask.shape == (BLUE_ALLOW_TRAFFIC_END,)
        assert mask.dtype == jnp.bool_


class TestSleepMonitorAlwaysValid:
    def test_sleep_always_true(self):
        const = _make_const([], [], [], [])
        mask = compute_blue_action_mask(const, 0)
        assert mask[0]

    def test_monitor_always_true(self):
        const = _make_const([], [], [], [])
        mask = compute_blue_action_mask(const, 0)
        assert mask[1]


class TestHostBasedActions:
    def test_inactive_hosts_masked(self):
        const = _make_const([0, 1], [], [0, 1, 2], [0])
        mask = compute_blue_action_mask(const, 0)
        assert mask[BLUE_ANALYSE_START + 0]
        assert mask[BLUE_ANALYSE_START + 1]
        assert not mask[BLUE_ANALYSE_START + 2]

    def test_router_hosts_masked(self):
        const = _make_const([0, 1, 2], [1], [0, 1, 2], [0])
        mask = compute_blue_action_mask(const, 0)
        assert mask[BLUE_ANALYSE_START + 0]
        assert not mask[BLUE_ANALYSE_START + 1]
        assert mask[BLUE_ANALYSE_START + 2]

    def test_unassigned_hosts_masked(self):
        const = _make_const([0, 1, 2], [], [0], [0])
        mask = compute_blue_action_mask(const, 0)
        assert mask[BLUE_ANALYSE_START + 0]
        assert not mask[BLUE_ANALYSE_START + 1]
        assert not mask[BLUE_ANALYSE_START + 2]

    def test_same_mask_for_analyse_remove_restore(self):
        const = _make_const([0, 1, 5, 10], [5], [0, 1, 5, 10], [0])
        mask = compute_blue_action_mask(const, 0)
        for h in range(GLOBAL_MAX_HOSTS):
            a = bool(mask[BLUE_ANALYSE_START + h])
            r = bool(mask[BLUE_REMOVE_START + h])
            s = bool(mask[BLUE_RESTORE_START + h])
            assert a == r == s, f"Mismatch at host {h}: analyse={a}, remove={r}, restore={s}"

    def test_decoy_same_host_mask_per_type(self):
        const = _make_const([0, 3], [], [0, 3], [0])
        mask = compute_blue_action_mask(const, 0)
        for d in range(NUM_DECOY_TYPES):
            offset = BLUE_DECOY_START + d * GLOBAL_MAX_HOSTS
            assert mask[offset + 0]
            assert not mask[offset + 1]
            assert not mask[offset + 2]
            assert mask[offset + 3]


class TestTrafficActions:
    def test_self_loops_masked(self):
        const = _make_const([0], [], [0], [0, 1, 2])
        mask = compute_blue_action_mask(const, 0)
        for s in range(NUM_SUBNETS):
            idx = BLUE_BLOCK_TRAFFIC_START + s * NUM_SUBNETS + s
            assert not mask[idx], f"Self-loop src={s} dst={s} should be masked"

    def test_uncontrolled_dst_masked(self):
        const = _make_const([0], [], [0], [2])
        mask = compute_blue_action_mask(const, 0)
        for src in range(NUM_SUBNETS):
            for dst in range(NUM_SUBNETS):
                idx_block = BLUE_BLOCK_TRAFFIC_START + src * NUM_SUBNETS + dst
                idx_allow = BLUE_ALLOW_TRAFFIC_START + src * NUM_SUBNETS + dst
                if dst != 2 or src == dst:
                    assert not mask[idx_block]
                    assert not mask[idx_allow]
                else:
                    assert mask[idx_block]
                    assert mask[idx_allow]

    def test_block_allow_same_mask(self):
        const = _make_const([0], [], [0], [1, 3])
        mask = compute_blue_action_mask(const, 0)
        n = NUM_SUBNETS * NUM_SUBNETS
        block_slice = mask[BLUE_BLOCK_TRAFFIC_START : BLUE_BLOCK_TRAFFIC_START + n]
        allow_slice = mask[BLUE_ALLOW_TRAFFIC_START : BLUE_ALLOW_TRAFFIC_START + n]
        np.testing.assert_array_equal(block_slice, allow_slice)


class TestJITCompatibility:
    def test_jit_compiles(self):
        const = _make_const([0, 1], [], [0, 1], [0])
        jitted = jax.jit(compute_blue_action_mask, static_argnums=(1,))
        mask = jitted(const, 0)
        assert mask.shape == (BLUE_ALLOW_TRAFFIC_END,)
        assert mask[0]


class TestWithRealTopology:
    def test_mask_from_build_topology(self):
        from jaxborg.topology import build_topology

        key = jax.random.PRNGKey(42)
        const = build_topology(key, num_steps=100)
        for agent_id in range(5):
            mask = compute_blue_action_mask(const, agent_id)
            assert mask.shape == (BLUE_ALLOW_TRAFFIC_END,)
            assert mask[0]
            assert mask[1]
            num_valid = int(mask.sum())
            assert num_valid > 2, f"Agent {agent_id} has only {num_valid} valid actions"
            assert num_valid < BLUE_ALLOW_TRAFFIC_END

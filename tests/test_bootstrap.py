import jax.numpy as jnp

from jaxborg.constants import (
    GLOBAL_MAX_HOSTS,
    NUM_SUBNETS,
    NUM_BLUE_AGENTS,
    NUM_RED_AGENTS,
    NUM_SERVICES,
    NUM_DECOY_TYPES,
    MISSION_PHASES,
    MESSAGE_LENGTH,
    MAX_STEPS,
    SUBNET_NAMES,
    SERVICE_NAMES,
    DECOY_NAMES,
)
from jaxborg.state import CC4State, CC4Const, create_initial_const, create_initial_state
from tests.catalog import SUBSYSTEMS, SUBSYSTEMS_BY_ID, get_next_incomplete


class TestConstants:
    def test_global_max_hosts(self):
        assert GLOBAL_MAX_HOSTS == 137

    def test_num_subnets(self):
        assert NUM_SUBNETS == 9

    def test_num_agents(self):
        assert NUM_BLUE_AGENTS == 5
        assert NUM_RED_AGENTS == 6

    def test_num_services(self):
        assert NUM_SERVICES == 5
        assert len(SERVICE_NAMES) == NUM_SERVICES

    def test_num_decoy_types(self):
        assert NUM_DECOY_TYPES == 4
        assert len(DECOY_NAMES) == NUM_DECOY_TYPES

    def test_mission_phases(self):
        assert MISSION_PHASES == 3

    def test_message_length(self):
        assert MESSAGE_LENGTH == 8

    def test_max_steps(self):
        assert MAX_STEPS == 500

    def test_subnet_names(self):
        assert len(SUBNET_NAMES) == NUM_SUBNETS
        assert "INTERNET" in SUBNET_NAMES
        assert "RESTRICTED_ZONE_A" in SUBNET_NAMES


class TestStructsImportable:
    def test_cc4_const_creates(self):
        const = create_initial_const()
        assert isinstance(const, CC4Const)
        assert const.host_active.shape == (GLOBAL_MAX_HOSTS,)
        assert const.subnet_adjacency.shape == (NUM_SUBNETS, NUM_SUBNETS)
        assert const.phase_rewards.shape == (MISSION_PHASES, NUM_SUBNETS, 3)

    def test_cc4_state_creates(self):
        state = create_initial_state()
        assert isinstance(state, CC4State)
        assert state.host_compromised.shape == (GLOBAL_MAX_HOSTS,)
        assert state.red_sessions.shape == (NUM_RED_AGENTS, GLOBAL_MAX_HOSTS)
        assert state.red_privilege.shape == (NUM_RED_AGENTS, GLOBAL_MAX_HOSTS)
        assert state.messages.shape == (NUM_BLUE_AGENTS, NUM_BLUE_AGENTS, MESSAGE_LENGTH)
        assert state.blocked_zones.shape == (NUM_SUBNETS, NUM_SUBNETS)

    def test_state_is_pytree(self):
        import jax
        state = create_initial_state()
        leaves = jax.tree.leaves(state)
        assert len(leaves) > 0
        assert all(isinstance(leaf, jnp.ndarray) or isinstance(leaf, int) for leaf in leaves)

    def test_const_is_pytree(self):
        import jax
        const = create_initial_const()
        leaves = jax.tree.leaves(const)
        assert len(leaves) > 0


class TestCatalog:
    def test_has_22_subsystems(self):
        assert len(SUBSYSTEMS) == 22

    def test_ids_are_sequential(self):
        ids = [s.id for s in SUBSYSTEMS]
        assert ids == list(range(1, 23))

    def test_all_deps_are_valid(self):
        valid_ids = {s.id for s in SUBSYSTEMS}
        for s in SUBSYSTEMS:
            for dep in s.depends_on:
                assert dep in valid_ids, f"Subsystem {s.id} ({s.name}) depends on nonexistent {dep}"

    def test_no_circular_deps(self):
        for s in SUBSYSTEMS:
            for dep in s.depends_on:
                assert dep < s.id, f"Subsystem {s.id} depends on {dep} which is not earlier"

    def test_get_next_incomplete_returns_subsystem_1(self):
        result = get_next_incomplete()
        assert result is not None
        assert result.id == 1
        assert result.name == "static_topology"

    def test_subsystem_22_depends_on_all_others(self):
        s22 = SUBSYSTEMS_BY_ID[22]
        assert set(s22.depends_on) == set(range(1, 22))


class TestStubsImportable:
    def test_topology_stub(self):
        from jaxborg.topology import build_const_from_cyborg
        assert callable(build_const_from_cyborg)

    def test_actions_stub(self):
        from jaxborg.actions import apply_blue_action, apply_red_action
        assert callable(apply_blue_action)
        assert callable(apply_red_action)

    def test_rewards_stub(self):
        from jaxborg.rewards import compute_rewards
        assert callable(compute_rewards)

    def test_observations_stub(self):
        from jaxborg.observations import get_blue_obs, get_red_obs
        assert callable(get_blue_obs)
        assert callable(get_red_obs)

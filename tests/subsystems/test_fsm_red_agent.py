import jax
import jax.numpy as jnp
import pytest

from jaxborg.actions.encoding import (
    RED_SLEEP,
)
from jaxborg.agents.fsm_red import (
    FSM_ACT_AGGRESSIVE_SCAN,
    FSM_ACT_DISCOVER,
    FSM_ACT_EXPLOIT,
    FSM_ACT_IMPACT,
    FSM_ACT_PRIVESC,
    FSM_ACT_WITHDRAW,
    FSM_F,
    FSM_K,
    FSM_KD,
    FSM_R,
    FSM_RD,
    FSM_S,
    FSM_U,
    PROBABILITY_MATRIX,
    TRANSITION_FAILURE,
    TRANSITION_SUCCESS,
    fsm_red_get_action,
    fsm_red_init_states,
    fsm_red_process_session_removal,
    fsm_red_update_state,
)
from jaxborg.constants import GLOBAL_MAX_HOSTS, NUM_RED_AGENTS
from jaxborg.state import create_initial_state
from jaxborg.topology import build_topology


@pytest.fixture
def jax_const():
    return build_topology(jnp.array([42]), num_steps=500)


class TestTransitionMatrices:
    def test_success_matrix_shape(self):
        assert TRANSITION_SUCCESS.shape == (9, 9)

    def test_failure_matrix_shape(self):
        assert TRANSITION_FAILURE.shape == (9, 9)

    def test_probability_matrix_shape(self):
        assert PROBABILITY_MATRIX.shape == (8, 9)

    def test_K_success_discover_goes_KD(self):
        assert int(TRANSITION_SUCCESS[FSM_K, FSM_ACT_DISCOVER]) == FSM_KD

    def test_K_success_aggressive_goes_S(self):
        assert int(TRANSITION_SUCCESS[FSM_K, FSM_ACT_AGGRESSIVE_SCAN]) == FSM_S

    def test_S_success_exploit_goes_U(self):
        assert int(TRANSITION_SUCCESS[FSM_S, FSM_ACT_EXPLOIT]) == FSM_U

    def test_U_success_privesc_goes_R(self):
        assert int(TRANSITION_SUCCESS[FSM_U, FSM_ACT_PRIVESC]) == FSM_R

    def test_R_success_impact_stays_R(self):
        assert int(TRANSITION_SUCCESS[FSM_R, FSM_ACT_IMPACT]) == FSM_R

    def test_U_success_withdraw_goes_S(self):
        assert int(TRANSITION_SUCCESS[FSM_U, FSM_ACT_WITHDRAW]) == FSM_S

    def test_R_success_withdraw_goes_S(self):
        assert int(TRANSITION_SUCCESS[FSM_R, FSM_ACT_WITHDRAW]) == FSM_S

    def test_K_failure_discover_stays_K(self):
        assert int(TRANSITION_FAILURE[FSM_K, FSM_ACT_DISCOVER]) == FSM_K

    def test_S_failure_exploit_stays_S(self):
        assert int(TRANSITION_FAILURE[FSM_S, FSM_ACT_EXPLOIT]) == FSM_S

    def test_F_success_discover_stays_F(self):
        assert int(TRANSITION_SUCCESS[FSM_F, FSM_ACT_DISCOVER]) == FSM_F

    def test_probability_sums_to_one(self):
        for state_idx in range(8):
            valid = PROBABILITY_MATRIX[state_idx] >= 0
            probs = PROBABILITY_MATRIX[state_idx][valid]
            total = float(jnp.sum(probs))
            assert abs(total - 1.0) < 1e-5, f"State {state_idx} probs sum to {total}"


class TestFsmUpdateState:
    def test_K_success_discover_transitions_to_KD(self):
        fsm = jnp.full((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), FSM_K, dtype=jnp.int32)
        new_fsm = fsm_red_update_state(fsm, 0, jnp.int32(5), FSM_ACT_DISCOVER, jnp.bool_(True))
        assert int(new_fsm[0, 5]) == FSM_KD

    def test_S_failure_exploit_stays_S(self):
        fsm = jnp.full((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), FSM_S, dtype=jnp.int32)
        new_fsm = fsm_red_update_state(fsm, 0, jnp.int32(10), FSM_ACT_EXPLOIT, jnp.bool_(False))
        assert int(new_fsm[0, 10]) == FSM_S

    def test_invalid_action_preserves_state(self):
        fsm = jnp.full((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), FSM_K, dtype=jnp.int32)
        new_fsm = fsm_red_update_state(fsm, 0, jnp.int32(5), FSM_ACT_EXPLOIT, jnp.bool_(True))
        assert int(new_fsm[0, 5]) == FSM_K

    def test_update_does_not_affect_other_hosts(self):
        fsm = jnp.full((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), FSM_K, dtype=jnp.int32)
        new_fsm = fsm_red_update_state(fsm, 0, jnp.int32(5), FSM_ACT_DISCOVER, jnp.bool_(True))
        assert int(new_fsm[0, 5]) == FSM_KD
        assert int(new_fsm[0, 6]) == FSM_K

    def test_update_does_not_affect_other_agents(self):
        fsm = jnp.full((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), FSM_S, dtype=jnp.int32)
        new_fsm = fsm_red_update_state(fsm, 0, jnp.int32(5), FSM_ACT_EXPLOIT, jnp.bool_(True))
        assert int(new_fsm[0, 5]) == FSM_U
        assert int(new_fsm[1, 5]) == FSM_S


class TestFsmGetAction:
    def test_returns_sleep_when_no_eligible_hosts(self, jax_const):
        state = create_initial_state()
        key = jax.random.PRNGKey(0)
        action = fsm_red_get_action(state, jax_const, 0, key)
        assert int(action) == RED_SLEEP

    def test_returns_valid_action_with_eligible_hosts(self, jax_const):
        state = create_initial_state()
        start_host = int(jax_const.red_start_hosts[0])
        discovered = state.red_discovered_hosts.at[0, start_host].set(True)
        sessions = state.red_sessions.at[0, start_host].set(True)
        fsm = state.fsm_host_states.at[0, start_host].set(FSM_K)
        state = state.replace(
            red_discovered_hosts=discovered,
            red_sessions=sessions,
            fsm_host_states=fsm,
        )
        key = jax.random.PRNGKey(42)
        action = int(fsm_red_get_action(state, jax_const, 0, key))
        assert action != RED_SLEEP

    def test_F_hosts_excluded(self, jax_const):
        state = create_initial_state()
        start_host = int(jax_const.red_start_hosts[0])
        discovered = state.red_discovered_hosts.at[0, start_host].set(True)
        sessions = state.red_sessions.at[0, start_host].set(True)
        fsm = state.fsm_host_states.at[0, start_host].set(FSM_F)
        state = state.replace(
            red_discovered_hosts=discovered,
            red_sessions=sessions,
            fsm_host_states=fsm,
        )
        key = jax.random.PRNGKey(0)
        action = int(fsm_red_get_action(state, jax_const, 0, key))
        assert action == RED_SLEEP

    def test_jit_compatible(self, jax_const):
        state = create_initial_state()
        start_host = int(jax_const.red_start_hosts[0])
        discovered = state.red_discovered_hosts.at[0, start_host].set(True)
        sessions = state.red_sessions.at[0, start_host].set(True)
        fsm = state.fsm_host_states.at[0, start_host].set(FSM_K)
        state = state.replace(
            red_discovered_hosts=discovered,
            red_sessions=sessions,
            fsm_host_states=fsm,
        )
        key = jax.random.PRNGKey(42)
        jitted = jax.jit(fsm_red_get_action, static_argnums=(2,))
        action = int(jitted(state, jax_const, 0, key))
        assert action != RED_SLEEP

    def test_multiple_calls_produce_different_actions(self, jax_const):
        state = create_initial_state()
        start_host = int(jax_const.red_start_hosts[0])
        discovered = state.red_discovered_hosts.at[0, start_host].set(True)
        sessions = state.red_sessions.at[0, start_host].set(True)
        fsm = state.fsm_host_states.at[0, start_host].set(FSM_K)
        state = state.replace(
            red_discovered_hosts=discovered,
            red_sessions=sessions,
            fsm_host_states=fsm,
        )

        actions = set()
        for seed in range(100):
            key = jax.random.PRNGKey(seed)
            action = int(fsm_red_get_action(state, jax_const, 0, key))
            actions.add(action)

        assert len(actions) > 1


class TestFsmInitStates:
    def test_start_host_gets_U(self, jax_const):
        fsm = fsm_red_init_states(jax_const, 0)
        start_host = int(jax_const.red_start_hosts[0])
        assert int(fsm[start_host]) == FSM_U

    def test_other_hosts_get_K(self, jax_const):
        fsm = fsm_red_init_states(jax_const, 0)
        start_host = int(jax_const.red_start_hosts[0])
        for h in range(GLOBAL_MAX_HOSTS):
            if h != start_host:
                assert int(fsm[h]) == FSM_K


class TestFsmSessionRemoval:
    def test_lost_session_transitions_to_KD(self):
        state = create_initial_state()
        fsm = state.fsm_host_states.at[0, 5].set(FSM_U)
        state = state.replace(fsm_host_states=fsm)

        new_fsm = fsm_red_process_session_removal(state, 0)
        assert int(new_fsm[0, 5]) == FSM_KD

    def test_kept_session_no_change(self):
        state = create_initial_state()
        fsm = state.fsm_host_states.at[0, 5].set(FSM_U)
        sessions = state.red_sessions.at[0, 5].set(True)
        state = state.replace(fsm_host_states=fsm, red_sessions=sessions)

        new_fsm = fsm_red_process_session_removal(state, 0)
        assert int(new_fsm[0, 5]) == FSM_U

    def test_R_lost_session_transitions_to_KD(self):
        state = create_initial_state()
        fsm = state.fsm_host_states.at[0, 5].set(FSM_R)
        state = state.replace(fsm_host_states=fsm)

        new_fsm = fsm_red_process_session_removal(state, 0)
        assert int(new_fsm[0, 5]) == FSM_KD

    def test_RD_lost_session_transitions_to_KD(self):
        state = create_initial_state()
        fsm = state.fsm_host_states.at[0, 5].set(FSM_RD)
        state = state.replace(fsm_host_states=fsm)

        new_fsm = fsm_red_process_session_removal(state, 0)
        assert int(new_fsm[0, 5]) == FSM_KD

    def test_K_state_unaffected(self):
        state = create_initial_state()
        fsm = state.fsm_host_states.at[0, 5].set(FSM_K)
        state = state.replace(fsm_host_states=fsm)

        new_fsm = fsm_red_process_session_removal(state, 0)
        assert int(new_fsm[0, 5]) == FSM_K

    def test_S_state_unaffected(self):
        state = create_initial_state()
        fsm = state.fsm_host_states.at[0, 5].set(FSM_S)
        state = state.replace(fsm_host_states=fsm)

        new_fsm = fsm_red_process_session_removal(state, 0)
        assert int(new_fsm[0, 5]) == FSM_S

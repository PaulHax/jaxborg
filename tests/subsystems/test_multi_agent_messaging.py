import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxborg.constants import (
    MESSAGE_LENGTH,
    NUM_BLUE_AGENTS,
)
from jaxborg.observations import get_blue_obs
from jaxborg.state import create_initial_state
from jaxborg.topology import build_const_from_cyborg, build_topology

try:
    from CybORG import CybORG
    from CybORG.Agents import SleepAgent
    from CybORG.Agents.Wrappers.BlueFlatWrapper import BlueFlatWrapper
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    HAS_CYBORG = True
except ImportError:
    HAS_CYBORG = False

cyborg_required = pytest.mark.skipif(not HAS_CYBORG, reason="CybORG not installed")

OBS_SIZE = 210
NUM_MESSAGES = 4
MESSAGE_SECTION_SIZE = NUM_MESSAGES * MESSAGE_LENGTH


@pytest.fixture
def jax_const():
    return build_topology(jnp.array([42]), num_steps=500)


@pytest.fixture
def jax_state(jax_const):
    state = create_initial_state()
    return state.replace(host_services=jax_const.initial_services)


class TestMessageSectionPosition:
    def test_messages_are_last_32_elements(self, jax_const, jax_state):
        obs = get_blue_obs(jax_state, jax_const, 0)
        assert obs.shape == (OBS_SIZE,)
        msg_section = obs[-MESSAGE_SECTION_SIZE:]
        assert msg_section.shape == (MESSAGE_SECTION_SIZE,)

    def test_zero_messages_give_zero_section(self, jax_const, jax_state):
        for agent_id in range(NUM_BLUE_AGENTS):
            obs = get_blue_obs(jax_state, jax_const, agent_id)
            msg_section = np.array(obs[-MESSAGE_SECTION_SIZE:])
            np.testing.assert_array_equal(
                msg_section,
                np.zeros(MESSAGE_SECTION_SIZE),
                err_msg=f"agent {agent_id} should have zero messages",
            )

    def test_message_section_size(self, jax_const, jax_state):
        assert MESSAGE_SECTION_SIZE == 32


class TestMessageContent:
    def test_setting_message_changes_obs(self, jax_const, jax_state):
        msg_val = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        msgs = jax_state.messages.at[1, 0, :].set(msg_val)
        state = jax_state.replace(messages=msgs)

        obs = get_blue_obs(state, jax_const, 0)
        msg_section = np.array(obs[-MESSAGE_SECTION_SIZE:])
        assert np.any(msg_section != 0), "message from agent 1 should appear in agent 0 obs"

    def test_self_messages_excluded(self, jax_const, jax_state):
        msg_val = jnp.ones(MESSAGE_LENGTH)
        msgs = jax_state.messages.at[0, 0, :].set(msg_val)
        state = jax_state.replace(messages=msgs)

        obs = get_blue_obs(state, jax_const, 0)
        msg_section = np.array(obs[-MESSAGE_SECTION_SIZE:])
        np.testing.assert_array_equal(
            msg_section,
            np.zeros(MESSAGE_SECTION_SIZE),
            err_msg="self-message should not appear in own observation",
        )

    def test_self_messages_excluded_all_agents(self, jax_const, jax_state):
        for agent_id in range(NUM_BLUE_AGENTS):
            msgs = jax_state.messages.at[agent_id, agent_id, :].set(jnp.ones(MESSAGE_LENGTH))
            state = jax_state.replace(messages=msgs)
            obs = get_blue_obs(state, jax_const, agent_id)
            msg_section = np.array(obs[-MESSAGE_SECTION_SIZE:])
            np.testing.assert_array_equal(
                msg_section,
                np.zeros(MESSAGE_SECTION_SIZE),
                err_msg=f"agent {agent_id}: self-message should not appear",
            )

    def test_messages_from_all_others(self, jax_const, jax_state):
        msgs = jax_state.messages
        for sender in range(NUM_BLUE_AGENTS):
            if sender == 0:
                continue
            msg_val = jnp.full(MESSAGE_LENGTH, float(sender))
            msgs = msgs.at[sender, 0, :].set(msg_val)
        state = jax_state.replace(messages=msgs)

        obs = get_blue_obs(state, jax_const, 0)
        msg_section = np.array(obs[-MESSAGE_SECTION_SIZE:])
        nonzero_count = np.count_nonzero(msg_section)
        assert nonzero_count == NUM_MESSAGES * MESSAGE_LENGTH, (
            f"expected {NUM_MESSAGES * MESSAGE_LENGTH} nonzero, got {nonzero_count}"
        )

    def test_message_ordering(self, jax_const, jax_state):
        msgs = jax_state.messages
        other_agents = [i for i in range(NUM_BLUE_AGENTS) if i != 2]
        for slot_idx, sender in enumerate(other_agents):
            msg_val = jnp.full(MESSAGE_LENGTH, float(sender + 1))
            msgs = msgs.at[sender, 2, :].set(msg_val)
        state = jax_state.replace(messages=msgs)

        obs = get_blue_obs(state, jax_const, 2)
        msg_section = np.array(obs[-MESSAGE_SECTION_SIZE:])

        for slot_idx in range(NUM_MESSAGES):
            slot = msg_section[slot_idx * MESSAGE_LENGTH : (slot_idx + 1) * MESSAGE_LENGTH]
            assert np.all(slot == slot[0]), f"slot {slot_idx} should be uniform: {slot}"

    def test_message_values_exact(self, jax_const, jax_state):
        msg_val = jnp.array([0.5, 1.0, 0.0, 0.25, 0.75, 0.0, 1.0, 0.5])
        msgs = jax_state.messages.at[3, 1, :].set(msg_val)
        state = jax_state.replace(messages=msgs)

        obs = get_blue_obs(state, jax_const, 1)
        msg_section = np.array(obs[-MESSAGE_SECTION_SIZE:])
        slot2 = msg_section[2 * MESSAGE_LENGTH : 3 * MESSAGE_LENGTH]
        np.testing.assert_array_almost_equal(slot2, np.array(msg_val))

    def test_message_ordering_agent0(self, jax_const, jax_state):
        msgs = jax_state.messages
        for sender in range(1, NUM_BLUE_AGENTS):
            msg_val = jnp.full(MESSAGE_LENGTH, float(sender * 10))
            msgs = msgs.at[sender, 0, :].set(msg_val)
        state = jax_state.replace(messages=msgs)

        obs = get_blue_obs(state, jax_const, 0)
        msg_section = np.array(obs[-MESSAGE_SECTION_SIZE:])

        expected_senders = [1, 2, 3, 4]
        for slot_idx, sender in enumerate(expected_senders):
            slot = msg_section[slot_idx * MESSAGE_LENGTH : (slot_idx + 1) * MESSAGE_LENGTH]
            expected_val = float(sender * 10)
            np.testing.assert_array_almost_equal(
                slot,
                np.full(MESSAGE_LENGTH, expected_val),
                err_msg=f"slot {slot_idx}: expected value {expected_val}",
            )

    def test_message_ordering_agent4(self, jax_const, jax_state):
        msgs = jax_state.messages
        for sender in range(4):
            msg_val = jnp.full(MESSAGE_LENGTH, float(sender + 100))
            msgs = msgs.at[sender, 4, :].set(msg_val)
        state = jax_state.replace(messages=msgs)

        obs = get_blue_obs(state, jax_const, 4)
        msg_section = np.array(obs[-MESSAGE_SECTION_SIZE:])

        expected_senders = [0, 1, 2, 3]
        for slot_idx, sender in enumerate(expected_senders):
            slot = msg_section[slot_idx * MESSAGE_LENGTH : (slot_idx + 1) * MESSAGE_LENGTH]
            expected_val = float(sender + 100)
            np.testing.assert_array_almost_equal(
                slot,
                np.full(MESSAGE_LENGTH, expected_val),
                err_msg=f"slot {slot_idx}: expected value {expected_val}",
            )

    def test_messages_do_not_leak_between_agents(self, jax_const, jax_state):
        msg_val = jnp.ones(MESSAGE_LENGTH) * 42.0
        msgs = jax_state.messages.at[1, 0, :].set(msg_val)
        state = jax_state.replace(messages=msgs)

        obs_agent2 = np.array(get_blue_obs(state, jax_const, 2))
        msg_section_agent2 = obs_agent2[-MESSAGE_SECTION_SIZE:]
        np.testing.assert_array_equal(
            msg_section_agent2,
            np.zeros(MESSAGE_SECTION_SIZE),
            err_msg="message to agent 0 should not appear in agent 2 obs",
        )


class TestMessageJIT:
    def test_jit_with_messages(self, jax_const, jax_state):
        msg_val = jnp.ones(MESSAGE_LENGTH)
        msgs = jax_state.messages.at[1, 0, :].set(msg_val)
        state = jax_state.replace(messages=msgs)

        jitted = jax.jit(get_blue_obs, static_argnums=(2,))
        obs = jitted(state, jax_const, 0)
        msg_section = np.array(obs[-MESSAGE_SECTION_SIZE:])
        assert np.any(msg_section != 0)

    def test_jit_all_agents_with_messages(self, jax_const, jax_state):
        msgs = jax_state.messages
        for sender in range(NUM_BLUE_AGENTS):
            for receiver in range(NUM_BLUE_AGENTS):
                if sender == receiver:
                    continue
                msg_val = jnp.full(MESSAGE_LENGTH, float(sender + receiver * 10))
                msgs = msgs.at[sender, receiver, :].set(msg_val)
        state = jax_state.replace(messages=msgs)

        for agent_id in range(NUM_BLUE_AGENTS):
            jitted = jax.jit(get_blue_obs, static_argnums=(2,))
            eager = np.array(get_blue_obs(state, jax_const, agent_id))
            jit_result = np.array(jitted(state, jax_const, agent_id))
            np.testing.assert_array_equal(
                eager,
                jit_result,
                err_msg=f"agent {agent_id}: JIT mismatch with messages",
            )


@cyborg_required
class TestDifferentialMessages:
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
    def wrapped_env(self, cyborg_env):
        return BlueFlatWrapper(cyborg_env, pad_spaces=True)

    @pytest.fixture
    def cyborg_and_jax(self, cyborg_env, wrapped_env):
        const = build_const_from_cyborg(cyborg_env)
        state = create_initial_state()
        state = state.replace(host_services=jnp.array(const.initial_services))
        return wrapped_env, const, state

    def test_initial_messages_zero_in_both(self, cyborg_and_jax):
        wrapped_env, const, state = cyborg_and_jax
        observations, _ = wrapped_env.reset()

        for agent_id in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{agent_id}"
            cyborg_msgs = observations[agent_name][-MESSAGE_SECTION_SIZE:]
            jax_obs = np.array(get_blue_obs(state, const, agent_id))
            jax_msgs = jax_obs[-MESSAGE_SECTION_SIZE:]

            np.testing.assert_array_equal(
                cyborg_msgs,
                jax_msgs,
                err_msg=f"{agent_name}: initial message section mismatch",
            )

    def test_non_message_obs_match_after_step(self, cyborg_and_jax):
        wrapped_env, const, state = cyborg_and_jax
        wrapped_env.reset()

        sleep_actions = {f"blue_agent_{i}": 0 for i in range(NUM_BLUE_AGENTS)}
        observations, *_ = wrapped_env.step(actions=sleep_actions)

        for agent_id in range(NUM_BLUE_AGENTS):
            agent_name = f"blue_agent_{agent_id}"
            cyborg_body = observations[agent_name][:-MESSAGE_SECTION_SIZE]
            jax_obs = np.array(get_blue_obs(state, const, agent_id))
            jax_body = jax_obs[:-MESSAGE_SECTION_SIZE]

            np.testing.assert_array_equal(
                cyborg_body,
                jax_body,
                err_msg=f"{agent_name}: non-message obs mismatch after step",
            )

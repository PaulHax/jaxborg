import numpy as np
import pytest
from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper


@pytest.fixture
def env():
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(scenario_generator=sg, seed=42)
    return BlueFlatWrapper(cyborg)


def test_env_loads(env):
    assert len(env.agents) == 5
    assert all("blue" in a for a in env.agents)


def test_reset_returns_observations(env):
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert set(obs.keys()) == set(env.agents)
    assert all(isinstance(obs[a], np.ndarray) for a in env.agents)


def test_reset_returns_action_masks(env):
    _, info = env.reset()
    for agent in env.agents:
        mask = info[agent]["action_mask"]
        assert mask[0] is True or mask[0] == 1


def test_step_with_sleep_returns_rewards(env):
    env.reset()
    actions = {a: 0 for a in env.agents}
    _, rewards, _, _, _ = env.step(actions)
    assert set(rewards.keys()) == set(env.agents)
    assert all(isinstance(float(rewards[a]), float) for a in env.agents)


def test_observation_shapes_consistent(env):
    obs, _ = env.reset()
    shapes_0_3 = [obs[f"blue_agent_{i}"].shape for i in range(4)]
    assert len(set(shapes_0_3)) == 1
    assert obs["blue_agent_4"].shape[0] >= shapes_0_3[0][0]


def test_rewards_are_negative_under_attack(env):
    env.reset()
    sleep_actions = {a: 0 for a in env.agents}
    cumulative = {a: 0.0 for a in env.agents}
    for _ in range(50):
        _, rewards, _, _, _ = env.step(sleep_actions)
        for a in env.agents:
            cumulative[a] += float(rewards[a])
    assert any(cumulative[a] < 0 for a in env.agents)

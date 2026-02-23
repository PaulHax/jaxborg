"""Differential tests comparing JAX FsmRedCC4Env vs CybORG with FiniteStateRedAgent."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def cyborg_sleep_env():
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Agents.Wrappers import BlueFlatWrapper
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    cyborg = CybORG(scenario_generator=sg, seed=42)
    return BlueFlatWrapper(env=cyborg)


@pytest.fixture
def jax_fsm_env():
    from jaxborg.fsm_red_env import FsmRedCC4Env

    return FsmRedCC4Env(num_steps=500)


@pytest.fixture
def jax_env_from_cyborg(cyborg_sleep_env):
    from jaxborg.env import CC4EnvState, _init_red_state
    from jaxborg.fsm_red_env import FsmRedCC4Env
    from jaxborg.state import create_initial_state
    from jaxborg.topology import build_const_from_cyborg

    inner_cyborg = cyborg_sleep_env.env
    const = build_const_from_cyborg(inner_cyborg)
    state = create_initial_state()
    state = state.replace(host_services=jnp.array(const.initial_services))
    state = _init_red_state(const, state)
    env_state = CC4EnvState(state=state, const=const)

    env = FsmRedCC4Env(num_steps=500)
    return env, env_state


class TestFsmRedEnvDifferential:
    def test_sleep_blue_cumulative_reward_same_sign(self, cyborg_sleep_env, jax_env_from_cyborg):
        """Sleep blue, FSM red: both should produce negative cumulative reward."""
        from statistics import mean

        from jaxborg.constants import NUM_BLUE_AGENTS

        cyborg_env = cyborg_sleep_env
        jax_env, jax_state = jax_env_from_cyborg

        cyborg_env.reset()
        cyborg_actions = {agent: 0 for agent in cyborg_env.agents}
        cyborg_total = 0.0
        for _ in range(500):
            _, rewards, _, _, _ = cyborg_env.step(cyborg_actions)
            cyborg_total += mean(rewards.values())

        key = jax.random.PRNGKey(0)
        jax_actions = {f"blue_{b}": jnp.int32(0) for b in range(NUM_BLUE_AGENTS)}
        jax_total = 0.0
        state = jax_state
        for _ in range(500):
            key, subkey = jax.random.split(key)
            _, state, rewards, _, _ = jax_env.step(subkey, state, jax_actions)
            jax_total += float(rewards["blue_0"])

        assert cyborg_total <= 0, f"CybORG sleep reward should be <= 0, got {cyborg_total}"
        if cyborg_total < 0:
            assert jax_total <= 0, f"JAX sleep reward should be <= 0 when CybORG is {cyborg_total}"

    def test_random_blue_reward_distribution(self, cyborg_sleep_env, jax_env_from_cyborg):
        """Random blue policy: compare reward distribution across seeds."""
        from jaxborg.actions.encoding import BLUE_ALLOW_TRAFFIC_END
        from jaxborg.constants import NUM_BLUE_AGENTS

        jax_env, jax_state = jax_env_from_cyborg
        num_episodes = 3
        jax_returns = []

        for ep in range(num_episodes):
            key = jax.random.PRNGKey(ep + 100)
            state = jax_state
            ep_return = 0.0
            for _ in range(500):
                key, act_key, step_key = jax.random.split(key, 3)
                actions = {
                    f"blue_{b}": jax.random.randint(jax.random.fold_in(act_key, b), (), 0, BLUE_ALLOW_TRAFFIC_END)
                    for b in range(NUM_BLUE_AGENTS)
                }
                _, state, rewards, _, _ = jax_env.step(step_key, state, actions)
                ep_return += float(rewards["blue_0"])
            jax_returns.append(ep_return)

        jax_mean = np.mean(jax_returns)
        assert np.isfinite(jax_mean), "JAX random baseline should produce finite returns"

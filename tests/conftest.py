import jax
import jax.numpy as jnp
import pytest

from jaxborg.actions import apply_blue_action, apply_red_action
from jaxborg.topology import build_topology

jit_apply_red = jax.jit(apply_red_action, static_argnums=(2,))
jit_apply_blue = jax.jit(apply_blue_action, static_argnums=(2,))


@pytest.fixture(scope="session")
def jax_const():
    return build_topology(jnp.array([42]), num_steps=500)


@pytest.fixture
def cyborg_env():
    from CybORG import CybORG
    from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    return CybORG(scenario_generator=sg, seed=42)

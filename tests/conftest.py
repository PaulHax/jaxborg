import pytest


def _cyborg_available():
    try:
        from CybORG import CybORG  # noqa: F401
        return True
    except ImportError:
        return False


cyborg_required = pytest.mark.skipif(
    not _cyborg_available(),
    reason="CybORG not installed",
)


@pytest.fixture
def cyborg_env():
    from CybORG import CybORG
    from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500,
    )
    return CybORG(scenario_generator=sg, seed=42)

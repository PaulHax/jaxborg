"""Regression tests for gaps found by differential fuzzing."""

import jax.numpy as jnp
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent

from jaxborg.constants import NUM_RED_AGENTS
from tests.differential.harness import CC4DifferentialHarness
from tests.differential.state_comparator import _ERROR_FIELDS


def _count_errors(harness, num_steps):
    total = 0
    for _ in range(num_steps):
        result = harness.full_step()
        total += sum(1 for d in result.diffs if d.field_name in _ERROR_FIELDS)
    return total


class TestFuzzGap1SessionReassignment:
    """Green phishing creates sessions that CybORG reassigns to subnet-owning agents."""

    def test_green_phishing_session_reassignment_seed0(self):
        harness = CC4DifferentialHarness(
            seed=0,
            max_steps=500,
            blue_cls=SleepAgent,
            green_cls=EnterpriseGreenAgent,
            red_cls=SleepAgent,
            sync_green_rng=True,
        )
        harness.reset()
        assert _count_errors(harness, 30) == 0

    def test_red_agent_subnets_populated(self):
        harness = CC4DifferentialHarness(
            seed=0,
            max_steps=500,
            blue_cls=SleepAgent,
            green_cls=EnterpriseGreenAgent,
            red_cls=SleepAgent,
            sync_green_rng=True,
        )
        harness.reset()
        for r in range(NUM_RED_AGENTS):
            assert bool(jnp.any(harness.jax_const.red_agent_subnets[r])), (
                f"red_agent_{r} has no allowed subnets in JAX const"
            )


class TestFuzzGap2ActionValidation:
    """JAX FSM actions must pass CybORG's action space validation.

    Fixes: (1) _pick_discover_subnet restricted to session subnets only
    (matching CybORG's observation-based subnet tracking), (2) exploit
    translation uses ExploitRemoteService (in CybORG's action space) instead
    of concrete classes like SSHBruteForce (not in action space).
    """

    def test_inactive_agent_no_sessions_seed0(self):
        harness = CC4DifferentialHarness(
            seed=0,
            max_steps=500,
            blue_cls=SleepAgent,
            green_cls=EnterpriseGreenAgent,
            red_cls=FiniteStateRedAgent,
            sync_green_rng=True,
        )
        harness.reset()
        assert _count_errors(harness, 10) == 0


class TestFuzzGap3ActionDuration:
    """CybORG actions have duration > 1. Harness defers JAX execution to match."""

    def test_red_action_duration_seed0(self):
        harness = CC4DifferentialHarness(
            seed=0,
            max_steps=500,
            blue_cls=SleepAgent,
            green_cls=EnterpriseGreenAgent,
            red_cls=FiniteStateRedAgent,
            sync_green_rng=True,
        )
        harness.reset()
        assert _count_errors(harness, 30) == 0


class TestFuzzGap4ExploitSelection:
    """ExploitRemoteService randomly picks a concrete exploit; translator must
    force the specific type matching the JAX action to avoid divergence.

    Fix: _FixedExploitSelector in translate.py overrides ExploitRemoteService's
    DefaultExploitActionSelector so CybORG delegates to the same concrete
    exploit (e.g. SSHBruteForce) that the JAX action represents.
    """

    def test_exploit_selection_seed0_50steps(self):
        harness = CC4DifferentialHarness(
            seed=0,
            max_steps=500,
            blue_cls=SleepAgent,
            green_cls=EnterpriseGreenAgent,
            red_cls=FiniteStateRedAgent,
            sync_green_rng=True,
        )
        harness.reset()
        assert _count_errors(harness, 50) == 0

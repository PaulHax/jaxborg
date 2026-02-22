from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions import apply_blue_action, apply_red_action
from jaxborg.actions.encoding import BLUE_SLEEP, RED_SLEEP
from jaxborg.actions.green import apply_green_agents
from jaxborg.agents.fsm_red import fsm_red_init_states
from jaxborg.constants import NUM_BLUE_AGENTS, NUM_RED_AGENTS
from jaxborg.state import create_initial_state
from jaxborg.topology import build_const_from_cyborg
from jaxborg.translate import (
    build_mappings_from_cyborg,
    jax_blue_to_cyborg,
    jax_red_to_cyborg,
)


@dataclass
class StateSnapshot:
    time: int = 0
    mission_phase: int = 0
    host_compromised: dict = field(default_factory=dict)
    red_privilege: dict = field(default_factory=dict)
    red_sessions: dict = field(default_factory=dict)
    host_services: dict = field(default_factory=dict)
    host_decoys: dict = field(default_factory=dict)
    ot_service_stopped: dict = field(default_factory=dict)
    blocked_zones: set = field(default_factory=set)
    rewards: dict = field(default_factory=dict)


@dataclass
class StateDiff:
    field_name: str
    cyborg_value: object
    jax_value: object
    host_or_agent: str = ""


@dataclass
class StepResult:
    step: int
    diffs: list[StateDiff] = field(default_factory=list)
    cyborg_rewards: dict = field(default_factory=dict)
    jax_rewards: dict = field(default_factory=dict)


@dataclass
class TestResult:
    steps_run: int = 0
    step_results: list[StepResult] = field(default_factory=list)
    error_diffs: int = 0


class CC4DifferentialHarness:
    def __init__(
        self,
        seed=42,
        max_steps=500,
        blue_cls=SleepAgent,
        green_cls=EnterpriseGreenAgent,
        red_cls=FiniteStateRedAgent,
        check_rewards=True,
        check_obs=False,
        sync_green_rng=False,
    ):
        self.seed = seed
        self.max_steps = max_steps
        self.blue_cls = blue_cls
        self.green_cls = green_cls
        self.red_cls = red_cls
        self.check_rewards = check_rewards
        self.check_obs = check_obs
        self.sync_green_rng = sync_green_rng
        self.cyborg_env = None
        self.jax_state = None
        self.jax_const = None
        self.mappings = None
        self.rng_key = None
        self.green_recorder = None

    def reset(self):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=self.blue_cls,
            green_agent_class=self.green_cls,
            red_agent_class=self.red_cls,
            steps=self.max_steps,
        )
        self.cyborg_env = CybORG(scenario_generator=sg, seed=self.seed)
        self.cyborg_env.reset()

        self.jax_const = build_const_from_cyborg(self.cyborg_env)
        self.mappings = build_mappings_from_cyborg(self.cyborg_env)

        self.jax_state = create_initial_state()
        self.jax_state = self.jax_state.replace(
            host_services=jnp.array(self.jax_const.initial_services),
        )

        start_sessions = jnp.zeros_like(self.jax_state.red_sessions)
        start_priv = jnp.zeros_like(self.jax_state.red_privilege)
        start_discovered = jnp.zeros_like(self.jax_state.red_discovered_hosts)
        host_compromised = self.jax_state.host_compromised
        fsm_states = self.jax_state.fsm_host_states
        cyborg_state = self.cyborg_env.environment_controller.state
        for agent_name, sessions in cyborg_state.sessions.items():
            if not agent_name.startswith("red_agent_"):
                continue
            red_idx = int(agent_name.split("_")[-1])
            if red_idx >= NUM_RED_AGENTS:
                continue
            for sess in sessions.values():
                if sess.hostname in self.mappings.hostname_to_idx:
                    hidx = self.mappings.hostname_to_idx[sess.hostname]
                    start_sessions = start_sessions.at[red_idx, hidx].set(True)
                    start_discovered = start_discovered.at[red_idx, hidx].set(True)
                    level = 1
                    if hasattr(sess, "username") and sess.username in ("root", "SYSTEM"):
                        level = 2
                    start_priv = start_priv.at[red_idx, hidx].set(jnp.maximum(start_priv[red_idx, hidx], level))
                    host_compromised = host_compromised.at[hidx].set(jnp.maximum(host_compromised[hidx], level))
            if self.jax_const.red_agent_active[red_idx]:
                fsm_states = fsm_states.at[red_idx].set(fsm_red_init_states(self.jax_const, red_idx))
        self.jax_state = self.jax_state.replace(
            red_sessions=start_sessions,
            red_privilege=start_priv,
            red_discovered_hosts=start_discovered,
            host_compromised=host_compromised,
            fsm_host_states=fsm_states,
        )

        self.rng_key = jax.random.PRNGKey(self.seed)

        if self.sync_green_rng:
            from tests.differential.green_recorder import GreenRecorder

            self.green_recorder = GreenRecorder()
            self.green_recorder.install(self.cyborg_env, self.mappings)
            self.jax_state = self.jax_state.replace(
                use_green_randoms=jnp.array(True),
            )

        from tests.differential.state_comparator import (
            extract_cyborg_snapshot,
            extract_jax_snapshot,
        )

        return (
            extract_cyborg_snapshot(self.cyborg_env, self.mappings),
            extract_jax_snapshot(self.jax_state, self.jax_const, self.mappings),
        )

    def step_red_only(self, agent_id: int, action_idx: int) -> StepResult:
        self.rng_key, subkey = jax.random.split(self.rng_key)

        cyborg_action = jax_red_to_cyborg(action_idx, agent_id, self.mappings)
        agent_name = f"red_agent_{agent_id}"
        self.cyborg_env.step(agent=agent_name, action=cyborg_action)

        self.jax_state = apply_red_action(self.jax_state, self.jax_const, agent_id, action_idx, subkey)

        from tests.differential.state_comparator import (
            compare_snapshots,
            extract_cyborg_snapshot,
            extract_jax_snapshot,
        )

        cyborg_snap = extract_cyborg_snapshot(self.cyborg_env, self.mappings)
        jax_snap = extract_jax_snapshot(self.jax_state, self.jax_const, self.mappings)
        diffs = compare_snapshots(cyborg_snap, jax_snap)

        return StepResult(step=int(self.jax_state.time), diffs=diffs)

    def step_blue_only(self, agent_id: int, action_idx: int) -> StepResult:
        cyborg_action = jax_blue_to_cyborg(action_idx, agent_id, self.mappings)
        agent_name = f"blue_agent_{agent_id}"
        self.cyborg_env.step(agent=agent_name, action=cyborg_action)

        self.jax_state = apply_blue_action(self.jax_state, self.jax_const, agent_id, action_idx)

        from tests.differential.state_comparator import (
            compare_snapshots,
            extract_cyborg_snapshot,
            extract_jax_snapshot,
        )

        cyborg_snap = extract_cyborg_snapshot(self.cyborg_env, self.mappings)
        jax_snap = extract_jax_snapshot(self.jax_state, self.jax_const, self.mappings)
        diffs = compare_snapshots(cyborg_snap, jax_snap)

        return StepResult(step=int(self.jax_state.time), diffs=diffs)

    def step(self, actions: dict) -> StepResult:
        self.rng_key, *subkeys = jax.random.split(self.rng_key, NUM_RED_AGENTS + 1)

        for agent_name, action_idx in actions.items():
            if agent_name.startswith("red_agent_"):
                cyborg_action = jax_red_to_cyborg(action_idx, _agent_idx(agent_name), self.mappings)
            else:
                cyborg_action = jax_blue_to_cyborg(action_idx, _agent_idx(agent_name), self.mappings)
            self.cyborg_env.step(agent=agent_name, action=cyborg_action)

        for agent_name, action_idx in actions.items():
            if agent_name.startswith("red_agent_"):
                aid = _agent_idx(agent_name)
                self.jax_state = apply_red_action(self.jax_state, self.jax_const, aid, action_idx, subkeys[aid])
            else:
                aid = _agent_idx(agent_name)
                self.jax_state = apply_blue_action(self.jax_state, self.jax_const, aid, action_idx)

        from tests.differential.state_comparator import (
            compare_snapshots,
            extract_cyborg_snapshot,
            extract_jax_snapshot,
        )

        cyborg_snap = extract_cyborg_snapshot(self.cyborg_env, self.mappings)
        jax_snap = extract_jax_snapshot(self.jax_state, self.jax_const, self.mappings)
        diffs = compare_snapshots(cyborg_snap, jax_snap)

        return StepResult(step=int(self.jax_state.time), diffs=diffs)

    def full_step(self, red_actions=None, blue_actions=None) -> StepResult:
        """Single CybORG step processing all agents at once. Required for green RNG sync."""
        self.rng_key, key_green, *subkeys = jax.random.split(self.rng_key, NUM_RED_AGENTS + 2)

        if red_actions is None:
            red_actions = {r: RED_SLEEP for r in range(NUM_RED_AGENTS)}
        if blue_actions is None:
            blue_actions = {b: BLUE_SLEEP for b in range(NUM_BLUE_AGENTS)}

        cyborg_actions = {}
        for r, action_idx in red_actions.items():
            cyborg_actions[f"red_agent_{r}"] = jax_red_to_cyborg(action_idx, r, self.mappings)
        for b, action_idx in blue_actions.items():
            cyborg_actions[f"blue_agent_{b}"] = jax_blue_to_cyborg(action_idx, b, self.mappings)

        self.cyborg_env.environment_controller.step(cyborg_actions)

        if self.green_recorder:
            step_fields = self.green_recorder.extract_step(int(self.jax_state.time))
            green_randoms = self.jax_state.green_randoms.at[self.jax_state.time].set(jnp.array(step_fields))
            self.jax_state = self.jax_state.replace(green_randoms=green_randoms)

        for r, action_idx in red_actions.items():
            self.jax_state = apply_red_action(self.jax_state, self.jax_const, r, action_idx, subkeys[r])
        for b, action_idx in blue_actions.items():
            self.jax_state = apply_blue_action(self.jax_state, self.jax_const, b, action_idx)

        self.jax_state = apply_green_agents(self.jax_state, self.jax_const, key_green)

        from tests.differential.state_comparator import (
            compare_snapshots,
            extract_cyborg_snapshot,
            extract_jax_snapshot,
        )

        cyborg_snap = extract_cyborg_snapshot(self.cyborg_env, self.mappings)
        jax_snap = extract_jax_snapshot(self.jax_state, self.jax_const, self.mappings)
        diffs = compare_snapshots(cyborg_snap, jax_snap)

        return StepResult(step=int(self.jax_state.time), diffs=diffs)

    def run_episode(self, blue_policies=None, red_policy=None, max_steps=None) -> TestResult:
        max_steps = max_steps or self.max_steps
        self.reset()

        step_results = []
        error_count = 0

        for t in range(max_steps):
            actions = {}

            if red_policy:
                for r in range(NUM_RED_AGENTS):
                    actions[f"red_agent_{r}"] = red_policy(self.jax_state, self.jax_const, r)
            else:
                for r in range(NUM_RED_AGENTS):
                    actions[f"red_agent_{r}"] = RED_SLEEP

            if blue_policies:
                for b in range(NUM_BLUE_AGENTS):
                    actions[f"blue_agent_{b}"] = blue_policies(self.jax_state, self.jax_const, b)
            else:
                for b in range(NUM_BLUE_AGENTS):
                    actions[f"blue_agent_{b}"] = BLUE_SLEEP

            result = self.step(actions)
            step_results.append(result)

            from tests.differential.state_comparator import _ERROR_FIELDS

            error_count += sum(1 for d in result.diffs if d.field_name in _ERROR_FIELDS)

            self.jax_state = self.jax_state.replace(time=t + 1)

        return TestResult(
            steps_run=max_steps,
            step_results=step_results,
            error_diffs=error_count,
        )

    def get_cyborg_snapshot(self) -> StateSnapshot:
        from tests.differential.state_comparator import extract_cyborg_snapshot

        return extract_cyborg_snapshot(self.cyborg_env, self.mappings)

    def get_jax_snapshot(self) -> StateSnapshot:
        from tests.differential.state_comparator import extract_jax_snapshot

        return extract_jax_snapshot(self.jax_state, self.jax_const, self.mappings)


def _agent_idx(agent_name: str) -> int:
    return int(agent_name.split("_")[-1])

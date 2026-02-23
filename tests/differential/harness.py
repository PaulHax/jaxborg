from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from jaxborg.actions import apply_blue_action, apply_red_action
from jaxborg.actions.encoding import (
    BLUE_SLEEP,
    RED_SLEEP,
)
from jaxborg.actions.green import apply_green_agents
from jaxborg.agents.fsm_red import (
    fsm_red_get_action_and_info,
    fsm_red_init_states,
    fsm_red_post_step_update,
)
from jaxborg.constants import GLOBAL_MAX_HOSTS, NUM_BLUE_AGENTS, NUM_RED_AGENTS
from jaxborg.reassignment import reassign_cross_subnet_sessions
from jaxborg.rewards import advance_mission_phase
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


_ZERO_INT_HOSTS = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32)
_ZERO_BOOL_HOSTS = jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_)


@jax.jit
def _jit_fsm_red_get_action_and_info(state, const, agent_id, key):
    return fsm_red_get_action_and_info(state, const, jnp.int32(agent_id), key)


@jax.jit
def _jit_apply_red_action(state, const, agent_id, action_idx, key):
    return apply_red_action(state, const, jnp.int32(agent_id), jnp.int32(action_idx), key)


@jax.jit
def _jit_apply_blue_action(state, const, agent_id, action_idx):
    return apply_blue_action(state, const, jnp.int32(agent_id), jnp.int32(action_idx))


@jax.jit
def _jit_advance_and_clear(state, const):
    state = advance_mission_phase(state, const)
    return state.replace(
        red_activity_this_step=_ZERO_INT_HOSTS,
        green_lwf_this_step=_ZERO_BOOL_HOSTS,
        green_asf_this_step=_ZERO_BOOL_HOSTS,
        host_activity_detected=_ZERO_BOOL_HOSTS,
    )


@jax.jit
def _jit_fsm_red_post_step_update(state_before, state_after, const, target_hosts, fsm_actions, eligible_flags):
    return fsm_red_post_step_update(
        state_before,
        state_after,
        const,
        target_hosts,
        fsm_actions,
        eligible_flags,
    )


@jax.jit
def _jit_green_and_reassign(state, const, key):
    state = apply_green_agents(state, const, key)
    return reassign_cross_subnet_sessions(state, const)


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
        self._red_pending_jax = [None] * NUM_RED_AGENTS

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
        cyborg_state = self.cyborg_env.environment_controller.state
        controller = self.cyborg_env.environment_controller

        # CybORG action spaces seed red knowledge (known IPs/processes) even for
        # agents without active sessions. Mirror that into JAX init state.
        known_hosts_by_red = [set() for _ in range(NUM_RED_AGENTS)]
        scanned_hosts_by_red = [set() for _ in range(NUM_RED_AGENTS)]
        red_start_hosts = self.jax_const.red_start_hosts
        red_agent_active = self.jax_const.red_agent_active
        red_initial_discovered = self.jax_const.red_initial_discovered_hosts
        red_initial_scanned = self.jax_const.red_initial_scanned_hosts

        pid_to_host_idx = {}
        for hostname, host in cyborg_state.hosts.items():
            if hostname not in self.mappings.hostname_to_idx:
                continue
            hidx = self.mappings.hostname_to_idx[hostname]
            for proc in host.processes:
                pid = getattr(proc, "pid", None)
                if pid is not None:
                    pid_to_host_idx[int(pid)] = hidx

        for r in range(NUM_RED_AGENTS):
            iface = controller.agent_interfaces.get(f"red_agent_{r}")
            if iface is None:
                continue
            aspace = iface.action_space

            for ip, known in getattr(aspace, "ip_address", {}).items():
                if not known:
                    continue
                hostname = cyborg_state.ip_addresses.get(ip)
                if hostname in self.mappings.hostname_to_idx:
                    known_hosts_by_red[r].add(self.mappings.hostname_to_idx[hostname])

            for pid, known in getattr(aspace, "process", {}).items():
                if known and int(pid) in pid_to_host_idx:
                    scanned_hosts_by_red[r].add(pid_to_host_idx[int(pid)])

            if known_hosts_by_red[r]:
                red_agent_active = red_agent_active.at[r].set(True)
                red_start_hosts = red_start_hosts.at[r].set(min(known_hosts_by_red[r]))

            for hidx in known_hosts_by_red[r]:
                red_initial_discovered = red_initial_discovered.at[r, hidx].set(True)
            for hidx in scanned_hosts_by_red[r]:
                red_initial_scanned = red_initial_scanned.at[r, hidx].set(True)

        self.jax_const = self.jax_const.replace(
            red_start_hosts=red_start_hosts,
            red_agent_active=red_agent_active,
            red_initial_discovered_hosts=red_initial_discovered,
            red_initial_scanned_hosts=red_initial_scanned,
        )

        self.jax_state = create_initial_state()
        self.jax_state = self.jax_state.replace(
            host_services=jnp.array(self.jax_const.initial_services),
        )

        start_sessions = jnp.zeros_like(self.jax_state.red_sessions)
        start_priv = jnp.zeros_like(self.jax_state.red_privilege)
        start_discovered = jnp.array(self.jax_const.red_initial_discovered_hosts)
        start_scanned = jnp.array(self.jax_const.red_initial_scanned_hosts)
        host_compromised = self.jax_state.host_compromised
        fsm_states = self.jax_state.fsm_host_states
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
            red_scanned_hosts=start_scanned,
            host_compromised=host_compromised,
            fsm_host_states=fsm_states,
        )

        self.rng_key = jax.random.PRNGKey(self.seed)
        self._red_pending_jax = [None] * NUM_RED_AGENTS

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

    def full_step(self, blue_actions=None) -> StepResult:
        """E2E step mirroring FsmRedCC4Env.step_env(): FSM red + green + blue + reassign + FSM update."""
        self.rng_key, key_green, key_red, *subkeys = jax.random.split(self.rng_key, NUM_RED_AGENTS + 3)
        red_keys = jax.random.split(key_red, NUM_RED_AGENTS)

        if blue_actions is None:
            blue_actions = {b: BLUE_SLEEP for b in range(NUM_BLUE_AGENTS)}

        use_fsm = self.red_cls is FiniteStateRedAgent

        # --- Mirror step_env: advance phase + clear per-step fields ---
        self.jax_state = _jit_advance_and_clear(self.jax_state, self.jax_const)

        state_before = self.jax_state

        # --- Red action selection ---
        red_actions = {}
        target_hosts = []
        fsm_actions = []
        eligible_flags = []
        for r in range(NUM_RED_AGENTS):
            if use_fsm:
                action, host, fsm_act, eligible = _jit_fsm_red_get_action_and_info(
                    self.jax_state, self.jax_const, r, red_keys[r]
                )
                red_actions[r] = int(action)
                target_hosts.append(host)
                fsm_actions.append(fsm_act)
                eligible_flags.append(eligible)
            else:
                red_actions[r] = RED_SLEEP
                target_hosts.append(jnp.int32(0))
                fsm_actions.append(jnp.int32(0))
                eligible_flags.append(jnp.bool_(False))

        # --- CybORG side ---
        cyborg_actions = {}
        for r, action_idx in red_actions.items():
            cyborg_actions[f"red_agent_{r}"] = jax_red_to_cyborg(action_idx, r, self.mappings)
        for b, action_idx in blue_actions.items():
            cyborg_actions[f"blue_agent_{b}"] = jax_blue_to_cyborg(action_idx, b, self.mappings)

        controller = self.cyborg_env.environment_controller

        for r, action_idx in red_actions.items():
            aip = controller.actions_in_progress.get(f"red_agent_{r}")
            if aip is None and action_idx != RED_SLEEP:
                self._red_pending_jax[r] = (action_idx, subkeys[r])

        controller.step(cyborg_actions)

        # --- Green RNG sync ---
        if self.green_recorder:
            step_fields = self.green_recorder.extract_step(int(self.jax_state.time))
            green_randoms = self.jax_state.green_randoms.at[self.jax_state.time].set(jnp.array(step_fields))
            self.jax_state = self.jax_state.replace(green_randoms=green_randoms)

        # --- JAX red actions (duration-aware) ---
        for r in range(NUM_RED_AGENTS):
            aip = controller.actions_in_progress.get(f"red_agent_{r}")
            if aip is None and self._red_pending_jax[r] is not None:
                action_idx, key = self._red_pending_jax[r]
                self.jax_state = _jit_apply_red_action(self.jax_state, self.jax_const, r, action_idx, key)
                self._red_pending_jax[r] = None

        # --- JAX blue actions ---
        for b, action_idx in blue_actions.items():
            if action_idx != BLUE_SLEEP:
                self.jax_state = _jit_apply_blue_action(self.jax_state, self.jax_const, b, action_idx)

        # --- JAX green + reassign (JIT'd) ---
        self.jax_state = _jit_green_and_reassign(self.jax_state, self.jax_const, key_green)

        # --- FSM state updates (shared with FsmRedCC4Env) ---
        if use_fsm:
            self.jax_state = _jit_fsm_red_post_step_update(
                state_before,
                self.jax_state,
                self.jax_const,
                jnp.asarray(target_hosts, dtype=jnp.int32),
                jnp.asarray(fsm_actions, dtype=jnp.int32),
                jnp.asarray(eligible_flags, dtype=jnp.bool_),
            )

        # --- Time increment ---
        self.jax_state = self.jax_state.replace(time=self.jax_state.time + 1)

        # --- Compare ---
        from tests.differential.state_comparator import compare_fast

        diffs = compare_fast(self.cyborg_env, self.jax_state, self.jax_const, self.mappings)

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

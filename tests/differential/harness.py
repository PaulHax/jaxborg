from dataclasses import dataclass, field


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
    def __init__(self, seed=42, max_steps=500, check_rewards=True, check_obs=False):
        self.seed = seed
        self.max_steps = max_steps
        self.check_rewards = check_rewards
        self.check_obs = check_obs

    def reset(self):
        raise NotImplementedError("Subsystem 1: topology comparison")

    def step(self, blue_actions: dict[str, int]) -> StepResult:
        raise NotImplementedError("Subsystem 1+: step comparison")

    def run_episode(self, blue_policies, red_policy=None, max_steps=None) -> TestResult:
        raise NotImplementedError("Subsystem 1+: full episode comparison")

    def get_cyborg_snapshot(self) -> StateSnapshot:
        raise NotImplementedError("Subsystem 1: state extraction")

    def get_jax_snapshot(self) -> StateSnapshot:
        raise NotImplementedError("Subsystem 1: state extraction")

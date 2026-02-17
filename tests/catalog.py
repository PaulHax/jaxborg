import json
from dataclasses import dataclass, field
from pathlib import Path

CATALOG_STATUS_PATH = Path(__file__).parent / "catalog_status.json"


@dataclass
class Subsystem:
    id: int
    name: str
    description: str
    depends_on: list[int] = field(default_factory=list)
    cyborg_source_paths: list[str] = field(default_factory=list)
    jax_target_files: list[str] = field(default_factory=list)


SUBSYSTEMS: list[Subsystem] = [
    Subsystem(
        id=1,
        name="static_topology",
        description="Static topology construction: hosts, subnets, adjacency, host properties",
        depends_on=[],
        cyborg_source_paths=[
            "CybORG/Simulator/Scenarios/EnterpriseScenarioGenerator.py",
            "CybORG/Simulator/State.py",
        ],
        jax_target_files=["src/jaxborg/topology.py", "src/jaxborg/state.py"],
    ),
    Subsystem(
        id=2,
        name="red_discover",
        description="Red: discover remote systems",
        depends_on=[1],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/DiscoverRemoteSystems.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
    ),
    Subsystem(
        id=3,
        name="red_scan",
        description="Red: scan host (DiscoverNetworkServices)",
        depends_on=[2],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/DiscoverNetworkServices.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
    ),
    Subsystem(
        id=4,
        name="red_exploit_ssh",
        description="Red: exploit — SSH brute force (simplest exploit)",
        depends_on=[3],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/ExploitActions/SSHBruteForce.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
    ),
    Subsystem(
        id=5,
        name="red_exploit_remaining",
        description="Red: exploit — remaining types (FTP, HTTP, HTTPS, Haraka, SQL, EternalBlue, BlueKeep)",
        depends_on=[4],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/ExploitActions/",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
    ),
    Subsystem(
        id=6,
        name="red_privesc",
        description="Red: privilege escalation",
        depends_on=[4],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/EscalateActions/",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
    ),
    Subsystem(
        id=7,
        name="red_impact",
        description="Red: impact",
        depends_on=[6],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/Impact.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
    ),
    Subsystem(
        id=8,
        name="blue_monitor",
        description="Blue: monitor (activity detection)",
        depends_on=[1],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/Monitor.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
    ),
    Subsystem(
        id=9,
        name="blue_analyse",
        description="Blue: analyse",
        depends_on=[8],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/Analyse.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
    ),
    Subsystem(
        id=10,
        name="blue_remove",
        description="Blue: remove",
        depends_on=[9],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/Remove.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
    ),
    Subsystem(
        id=11,
        name="blue_restore",
        description="Blue: restore",
        depends_on=[10],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/Restore.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
    ),
    Subsystem(
        id=12,
        name="blue_decoys",
        description="Blue: decoys (all types, OS restrictions, exploit blocking matrix)",
        depends_on=[11],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/DecoyActions/",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
    ),
    Subsystem(
        id=13,
        name="rewards",
        description="Rewards (confidentiality + availability + restore/impact costs)",
        depends_on=[7, 11],
        cyborg_source_paths=[
            "CybORG/Shared/BlueRewardMachine.py",
        ],
        jax_target_files=["src/jaxborg/rewards.py"],
    ),
    Subsystem(
        id=14,
        name="observations",
        description="Observations (blue obs encoding per agent, red obs encoding)",
        depends_on=[8],
        cyborg_source_paths=[
            "CybORG/Agents/Wrappers/BlueFlatWrapper.py",
            "CybORG/Agents/Wrappers/EnterpriseMAE.py",
        ],
        jax_target_files=["src/jaxborg/observations.py"],
    ),
    Subsystem(
        id=15,
        name="phase_transitions",
        description="Phase transitions (0->1->2, reward weight changes, allowed subnet pairs)",
        depends_on=[13],
        cyborg_source_paths=[
            "CybORG/Simulator/Scenarios/EnterpriseScenarioGenerator.py",
            "CybORG/Shared/BlueRewardMachine.py",
        ],
        jax_target_files=["src/jaxborg/state.py", "src/jaxborg/rewards.py"],
    ),
    Subsystem(
        id=16,
        name="green_agents",
        description="Green agents (false positives, phishing)",
        depends_on=[1],
        cyborg_source_paths=[
            "CybORG/Agents/SimpleAgents/EnterpriseGreenAgent.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
    ),
    Subsystem(
        id=17,
        name="blue_traffic_zones",
        description="Blue: block/allow traffic zones",
        depends_on=[11],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/ControlTraffic.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
    ),
    Subsystem(
        id=18,
        name="multi_agent_messaging",
        description="Multi-agent observations and messaging (8-bit inter-agent vectors)",
        depends_on=[14],
        cyborg_source_paths=[
            "CybORG/Agents/Wrappers/EnterpriseMAE.py",
        ],
        jax_target_files=["src/jaxborg/observations.py"],
    ),
    Subsystem(
        id=19,
        name="dynamic_topology",
        description="Dynamic topology (pad-to-max with host_active masking)",
        depends_on=[1],
        cyborg_source_paths=[
            "CybORG/Simulator/Scenarios/EnterpriseScenarioGenerator.py",
        ],
        jax_target_files=["src/jaxborg/topology.py", "src/jaxborg/state.py"],
    ),
    Subsystem(
        id=20,
        name="fsm_red_agent",
        description="FiniteStateRedAgent (8-state FSM, probabilistic transitions)",
        depends_on=[7],
        cyborg_source_paths=[
            "CybORG/Agents/SimpleAgents/FiniteStateRedAgent.py",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
    ),
    Subsystem(
        id=21,
        name="cc4_new_red_actions",
        description="CC4-new red actions (AggressiveServiceDiscovery, StealthServiceDiscovery, etc.)",
        depends_on=[5],
        cyborg_source_paths=[
            "CybORG/Simulator/Actions/ConcreteActions/",
        ],
        jax_target_files=["src/jaxborg/actions.py"],
    ),
    Subsystem(
        id=22,
        name="full_episode_fuzzing",
        description="Full episode fuzzing (all subsystems, 200+ seeds, extended episodes)",
        depends_on=list(range(1, 22)),
        cyborg_source_paths=[],
        jax_target_files=[],
    ),
]

SUBSYSTEMS_BY_ID = {s.id: s for s in SUBSYSTEMS}


def _load_status() -> dict[int, str]:
    if not CATALOG_STATUS_PATH.exists():
        return {}
    return {int(k): v for k, v in json.loads(CATALOG_STATUS_PATH.read_text()).items()}


def _save_status(status: dict[int, str]) -> None:
    CATALOG_STATUS_PATH.write_text(json.dumps({str(k): v for k, v in sorted(status.items())}, indent=2) + "\n")


def get_next_incomplete() -> Subsystem | None:
    status = _load_status()
    for s in SUBSYSTEMS:
        if status.get(s.id) == "passing":
            continue
        deps_met = all(status.get(d) == "passing" for d in s.depends_on)
        if deps_met:
            return s
    return None


def mark_passing(subsystem_id: int) -> None:
    status = _load_status()
    status[subsystem_id] = "passing"
    _save_status(status)


def is_all_done() -> bool:
    status = _load_status()
    return all(status.get(s.id) == "passing" for s in SUBSYSTEMS)

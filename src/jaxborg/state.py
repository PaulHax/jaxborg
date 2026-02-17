import chex
import jax.numpy as jnp
from flax import struct

from jaxborg.constants import (
    GLOBAL_MAX_HOSTS,
    MAX_DETECTION_RANDOMS,
    MAX_STEPS,
    MESSAGE_LENGTH,
    MISSION_PHASES,
    NUM_BLUE_AGENTS,
    NUM_DECOY_TYPES,
    NUM_RED_AGENTS,
    NUM_SERVICES,
    NUM_SUBNETS,
    OBS_HOSTS_PER_SUBNET,
)


@struct.dataclass
class CC4Const:
    host_active: chex.Array  # (GLOBAL_MAX_HOSTS,) bool
    host_subnet: chex.Array  # (GLOBAL_MAX_HOSTS,) int
    host_is_router: chex.Array  # (GLOBAL_MAX_HOSTS,) bool
    host_is_server: chex.Array  # (GLOBAL_MAX_HOSTS,) bool
    host_is_user: chex.Array  # (GLOBAL_MAX_HOSTS,) bool
    subnet_adjacency: chex.Array  # (NUM_SUBNETS, NUM_SUBNETS) bool
    data_links: chex.Array  # (GLOBAL_MAX_HOSTS, GLOBAL_MAX_HOSTS) bool

    initial_services: chex.Array  # (GLOBAL_MAX_HOSTS, NUM_SERVICES) bool
    host_has_bruteforceable_user: chex.Array  # (GLOBAL_MAX_HOSTS,) bool
    host_has_rfi: chex.Array  # (GLOBAL_MAX_HOSTS,) bool
    host_respond_to_ping: chex.Array  # (GLOBAL_MAX_HOSTS,) bool

    blue_agent_subnets: chex.Array  # (NUM_BLUE_AGENTS, NUM_SUBNETS) bool
    blue_agent_hosts: chex.Array  # (NUM_BLUE_AGENTS, GLOBAL_MAX_HOSTS) bool
    red_start_hosts: chex.Array  # (NUM_RED_AGENTS,) int
    red_agent_active: chex.Array  # (NUM_RED_AGENTS,) bool

    green_agent_host: chex.Array  # (GLOBAL_MAX_HOSTS,) int — green agent index per host, -1 if none
    green_agent_active: chex.Array  # (GLOBAL_MAX_HOSTS,) bool
    num_green_agents: int

    phase_rewards: chex.Array  # (MISSION_PHASES, NUM_SUBNETS, 3) float — LWF/ASF/RIA per subnet per phase
    phase_boundaries: chex.Array  # (MISSION_PHASES,) int — step at which each phase starts
    allowed_subnet_pairs: chex.Array  # (MISSION_PHASES, NUM_SUBNETS, NUM_SUBNETS) bool

    obs_host_map: chex.Array  # (NUM_SUBNETS, OBS_HOSTS_PER_SUBNET) int — global host idx per subnet in obs order
    blue_obs_subnets: chex.Array  # (NUM_BLUE_AGENTS, 3) int — subnet IDs per agent in CybORG alphabetical order
    comms_policy: chex.Array  # (MISSION_PHASES, NUM_SUBNETS, NUM_SUBNETS) bool — True = not connected

    max_steps: int
    num_hosts: int


@struct.dataclass
class CC4State:
    time: int
    done: chex.Array  # scalar bool
    mission_phase: chex.Array  # scalar int

    host_compromised: chex.Array  # (GLOBAL_MAX_HOSTS,) int — 0=None, 1=User, 2=Privileged
    host_services: chex.Array  # (GLOBAL_MAX_HOSTS, NUM_SERVICES) bool
    host_service_reliability: chex.Array  # (GLOBAL_MAX_HOSTS, NUM_SERVICES) int32 — 0-100
    host_decoys: chex.Array  # (GLOBAL_MAX_HOSTS, NUM_DECOY_TYPES) bool
    ot_service_stopped: chex.Array  # (GLOBAL_MAX_HOSTS,) bool

    red_sessions: chex.Array  # (NUM_RED_AGENTS, GLOBAL_MAX_HOSTS) bool
    red_privilege: chex.Array  # (NUM_RED_AGENTS, GLOBAL_MAX_HOSTS) int — 0/1/2
    red_discovered_hosts: chex.Array  # (NUM_RED_AGENTS, GLOBAL_MAX_HOSTS) bool
    red_scanned_hosts: chex.Array  # (NUM_RED_AGENTS, GLOBAL_MAX_HOSTS) bool

    red_activity_this_step: chex.Array  # (GLOBAL_MAX_HOSTS,) int — 0=None, 1=Scan, 2=Exploit
    host_activity_detected: chex.Array  # (GLOBAL_MAX_HOSTS,) bool
    host_has_malware: chex.Array  # (GLOBAL_MAX_HOSTS,) bool

    blocked_zones: chex.Array  # (NUM_SUBNETS, NUM_SUBNETS) bool
    messages: chex.Array  # (NUM_BLUE_AGENTS, NUM_BLUE_AGENTS, MESSAGE_LENGTH) float

    fsm_host_states: chex.Array  # (NUM_RED_AGENTS, GLOBAL_MAX_HOSTS) int — FSM state per red agent per host

    detection_randoms: chex.Array  # (MAX_DETECTION_RANDOMS,) float — precomputed sequence
    detection_random_index: chex.Array  # scalar int — next index to consume
    use_detection_randoms: chex.Array  # scalar bool — True = use sequence, False = use JAX RNG


def create_initial_const() -> CC4Const:
    return CC4Const(
        host_active=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
        host_subnet=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32),
        host_is_router=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
        host_is_server=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
        host_is_user=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
        subnet_adjacency=jnp.zeros((NUM_SUBNETS, NUM_SUBNETS), dtype=jnp.bool_),
        data_links=jnp.zeros((GLOBAL_MAX_HOSTS, GLOBAL_MAX_HOSTS), dtype=jnp.bool_),
        initial_services=jnp.zeros((GLOBAL_MAX_HOSTS, NUM_SERVICES), dtype=jnp.bool_),
        host_has_bruteforceable_user=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
        host_has_rfi=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
        host_respond_to_ping=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
        blue_agent_subnets=jnp.zeros((NUM_BLUE_AGENTS, NUM_SUBNETS), dtype=jnp.bool_),
        blue_agent_hosts=jnp.zeros((NUM_BLUE_AGENTS, GLOBAL_MAX_HOSTS), dtype=jnp.bool_),
        red_start_hosts=jnp.zeros(NUM_RED_AGENTS, dtype=jnp.int32),
        red_agent_active=jnp.zeros(NUM_RED_AGENTS, dtype=jnp.bool_),
        green_agent_host=jnp.full(GLOBAL_MAX_HOSTS, -1, dtype=jnp.int32),
        green_agent_active=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
        num_green_agents=0,
        phase_rewards=jnp.zeros((MISSION_PHASES, NUM_SUBNETS, 3), dtype=jnp.float32),
        phase_boundaries=jnp.zeros(MISSION_PHASES, dtype=jnp.int32),
        allowed_subnet_pairs=jnp.zeros((MISSION_PHASES, NUM_SUBNETS, NUM_SUBNETS), dtype=jnp.bool_),
        obs_host_map=jnp.full((NUM_SUBNETS, OBS_HOSTS_PER_SUBNET), GLOBAL_MAX_HOSTS, dtype=jnp.int32),
        blue_obs_subnets=jnp.full((NUM_BLUE_AGENTS, 3), -1, dtype=jnp.int32),
        comms_policy=jnp.zeros((MISSION_PHASES, NUM_SUBNETS, NUM_SUBNETS), dtype=jnp.bool_),
        max_steps=MAX_STEPS,
        num_hosts=0,
    )


def create_initial_state() -> CC4State:
    return CC4State(
        time=0,
        done=jnp.array(False),
        mission_phase=jnp.array(0, dtype=jnp.int32),
        host_compromised=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32),
        host_services=jnp.zeros((GLOBAL_MAX_HOSTS, NUM_SERVICES), dtype=jnp.bool_),
        host_service_reliability=jnp.full((GLOBAL_MAX_HOSTS, NUM_SERVICES), 100, dtype=jnp.int32),
        host_decoys=jnp.zeros((GLOBAL_MAX_HOSTS, NUM_DECOY_TYPES), dtype=jnp.bool_),
        ot_service_stopped=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
        red_sessions=jnp.zeros((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), dtype=jnp.bool_),
        red_privilege=jnp.zeros((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), dtype=jnp.int32),
        red_discovered_hosts=jnp.zeros((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), dtype=jnp.bool_),
        red_scanned_hosts=jnp.zeros((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), dtype=jnp.bool_),
        red_activity_this_step=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.int32),
        host_activity_detected=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
        host_has_malware=jnp.zeros(GLOBAL_MAX_HOSTS, dtype=jnp.bool_),
        blocked_zones=jnp.zeros((NUM_SUBNETS, NUM_SUBNETS), dtype=jnp.bool_),
        messages=jnp.zeros((NUM_BLUE_AGENTS, NUM_BLUE_AGENTS, MESSAGE_LENGTH), dtype=jnp.float32),
        fsm_host_states=jnp.zeros((NUM_RED_AGENTS, GLOBAL_MAX_HOSTS), dtype=jnp.int32),
        detection_randoms=jnp.zeros(MAX_DETECTION_RANDOMS, dtype=jnp.float32),
        detection_random_index=jnp.array(0, dtype=jnp.int32),
        use_detection_randoms=jnp.array(False),
    )

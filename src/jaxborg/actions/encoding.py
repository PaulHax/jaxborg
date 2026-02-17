import jax.numpy as jnp

from jaxborg.constants import GLOBAL_MAX_HOSTS, NUM_SUBNETS
from jaxborg.state import CC4Const

RED_SLEEP = 0
RED_DISCOVER_START = 1
RED_DISCOVER_END = RED_DISCOVER_START + NUM_SUBNETS
RED_SCAN_START = RED_DISCOVER_END
RED_SCAN_END = RED_SCAN_START + GLOBAL_MAX_HOSTS
RED_EXPLOIT_SSH_START = RED_SCAN_END
RED_EXPLOIT_SSH_END = RED_EXPLOIT_SSH_START + GLOBAL_MAX_HOSTS
RED_EXPLOIT_FTP_START = RED_EXPLOIT_SSH_END
RED_EXPLOIT_FTP_END = RED_EXPLOIT_FTP_START + GLOBAL_MAX_HOSTS
RED_EXPLOIT_HTTP_START = RED_EXPLOIT_FTP_END
RED_EXPLOIT_HTTP_END = RED_EXPLOIT_HTTP_START + GLOBAL_MAX_HOSTS
RED_EXPLOIT_HTTPS_START = RED_EXPLOIT_HTTP_END
RED_EXPLOIT_HTTPS_END = RED_EXPLOIT_HTTPS_START + GLOBAL_MAX_HOSTS
RED_EXPLOIT_HARAKA_START = RED_EXPLOIT_HTTPS_END
RED_EXPLOIT_HARAKA_END = RED_EXPLOIT_HARAKA_START + GLOBAL_MAX_HOSTS
RED_EXPLOIT_SQL_START = RED_EXPLOIT_HARAKA_END
RED_EXPLOIT_SQL_END = RED_EXPLOIT_SQL_START + GLOBAL_MAX_HOSTS
RED_EXPLOIT_ETERNALBLUE_START = RED_EXPLOIT_SQL_END
RED_EXPLOIT_ETERNALBLUE_END = RED_EXPLOIT_ETERNALBLUE_START + GLOBAL_MAX_HOSTS
RED_EXPLOIT_BLUEKEEP_START = RED_EXPLOIT_ETERNALBLUE_END
RED_EXPLOIT_BLUEKEEP_END = RED_EXPLOIT_BLUEKEEP_START + GLOBAL_MAX_HOSTS
RED_PRIVESC_START = RED_EXPLOIT_BLUEKEEP_END
RED_PRIVESC_END = RED_PRIVESC_START + GLOBAL_MAX_HOSTS
RED_IMPACT_START = RED_PRIVESC_END
RED_IMPACT_END = RED_IMPACT_START + GLOBAL_MAX_HOSTS

ACTION_TYPE_SLEEP = 0
ACTION_TYPE_DISCOVER = 1
ACTION_TYPE_SCAN = 2
ACTION_TYPE_EXPLOIT_SSH = 3
ACTION_TYPE_EXPLOIT_FTP = 4
ACTION_TYPE_EXPLOIT_HTTP = 5
ACTION_TYPE_EXPLOIT_HTTPS = 6
ACTION_TYPE_EXPLOIT_HARAKA = 7
ACTION_TYPE_EXPLOIT_SQL = 8
ACTION_TYPE_EXPLOIT_ETERNALBLUE = 9
ACTION_TYPE_EXPLOIT_BLUEKEEP = 10
ACTION_TYPE_PRIVESC = 11
ACTION_TYPE_IMPACT = 12

_EXPLOIT_RANGES = (
    (RED_EXPLOIT_SSH_START, RED_EXPLOIT_SSH_END, ACTION_TYPE_EXPLOIT_SSH),
    (RED_EXPLOIT_FTP_START, RED_EXPLOIT_FTP_END, ACTION_TYPE_EXPLOIT_FTP),
    (RED_EXPLOIT_HTTP_START, RED_EXPLOIT_HTTP_END, ACTION_TYPE_EXPLOIT_HTTP),
    (RED_EXPLOIT_HTTPS_START, RED_EXPLOIT_HTTPS_END, ACTION_TYPE_EXPLOIT_HTTPS),
    (RED_EXPLOIT_HARAKA_START, RED_EXPLOIT_HARAKA_END, ACTION_TYPE_EXPLOIT_HARAKA),
    (RED_EXPLOIT_SQL_START, RED_EXPLOIT_SQL_END, ACTION_TYPE_EXPLOIT_SQL),
    (RED_EXPLOIT_ETERNALBLUE_START, RED_EXPLOIT_ETERNALBLUE_END, ACTION_TYPE_EXPLOIT_ETERNALBLUE),
    (RED_EXPLOIT_BLUEKEEP_START, RED_EXPLOIT_BLUEKEEP_END, ACTION_TYPE_EXPLOIT_BLUEKEEP),
)

_ENCODE_MAP = {
    "ExploitRemoteService_cc4SSHBruteForce": RED_EXPLOIT_SSH_START,
    "ExploitRemoteService_cc4FTPDirectoryTraversal": RED_EXPLOIT_FTP_START,
    "ExploitRemoteService_cc4HTTPRFI": RED_EXPLOIT_HTTP_START,
    "ExploitRemoteService_cc4HTTPSRFI": RED_EXPLOIT_HTTPS_START,
    "ExploitRemoteService_cc4HarakaRCE": RED_EXPLOIT_HARAKA_START,
    "ExploitRemoteService_cc4SQLInjection": RED_EXPLOIT_SQL_START,
    "ExploitRemoteService_cc4EternalBlue": RED_EXPLOIT_ETERNALBLUE_START,
    "ExploitRemoteService_cc4BlueKeep": RED_EXPLOIT_BLUEKEEP_START,
    "PrivilegeEscalate": RED_PRIVESC_START,
    "Impact": RED_IMPACT_START,
}

BLUE_SLEEP = 0
BLUE_MONITOR = 1

BLUE_ACTION_TYPE_SLEEP = 0
BLUE_ACTION_TYPE_MONITOR = 1


def encode_red_action(action_name: str, target: int, agent_id: int) -> int:
    if action_name == "Sleep":
        return RED_SLEEP
    if action_name == "DiscoverRemoteSystems":
        return RED_DISCOVER_START + target
    if action_name == "DiscoverNetworkServices":
        return RED_SCAN_START + target
    if action_name == "Impact":
        return RED_IMPACT_START + target
    base = _ENCODE_MAP.get(action_name)
    if base is not None:
        return base + target
    raise NotImplementedError(f"Unknown red action {action_name}")


def decode_red_action(action_idx: int, agent_id: int, const: CC4Const):
    is_discover = (action_idx >= RED_DISCOVER_START) & (action_idx < RED_DISCOVER_END)
    is_scan = (action_idx >= RED_SCAN_START) & (action_idx < RED_SCAN_END)

    action_type = jnp.where(is_discover, ACTION_TYPE_DISCOVER, jnp.where(is_scan, ACTION_TYPE_SCAN, ACTION_TYPE_SLEEP))
    target_host = jnp.where(is_scan, action_idx - RED_SCAN_START, jnp.int32(-1))

    for start, end, atype in _EXPLOIT_RANGES:
        in_range = (action_idx >= start) & (action_idx < end)
        action_type = jnp.where(in_range, atype, action_type)
        target_host = jnp.where(in_range, action_idx - start, target_host)

    is_privesc = (action_idx >= RED_PRIVESC_START) & (action_idx < RED_PRIVESC_END)
    action_type = jnp.where(is_privesc, ACTION_TYPE_PRIVESC, action_type)
    target_host = jnp.where(is_privesc, action_idx - RED_PRIVESC_START, target_host)

    is_impact = (action_idx >= RED_IMPACT_START) & (action_idx < RED_IMPACT_END)
    action_type = jnp.where(is_impact, ACTION_TYPE_IMPACT, action_type)
    target_host = jnp.where(is_impact, action_idx - RED_IMPACT_START, target_host)

    target_subnet = jnp.where(is_discover, action_idx - RED_DISCOVER_START, jnp.int32(-1))
    return action_type, target_subnet, target_host


def encode_blue_action(action_name: str, target_host: int, agent_id: int) -> int:
    if action_name == "Sleep":
        return BLUE_SLEEP
    if action_name == "Monitor":
        return BLUE_MONITOR
    raise NotImplementedError(f"Unknown blue action {action_name}")


def decode_blue_action(action_idx: int, agent_id: int, const: CC4Const):
    action_type = jnp.where(action_idx == BLUE_MONITOR, BLUE_ACTION_TYPE_MONITOR, BLUE_ACTION_TYPE_SLEEP)
    return action_type

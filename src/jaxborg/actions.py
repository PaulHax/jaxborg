import jax
import jax.numpy as jnp
import chex

from jaxborg.constants import (
    GLOBAL_MAX_HOSTS,
    NUM_SUBNETS,
    NUM_SERVICES,
    SERVICE_IDS,
    DECOY_IDS,
    ACTIVITY_SCAN,
    ACTIVITY_EXPLOIT,
    COMPROMISE_NONE,
    COMPROMISE_USER,
    COMPROMISE_PRIVILEGED,
)
from jaxborg.state import CC4State, CC4Const

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

    action_type = jnp.where(is_discover, ACTION_TYPE_DISCOVER,
                  jnp.where(is_scan, ACTION_TYPE_SCAN, ACTION_TYPE_SLEEP))
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


def _has_any_session(
    session_hosts: chex.Array,
    const: CC4Const,
) -> chex.Array:
    """CC4's network is fully connected via routers; any session can reach any subnet."""
    return jnp.any(session_hosts & const.host_active)


def _apply_discover(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_subnet: chex.Array,
) -> CC4State:
    """Pingsweep: discover all ping-responding hosts in target_subnet."""
    session_hosts = state.red_sessions[agent_id]
    can_reach = _has_any_session(session_hosts, const)

    in_subnet = (const.host_subnet == target_subnet) & const.host_active
    pingable = in_subnet & const.host_respond_to_ping
    newly_discovered = pingable & can_reach

    new_discovered = state.red_discovered_hosts[agent_id] | newly_discovered
    red_discovered_hosts = state.red_discovered_hosts.at[agent_id].set(new_discovered)

    activity = jnp.where(newly_discovered, ACTIVITY_SCAN, state.red_activity_this_step)
    red_activity_this_step = jnp.where(can_reach, activity, state.red_activity_this_step)

    return state.replace(
        red_discovered_hosts=red_discovered_hosts,
        red_activity_this_step=red_activity_this_step,
    )


def _can_reach_subnet(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_subnet: chex.Array,
) -> chex.Array:
    session_hosts = state.red_sessions[agent_id]
    has_session = _has_any_session(session_hosts, const)
    active_sessions = session_hosts & const.host_active
    subnet_one_hot = jax.nn.one_hot(const.host_subnet, NUM_SUBNETS, dtype=jnp.bool_)
    session_subnets = jnp.any(active_sessions[:, None] & subnet_one_hot, axis=0)
    not_blocked = ~state.blocked_zones[target_subnet]
    can_route = jnp.any(session_subnets & not_blocked)
    return has_session & can_route


def _apply_scan(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    is_active = const.host_active[target_host]
    is_discovered = state.red_discovered_hosts[agent_id, target_host]
    target_subnet = const.host_subnet[target_host]
    can_reach = _can_reach_subnet(state, const, agent_id, target_subnet)

    success = is_active & is_discovered & can_reach

    red_scanned_hosts = state.red_scanned_hosts.at[agent_id, target_host].set(
        state.red_scanned_hosts[agent_id, target_host] | success
    )

    activity = jnp.where(
        success,
        state.red_activity_this_step.at[target_host].set(ACTIVITY_SCAN),
        state.red_activity_this_step,
    )

    return state.replace(
        red_scanned_hosts=red_scanned_hosts,
        red_activity_this_step=activity,
    )


SSH_SERVICE_IDX = SERVICE_IDS["SSHD"]
APACHE_SERVICE_IDX = SERVICE_IDS["APACHE2"]
MYSQL_SERVICE_IDX = SERVICE_IDS["MYSQLD"]
SMTP_SERVICE_IDX = SERVICE_IDS["SMTP"]

OTSERVICE_IDX = SERVICE_IDS["OTSERVICE"]

DECOY_HARAKA_IDX = DECOY_IDS["HarakaSMPT"]
DECOY_APACHE_IDX = DECOY_IDS["Apache"]
DECOY_TOMCAT_IDX = DECOY_IDS["Tomcat"]
DECOY_VSFTPD_IDX = DECOY_IDS["Vsftpd"]


def _exploit_common_preconditions(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> chex.Array:
    is_active = const.host_active[target_host]
    is_scanned = state.red_scanned_hosts[agent_id, target_host]
    target_subnet = const.host_subnet[target_host]
    can_reach = _can_reach_subnet(state, const, agent_id, target_subnet)
    no_session = ~state.red_sessions[agent_id, target_host]
    return is_active & is_scanned & can_reach & no_session


def _apply_exploit_success(
    state: CC4State,
    agent_id: int,
    target_host: chex.Array,
    success: chex.Array,
) -> CC4State:
    red_sessions = jnp.where(
        success,
        state.red_sessions.at[agent_id, target_host].set(True),
        state.red_sessions,
    )

    new_priv = jnp.where(
        success,
        jnp.maximum(state.red_privilege[agent_id, target_host], COMPROMISE_USER),
        state.red_privilege[agent_id, target_host],
    )
    red_privilege = jnp.where(
        success,
        state.red_privilege.at[agent_id, target_host].set(new_priv),
        state.red_privilege,
    )

    host_compromised = jnp.where(
        success,
        state.host_compromised.at[target_host].set(
            jnp.maximum(state.host_compromised[target_host], COMPROMISE_USER)
        ),
        state.host_compromised,
    )

    host_has_malware = jnp.where(
        success,
        state.host_has_malware.at[target_host].set(True),
        state.host_has_malware,
    )

    activity = jnp.where(
        success,
        state.red_activity_this_step.at[target_host].set(ACTIVITY_EXPLOIT),
        state.red_activity_this_step,
    )

    return state.replace(
        red_sessions=red_sessions,
        red_privilege=red_privilege,
        host_compromised=host_compromised,
        host_has_malware=host_has_malware,
        red_activity_this_step=activity,
    )


def _apply_exploit_ssh(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    base_ok = _exploit_common_preconditions(state, const, agent_id, target_host)
    has_ssh = state.host_services[target_host, SSH_SERVICE_IDX]
    has_bruteforceable = const.host_has_bruteforceable_user[target_host]
    success = base_ok & has_ssh & has_bruteforceable
    return _apply_exploit_success(state, agent_id, target_host, success)


def _apply_exploit_ftp(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    """FTPDirectoryTraversal: port 21, femitter process. No FTP service in CC4 → always fails.
    Decoy Vsftpd creates a fake service that traps this exploit."""
    base_ok = _exploit_common_preconditions(state, const, agent_id, target_host)
    has_decoy = state.host_decoys[target_host, DECOY_VSFTPD_IDX]
    # FTP service doesn't exist in the CC4 service model, so only decoy can trigger.
    # Decoy blocks the exploit (always fails), but still creates detection activity.
    detected = base_ok & has_decoy
    activity = jnp.where(
        detected,
        state.red_activity_this_step.at[target_host].set(ACTIVITY_EXPLOIT),
        state.red_activity_this_step,
    )
    return state.replace(red_activity_this_step=activity)


def _apply_exploit_http(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    """HTTPRFI: port 80, Apache2/WEBSERVER process. Needs 'rfi' property."""
    base_ok = _exploit_common_preconditions(state, const, agent_id, target_host)
    has_apache = state.host_services[target_host, APACHE_SERVICE_IDX]
    has_rfi = const.host_has_rfi[target_host]
    has_decoy = state.host_decoys[target_host, DECOY_APACHE_IDX]
    detected = base_ok & has_apache & has_decoy
    success = base_ok & has_apache & has_rfi & ~has_decoy
    activity = jnp.where(
        detected,
        state.red_activity_this_step.at[target_host].set(ACTIVITY_EXPLOIT),
        state.red_activity_this_step,
    )
    state = state.replace(red_activity_this_step=activity)
    return _apply_exploit_success(state, agent_id, target_host, success)


def _apply_exploit_https(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    """HTTPSRFI: port 443, webserver process. Needs 'rfi' property.
    No port-443 service in CC4 → only Tomcat decoy can trigger (and block)."""
    base_ok = _exploit_common_preconditions(state, const, agent_id, target_host)
    has_decoy = state.host_decoys[target_host, DECOY_TOMCAT_IDX]
    detected = base_ok & has_decoy
    activity = jnp.where(
        detected,
        state.red_activity_this_step.at[target_host].set(ACTIVITY_EXPLOIT),
        state.red_activity_this_step,
    )
    return state.replace(red_activity_this_step=activity)


def _apply_exploit_haraka(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    """HarakaRCE: port 25, SMTP process. Needs Haraka version < 2.8.9.
    In CC4, SMTP always starts at version 2.8.9 → real service never vulnerable.
    Decoy HarakaSMPT creates a fake vulnerable SMTP that traps and blocks."""
    base_ok = _exploit_common_preconditions(state, const, agent_id, target_host)
    has_smtp = state.host_services[target_host, SMTP_SERVICE_IDX]
    has_decoy = state.host_decoys[target_host, DECOY_HARAKA_IDX]
    detected = base_ok & has_smtp & has_decoy
    # Real SMTP is never vulnerable (version == 2.8.9), decoy blocks. Always fails.
    activity = jnp.where(
        detected,
        state.red_activity_this_step.at[target_host].set(ACTIVITY_EXPLOIT),
        state.red_activity_this_step,
    )
    return state.replace(red_activity_this_step=activity)


def _apply_exploit_sql(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    """SQLInjection: port 3390, MySQL process. Also requires port 80 or 443 (web frontend).
    test_exploit_works always returns True. No decoy blocks SQL specifically."""
    base_ok = _exploit_common_preconditions(state, const, agent_id, target_host)
    has_mysql = state.host_services[target_host, MYSQL_SERVICE_IDX]
    has_web = state.host_services[target_host, APACHE_SERVICE_IDX]
    success = base_ok & has_mysql & has_web
    return _apply_exploit_success(state, agent_id, target_host, success)


def _apply_exploit_eternalblue(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    """EternalBlue: port 139, SMB. Needs Windows + unpatched.
    CC4 is all Linux with no SMB service → always fails."""
    return state


def _apply_exploit_bluekeep(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    """BlueKeep: port 3389, RDP. Needs Windows + unpatched.
    CC4 is all Linux with no RDP service → always fails."""
    return state


def _apply_privesc(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    is_active = const.host_active[target_host]
    has_session = state.red_sessions[agent_id, target_host]
    not_already_privileged = state.red_privilege[agent_id, target_host] < COMPROMISE_PRIVILEGED
    success = is_active & has_session & not_already_privileged

    new_priv = jnp.where(success, COMPROMISE_PRIVILEGED, state.red_privilege[agent_id, target_host])
    red_privilege = jnp.where(
        success,
        state.red_privilege.at[agent_id, target_host].set(new_priv),
        state.red_privilege,
    )

    host_compromised = jnp.where(
        success,
        state.host_compromised.at[target_host].set(
            jnp.maximum(state.host_compromised[target_host], COMPROMISE_PRIVILEGED)
        ),
        state.host_compromised,
    )

    activity = jnp.where(
        success,
        state.red_activity_this_step.at[target_host].set(ACTIVITY_EXPLOIT),
        state.red_activity_this_step,
    )

    return state.replace(
        red_privilege=red_privilege,
        host_compromised=host_compromised,
        red_activity_this_step=activity,
    )


def _apply_impact(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    target_host: chex.Array,
) -> CC4State:
    is_active = const.host_active[target_host]
    has_session = state.red_sessions[agent_id, target_host]
    is_privileged = state.red_privilege[agent_id, target_host] >= COMPROMISE_PRIVILEGED
    has_ot = state.host_services[target_host, OTSERVICE_IDX]
    success = is_active & has_session & is_privileged & has_ot

    host_services = jnp.where(
        success,
        state.host_services.at[target_host, OTSERVICE_IDX].set(False),
        state.host_services,
    )

    ot_service_stopped = jnp.where(
        success,
        state.ot_service_stopped.at[target_host].set(True),
        state.ot_service_stopped,
    )

    activity = jnp.where(
        success,
        state.red_activity_this_step.at[target_host].set(ACTIVITY_EXPLOIT),
        state.red_activity_this_step,
    )

    return state.replace(
        host_services=host_services,
        ot_service_stopped=ot_service_stopped,
        red_activity_this_step=activity,
    )


_EXPLOIT_DISPATCH = (
    (ACTION_TYPE_EXPLOIT_SSH, _apply_exploit_ssh),
    (ACTION_TYPE_EXPLOIT_FTP, _apply_exploit_ftp),
    (ACTION_TYPE_EXPLOIT_HTTP, _apply_exploit_http),
    (ACTION_TYPE_EXPLOIT_HTTPS, _apply_exploit_https),
    (ACTION_TYPE_EXPLOIT_HARAKA, _apply_exploit_haraka),
    (ACTION_TYPE_EXPLOIT_SQL, _apply_exploit_sql),
    (ACTION_TYPE_EXPLOIT_ETERNALBLUE, _apply_exploit_eternalblue),
    (ACTION_TYPE_EXPLOIT_BLUEKEEP, _apply_exploit_bluekeep),
)


def apply_red_action(
    state: CC4State,
    const: CC4Const,
    agent_id: int,
    action_idx: int,
) -> CC4State:
    action_type, target_subnet, target_host = decode_red_action(action_idx, agent_id, const)

    state = jax.lax.cond(
        action_type == ACTION_TYPE_DISCOVER,
        lambda s: _apply_discover(s, const, agent_id, target_subnet),
        lambda s: s,
        state,
    )

    state = jax.lax.cond(
        action_type == ACTION_TYPE_SCAN,
        lambda s: _apply_scan(s, const, agent_id, target_host),
        lambda s: s,
        state,
    )

    for atype, apply_fn in _EXPLOIT_DISPATCH:
        state = jax.lax.cond(
            action_type == atype,
            lambda s, fn=apply_fn: fn(s, const, agent_id, target_host),
            lambda s: s,
            state,
        )

    state = jax.lax.cond(
        action_type == ACTION_TYPE_PRIVESC,
        lambda s: _apply_privesc(s, const, agent_id, target_host),
        lambda s: s,
        state,
    )

    state = jax.lax.cond(
        action_type == ACTION_TYPE_IMPACT,
        lambda s: _apply_impact(s, const, agent_id, target_host),
        lambda s: s,
        state,
    )

    return state


def encode_blue_action(action_name: str, target_host: int, agent_id: int) -> int:
    raise NotImplementedError("Subsystem 8+: blue action encoding")


def decode_blue_action(action_idx: int, agent_id: int, const: CC4Const):
    raise NotImplementedError("Subsystem 8+: blue action decoding")


def apply_blue_action(state: CC4State, const: CC4Const, agent_id: int, action_idx: int) -> CC4State:
    raise NotImplementedError("Subsystem 8+: blue action application")

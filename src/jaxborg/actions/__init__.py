import jax

from jaxborg.actions.encoding import (
    ACTION_TYPE_DISCOVER,
    ACTION_TYPE_EXPLOIT_BLUEKEEP,
    ACTION_TYPE_EXPLOIT_ETERNALBLUE,
    ACTION_TYPE_EXPLOIT_FTP,
    ACTION_TYPE_EXPLOIT_HARAKA,
    ACTION_TYPE_EXPLOIT_HTTP,
    ACTION_TYPE_EXPLOIT_HTTPS,
    ACTION_TYPE_EXPLOIT_SQL,
    ACTION_TYPE_EXPLOIT_SSH,
    ACTION_TYPE_IMPACT,
    ACTION_TYPE_PRIVESC,
    ACTION_TYPE_SCAN,
    decode_red_action,
)
from jaxborg.actions.red_discover import apply_discover
from jaxborg.actions.red_exploit import (
    apply_exploit_bluekeep,
    apply_exploit_eternalblue,
    apply_exploit_ftp,
    apply_exploit_haraka,
    apply_exploit_http,
    apply_exploit_https,
    apply_exploit_sql,
    apply_exploit_ssh,
)
from jaxborg.actions.red_impact import apply_impact
from jaxborg.actions.red_privesc import apply_privesc
from jaxborg.actions.red_scan import apply_scan
from jaxborg.state import CC4Const, CC4State

_EXPLOIT_DISPATCH = (
    (ACTION_TYPE_EXPLOIT_SSH, apply_exploit_ssh),
    (ACTION_TYPE_EXPLOIT_FTP, apply_exploit_ftp),
    (ACTION_TYPE_EXPLOIT_HTTP, apply_exploit_http),
    (ACTION_TYPE_EXPLOIT_HTTPS, apply_exploit_https),
    (ACTION_TYPE_EXPLOIT_HARAKA, apply_exploit_haraka),
    (ACTION_TYPE_EXPLOIT_SQL, apply_exploit_sql),
    (ACTION_TYPE_EXPLOIT_ETERNALBLUE, apply_exploit_eternalblue),
    (ACTION_TYPE_EXPLOIT_BLUEKEEP, apply_exploit_bluekeep),
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
        lambda s: apply_discover(s, const, agent_id, target_subnet),
        lambda s: s,
        state,
    )

    state = jax.lax.cond(
        action_type == ACTION_TYPE_SCAN,
        lambda s: apply_scan(s, const, agent_id, target_host),
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
        lambda s: apply_privesc(s, const, agent_id, target_host),
        lambda s: s,
        state,
    )

    state = jax.lax.cond(
        action_type == ACTION_TYPE_IMPACT,
        lambda s: apply_impact(s, const, agent_id, target_host),
        lambda s: s,
        state,
    )

    return state


def apply_blue_action(state: CC4State, const: CC4Const, agent_id: int, action_idx: int) -> CC4State:
    return state

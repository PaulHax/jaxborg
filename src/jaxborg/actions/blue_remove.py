import jax.numpy as jnp

from jaxborg.actions.session_counts import effective_session_counts
from jaxborg.constants import COMPROMISE_NONE, COMPROMISE_USER, NUM_RED_AGENTS
from jaxborg.state import CC4Const, CC4State


def apply_blue_remove(state: CC4State, const: CC4Const, agent_id: int, target_host: int) -> CC4State:
    covers_host = const.blue_agent_hosts[agent_id, target_host]
    blue_budget = state.blue_suspicious_pid_budget[agent_id, target_host]
    fallback_budget = jnp.sum(state.red_suspicious_process_count[:, target_host])
    start_budget = jnp.maximum(blue_budget, fallback_budget)
    has_suspicious_process = start_budget > 0

    session_count_before = effective_session_counts(state)
    new_session_count = session_count_before
    new_suspicious_count = state.red_suspicious_process_count
    new_privilege = state.red_privilege
    remaining_budget = start_budget
    for r in range(NUM_RED_AGENTS):
        count = new_session_count[r, target_host]
        is_user = new_privilege[r, target_host] == COMPROMISE_USER
        suspicious_count = new_suspicious_count[r, target_host]
        target_is_scanned = state.red_scanned_hosts[r, target_host]
        has_valid_signal = (suspicious_count > 0) | (
            (remaining_budget > count)
            & (
                state.host_suspicious_process[target_host]
                | (~target_is_scanned & state.host_activity_detected[target_host])
                | (~state.host_has_malware[target_host] & target_is_scanned)
            )
        )
        should_clear = covers_host & is_user & has_valid_signal & (remaining_budget > 0)
        remove_n = jnp.where(should_clear, jnp.minimum(count, remaining_budget), 0)
        has_abstract = state.red_session_is_abstract[r, target_host]
        budget_gap = blue_budget - suspicious_count
        prefer_blue_budget = has_abstract & (blue_budget == count) & (budget_gap >= 2)
        max_removable = jnp.where(
            prefer_blue_budget,
            remaining_budget,
            jnp.where(
                suspicious_count > 0,
                suspicious_count,
                jnp.where(remaining_budget > count, count, suspicious_count),
            ),
        )
        remove_n = jnp.minimum(remove_n, max_removable)
        non_abstract_count = count - has_abstract.astype(jnp.int32)
        all_suspicious = (suspicious_count >= count) | prefer_blue_budget
        abstract_safe = has_abstract & (non_abstract_count > 0) & (remove_n > non_abstract_count) & ~all_suspicious
        remove_n = jnp.where(abstract_safe, non_abstract_count, remove_n)
        remaining_budget = jnp.maximum(remaining_budget - remove_n, 0)
        count_after = jnp.maximum(count - remove_n, 0)
        new_session_count = jnp.where(
            should_clear,
            new_session_count.at[r, target_host].set(count_after),
            new_session_count,
        )

        priv_after = jnp.where(count_after > 0, COMPROMISE_USER, COMPROMISE_NONE)
        new_privilege = jnp.where(
            should_clear,
            new_privilege.at[r, target_host].set(priv_after),
            new_privilege,
        )
        suspicious_after = jnp.maximum(suspicious_count - remove_n, 0)
        new_suspicious_count = jnp.where(
            should_clear,
            new_suspicious_count.at[r, target_host].set(suspicious_after),
            new_suspicious_count,
        )

    new_sessions = new_session_count > 0
    new_multiple = new_session_count > 1
    new_many = new_session_count > 2

    any_compromised = jnp.any(new_privilege[:, target_host] > 0)
    new_host_compromised = jnp.where(
        covers_host & has_suspicious_process & ~any_compromised,
        state.host_compromised.at[target_host].set(COMPROMISE_NONE),
        state.host_compromised,
    )
    had_any_sessions = jnp.any(session_count_before > 0, axis=1)
    has_any_sessions_now = jnp.any(new_session_count > 0, axis=1)
    cleared_all_sessions = had_any_sessions & ~has_any_sessions_now
    sessions_lost_on_target = (session_count_before[:, target_host] > 0) & (new_session_count[:, target_host] == 0)
    via_target = state.red_scanned_via == jnp.int32(target_host)
    via_clear = sessions_lost_on_target[:, None] & via_target
    full_clear = cleared_all_sessions[:, None]
    new_scanned_hosts = state.red_scanned_hosts & ~(full_clear | via_clear)
    new_scanned_via = jnp.where(full_clear | via_clear, -1, state.red_scanned_via)
    session_hosts = new_sessions & const.host_active[None, :]
    abstract_hosts = state.red_session_is_abstract & session_hosts
    has_abstract = jnp.any(abstract_hosts, axis=1)
    first_abstract = jnp.argmax(abstract_hosts, axis=1)
    first_session = jnp.argmax(session_hosts, axis=1)
    fallback_anchor = jnp.where(has_abstract, first_abstract, first_session)
    fallback_anchor = jnp.where(has_any_sessions_now, fallback_anchor, -1)
    anchor_host_cleared = sessions_lost_on_target & (state.red_scan_anchor_host == jnp.int32(target_host))
    removed_anchor = cleared_all_sessions | anchor_host_cleared
    red_scan_anchor_host = jnp.where(
        removed_anchor,
        fallback_anchor,
        state.red_scan_anchor_host,
    )
    red_scan_anchor_host = jnp.where(has_any_sessions_now, red_scan_anchor_host, -1)
    any_suspicious_after = jnp.any(new_suspicious_count[:, target_host] > 0)
    new_suspicious_process = jnp.where(
        covers_host,
        state.host_suspicious_process.at[target_host].set(any_suspicious_after),
        state.host_suspicious_process,
    )
    sessions_cleared_on_host = (session_count_before[:, target_host] > 0) & (new_session_count[:, target_host] == 0)
    abstract_update = state.red_session_is_abstract.at[:, target_host].set(
        state.red_session_is_abstract[:, target_host] & ~sessions_cleared_on_host
    )
    red_session_is_abstract = jnp.where(
        covers_host,
        abstract_update,
        state.red_session_is_abstract,
    )
    return state.replace(
        red_sessions=new_sessions,
        red_session_count=new_session_count,
        red_session_multiple=new_multiple,
        red_session_many=new_many,
        red_suspicious_process_count=new_suspicious_count,
        red_privilege=new_privilege,
        red_scan_anchor_host=red_scan_anchor_host,
        red_scanned_hosts=new_scanned_hosts,
        red_scanned_via=new_scanned_via,
        host_compromised=new_host_compromised,
        host_suspicious_process=new_suspicious_process,
        blue_suspicious_pid_budget=state.blue_suspicious_pid_budget,
        red_session_is_abstract=red_session_is_abstract,
    )

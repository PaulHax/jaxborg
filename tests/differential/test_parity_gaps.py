import numpy as np
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent

from tests.differential.harness import CC4DifferentialHarness


def _known_hosts_for_red_agent(harness: CC4DifferentialHarness, red_agent_idx: int) -> set[str]:
    state = harness.cyborg_env.environment_controller.state
    aspace = harness.cyborg_env.environment_controller.agent_interfaces[f"red_agent_{red_agent_idx}"].action_space
    known: set[str] = set()
    for ip, is_known in getattr(aspace, "ip_address", {}).items():
        if not is_known:
            continue
        hostname = state.ip_addresses.get(ip)
        if hostname is not None:
            known.add(hostname)
    return known


def _session_hosts_for_red_agent(harness: CC4DifferentialHarness, red_agent_idx: int) -> set[str]:
    sessions = harness.cyborg_env.environment_controller.state.sessions[f"red_agent_{red_agent_idx}"]
    return {sess.hostname for sess in sessions.values()}


def test_PARITY_GAP_seed0_step3_red_scanned_hosts_matches_cyborg():
    # PARITY_GAP: seed=0 step=3 red_scanned_hosts diverges (cyborg={21} jax={18,21})
    harness = CC4DifferentialHarness(
        seed=0,
        max_steps=500,
        blue_cls=SleepAgent,
        green_cls=EnterpriseGreenAgent,
        red_cls=FiniteStateRedAgent,
        sync_green_rng=True,
    )
    harness.reset()

    step_result = None
    for _ in range(4):
        step_result = harness.full_step()

    red_scanned_diffs = [d for d in step_result.diffs if d.field_name == "red_scanned_hosts"]
    assert not red_scanned_diffs, f"Unexpected red_scanned_hosts diff: {red_scanned_diffs}"


def test_PARITY_GAP_seed0_step22_host_has_malware_matches_cyborg():
    # PARITY_GAP: seed=0 step=22 host_has_malware diverges (host_3 cyborg=True jax=False)
    harness = CC4DifferentialHarness(
        seed=0,
        max_steps=500,
        blue_cls=SleepAgent,
        green_cls=EnterpriseGreenAgent,
        red_cls=FiniteStateRedAgent,
        sync_green_rng=True,
    )
    harness.reset()

    step_result = None
    for _ in range(23):
        step_result = harness.full_step()

    malware_diffs = [d for d in step_result.diffs if d.field_name == "host_has_malware"]
    assert not malware_diffs, f"Unexpected host_has_malware diff: {malware_diffs}"


def test_PARITY_GAP_seed4_step5_host_service_reliability_matches_cyborg():
    # PARITY_GAP: seed=4 step=5 host_service_reliability diverges on host_19.
    harness = CC4DifferentialHarness(
        seed=4,
        max_steps=500,
        blue_cls=SleepAgent,
        green_cls=EnterpriseGreenAgent,
        red_cls=FiniteStateRedAgent,
        sync_green_rng=True,
    )
    harness.reset()

    step_result = None
    for _ in range(6):
        step_result = harness.full_step()

    reliability_diffs = [d for d in step_result.diffs if d.field_name == "host_service_reliability"]
    assert not reliability_diffs, f"Unexpected host_service_reliability diff: {reliability_diffs}"


def test_PARITY_GAP_seed7_step21_ot_service_stopped_matches_cyborg():
    # PARITY_GAP: seed=7 step=21 ot_service_stopped diverges on host_45.
    harness = CC4DifferentialHarness(
        seed=7,
        max_steps=500,
        blue_cls=SleepAgent,
        green_cls=EnterpriseGreenAgent,
        red_cls=FiniteStateRedAgent,
        sync_green_rng=True,
    )
    harness.reset()

    step_result = None
    for _ in range(22):
        step_result = harness.full_step()

    impact_diffs = [d for d in step_result.diffs if d.field_name == "ot_service_stopped"]
    assert not impact_diffs, f"Unexpected ot_service_stopped diff: {impact_diffs}"


def test_PARITY_GAP_seed7_step18_red_privilege_matches_cyborg():
    # PARITY_GAP: seed=7 step=18 red_privilege diverges (red_4_host_45 cyborg=1 jax=2).
    harness = CC4DifferentialHarness(
        seed=7,
        max_steps=500,
        blue_cls=SleepAgent,
        green_cls=EnterpriseGreenAgent,
        red_cls=FiniteStateRedAgent,
        sync_green_rng=True,
    )
    harness.reset()

    step_result = None
    for _ in range(19):
        step_result = harness.full_step()

    priv_diffs = [d for d in step_result.diffs if d.field_name == "red_privilege"]
    assert not priv_diffs, f"Unexpected red_privilege diff: {priv_diffs}"


def test_PARITY_GAP_seed12_step22_privesc_discovers_info_link_hosts_matches_cyborg():
    # PARITY_GAP: seed=12 step=22 red_discovered_hosts diverged for red_agent_0.
    harness = CC4DifferentialHarness(
        seed=12,
        max_steps=500,
        blue_cls=SleepAgent,
        green_cls=EnterpriseGreenAgent,
        red_cls=FiniteStateRedAgent,
        sync_green_rng=True,
    )
    harness.reset()

    source_host = "contractor_network_subnet_server_host_0"
    expected_new_hosts = {
        "public_access_zone_subnet_server_host_0",
        "restricted_zone_a_subnet_server_host_0",
        "restricted_zone_b_subnet_server_host_0",
    }

    source_idx = harness.mappings.hostname_to_idx[source_host]
    expected_new_idxs = {harness.mappings.hostname_to_idx[h] for h in expected_new_hosts}
    info_link_targets = set(np.where(np.array(harness.jax_const.host_info_links[source_idx]))[0])
    assert expected_new_idxs.issubset(info_link_targets)

    for _ in range(22):
        harness.full_step()

    known_before = _known_hosts_for_red_agent(harness, 0)
    jax_before = np.array(harness.jax_state.red_discovered_hosts[0])
    assert source_host in known_before
    for host in expected_new_hosts:
        assert host not in known_before
        assert not bool(jax_before[harness.mappings.hostname_to_idx[host]])

    step_result = harness.full_step()
    action = harness.cyborg_env.environment_controller.action["red_agent_0"][0]
    assert type(action).__name__ == "PrivilegeEscalate"
    assert getattr(action, "hostname", None) == source_host

    known_after = _known_hosts_for_red_agent(harness, 0)
    jax_after = np.array(harness.jax_state.red_discovered_hosts[0])
    for host in expected_new_hosts:
        assert host in known_after
        assert bool(jax_after[harness.mappings.hostname_to_idx[host]])

    discovered_diffs = [d for d in step_result.diffs if d.field_name == "red_discovered_hosts"]
    assert not discovered_diffs, f"Unexpected red_discovered_hosts diff: {discovered_diffs}"


def test_PARITY_GAP_seed16_step1_reassigned_session_is_discovered_matches_cyborg():
    # PARITY_GAP: seed=16 step=1 red_discovered_hosts diverged (cyborg={19,20} jax={19}).
    harness = CC4DifferentialHarness(
        seed=16,
        max_steps=500,
        blue_cls=SleepAgent,
        green_cls=EnterpriseGreenAgent,
        red_cls=FiniteStateRedAgent,
        sync_green_rng=True,
    )
    harness.reset()

    host = "contractor_network_subnet_user_host_6"
    host_idx = harness.mappings.hostname_to_idx[host]

    known_before = _known_hosts_for_red_agent(harness, 0)
    sessions_before = _session_hosts_for_red_agent(harness, 0)
    jax_sessions_before = np.array(harness.jax_state.red_sessions[0])
    jax_known_before = np.array(harness.jax_state.red_discovered_hosts[0])

    assert host not in known_before
    assert host not in sessions_before
    assert not bool(jax_sessions_before[host_idx])
    assert not bool(jax_known_before[host_idx])

    harness.full_step()
    step_result = harness.full_step()

    action = harness.cyborg_env.environment_controller.action["red_agent_0"][0]
    assert type(action).__name__ == "PrivilegeEscalate"
    assert getattr(action, "hostname", None) == "contractor_network_subnet_user_host_5"

    known_after = _known_hosts_for_red_agent(harness, 0)
    sessions_after = _session_hosts_for_red_agent(harness, 0)
    jax_sessions_after = np.array(harness.jax_state.red_sessions[0])
    jax_known_after = np.array(harness.jax_state.red_discovered_hosts[0])

    assert host in sessions_after
    assert host in known_after
    assert bool(jax_sessions_after[host_idx])
    assert bool(jax_known_after[host_idx])

    cy_known_idxs = {harness.mappings.hostname_to_idx[h] for h in known_after}
    jax_known_idxs = set(np.where(jax_known_after)[0].tolist())
    assert cy_known_idxs == jax_known_idxs

    discovered_diffs = [d for d in step_result.diffs if d.field_name == "red_discovered_hosts"]
    assert not discovered_diffs, f"Unexpected red_discovered_hosts diff: {discovered_diffs}"

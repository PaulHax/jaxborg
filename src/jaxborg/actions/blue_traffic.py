from jaxborg.state import CC4Const, CC4State


def apply_block_traffic(state: CC4State, const: CC4Const, agent_id: int, src_subnet: int, dst_subnet: int) -> CC4State:
    blocked_zones = state.blocked_zones.at[dst_subnet, src_subnet].set(True)
    return state.replace(blocked_zones=blocked_zones)


def apply_allow_traffic(state: CC4State, const: CC4Const, agent_id: int, src_subnet: int, dst_subnet: int) -> CC4State:
    blocked_zones = state.blocked_zones.at[dst_subnet, src_subnet].set(False)
    return state.replace(blocked_zones=blocked_zones)

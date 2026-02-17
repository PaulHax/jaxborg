from jaxborg.state import CC4Const, CC4State


def apply_block_traffic(state: CC4State, const: CC4Const, agent_id: int, src_subnet: int, dst_subnet: int) -> CC4State:
    raise NotImplementedError("Subsystem 17: block traffic zones")


def apply_allow_traffic(state: CC4State, const: CC4Const, agent_id: int, src_subnet: int, dst_subnet: int) -> CC4State:
    raise NotImplementedError("Subsystem 17: allow traffic zones")

from jaxborg.state import CC4State, CC4Const


def encode_blue_action(action_name: str, target_host: int, agent_id: int) -> int:
    raise NotImplementedError("Subsystem 8+: blue action encoding")


def decode_blue_action(action_idx: int, agent_id: int, const: CC4Const):
    raise NotImplementedError("Subsystem 8+: blue action decoding")


def apply_blue_action(state: CC4State, const: CC4Const, agent_id: int, action_idx: int) -> CC4State:
    raise NotImplementedError("Subsystem 8+: blue action application")


def encode_red_action(action_name: str, target_host: int, agent_id: int) -> int:
    raise NotImplementedError("Subsystem 2+: red action encoding")


def decode_red_action(action_idx: int, agent_id: int, const: CC4Const):
    raise NotImplementedError("Subsystem 2+: red action decoding")


def apply_red_action(state: CC4State, const: CC4Const, agent_id: int, action_idx: int) -> CC4State:
    raise NotImplementedError("Subsystem 2+: red action application")

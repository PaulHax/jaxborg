import chex

from jaxborg.state import CC4Const, CC4State


def get_blue_obs(state: CC4State, const: CC4Const, agent_id: int) -> chex.Array:
    raise NotImplementedError("Subsystem 14: blue observation encoding")


def get_red_obs(state: CC4State, const: CC4Const, agent_id: int) -> chex.Array:
    raise NotImplementedError("Subsystem 14: red observation encoding")

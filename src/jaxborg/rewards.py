import chex

from jaxborg.state import CC4Const, CC4State


def compute_rewards(state: CC4State, const: CC4Const) -> chex.Array:
    raise NotImplementedError("Subsystem 13: reward computation")

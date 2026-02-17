from tests.differential.harness import StateSnapshot, StateDiff


def compare_snapshots(cyborg: StateSnapshot, jax: StateSnapshot) -> list[StateDiff]:
    raise NotImplementedError("Subsystem 1+: state comparison")

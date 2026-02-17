from tests.differential.harness import StateDiff, StateSnapshot


def compare_snapshots(cyborg: StateSnapshot, jax: StateSnapshot) -> list[StateDiff]:
    raise NotImplementedError("Subsystem 1+: state comparison")

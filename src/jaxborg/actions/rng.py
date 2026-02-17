import jax

from jaxborg.state import CC4State


def sample_detection_random(state: CC4State, key: jax.Array):
    """Return (random_float, updated_state). Uses precomputed sequence if enabled, else JAX RNG."""
    return jax.lax.cond(
        state.use_detection_randoms,
        lambda s: _from_sequence(s),
        lambda s: (jax.random.uniform(key), s),
        state,
    )


def _from_sequence(state: CC4State):
    idx = state.detection_random_index
    val = state.detection_randoms[idx]
    new_state = state.replace(detection_random_index=idx + 1)
    return val, new_state

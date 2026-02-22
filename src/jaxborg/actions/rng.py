import jax
import jax.numpy as jnp

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


def sample_green_random(state: CC4State, time, host_idx, field_idx, key, *, int_range=None):
    """Return a random value. Uses precomputed green_randoms if enabled, else JAX RNG.

    When int_range is provided, returns an int32 in [0, int_range).
    When int_range is None, returns a float32 uniform in [0, 1).
    """
    if int_range is not None:

        def from_precomputed(_):
            v = state.green_randoms[time, host_idx, field_idx]
            return jnp.floor(v * int_range).astype(jnp.int32)

        def from_rng(_):
            return jax.random.randint(key, (), 0, jnp.maximum(int_range, 1))
    else:

        def from_precomputed(_):
            return state.green_randoms[time, host_idx, field_idx]

        def from_rng(_):
            return jax.random.uniform(key)

    return jax.lax.cond(state.use_green_randoms, from_precomputed, from_rng, None)

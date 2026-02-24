import jax.numpy as jnp


def effective_session_counts(state):
    """Best-effort exact count with backward-compatible fallback from legacy flags."""
    inferred = (
        state.red_sessions.astype(jnp.int32)
        + state.red_session_multiple.astype(jnp.int32)
        + state.red_session_many.astype(jnp.int32)
    )
    return jnp.maximum(state.red_session_count, inferred)

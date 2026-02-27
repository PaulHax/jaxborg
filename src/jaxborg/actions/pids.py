import jax.numpy as jnp


def append_pid_to_row(pid_row, pid):
    already_present = jnp.any(pid_row == pid)
    empty_mask = pid_row < 0
    has_empty = jnp.any(empty_mask)
    insert_idx = jnp.argmax(empty_mask)
    updated = pid_row.at[insert_idx].set(pid)
    return jnp.where(already_present | ~has_empty, pid_row, updated)


def pid_row_contains(pid_row, pid):
    return (pid >= 0) & jnp.any(pid_row == pid)


def remove_pid_from_row(pid_row, pid):
    return jnp.where(pid_row == pid, -1, pid_row)


def count_pid_matches(pid_row, candidate_pids):
    return jnp.sum((candidate_pids[:, None] >= 0) & (pid_row[None, :] == candidate_pids[:, None]))


def first_valid_pid(pid_row):
    valid = pid_row >= 0
    idx = jnp.argmax(valid)
    return jnp.where(jnp.any(valid), pid_row[idx], -1)

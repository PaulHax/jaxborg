"""Differential fuzzer: runs episodes across many seeds to find CybORG/JAX divergences."""

import time
from dataclasses import dataclass

from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent

from tests.differential.harness import CC4DifferentialHarness
from tests.differential.state_comparator import _ERROR_FIELDS, format_diffs


@dataclass
class MismatchReport:
    seed: int
    step: int
    field_name: str
    host_or_agent: str
    cyborg_value: object
    jax_value: object
    all_diffs_str: str

    def __str__(self):
        return (
            f"Mismatch at seed={self.seed}, step={self.step}: "
            f"{self.field_name} [{self.host_or_agent}] "
            f"cyborg={self.cyborg_value} jax={self.jax_value}\n"
            f"All diffs:\n{self.all_diffs_str}"
        )


def run_differential_fuzz(
    seeds=range(20),
    max_steps_per_seed=100,
    verbose=False,
) -> MismatchReport | None:
    wall_start = time.time()
    for seed in seeds:
        seed_start = time.time()
        if verbose:
            print(f"--- Seed {seed} ---")

        harness = CC4DifferentialHarness(
            seed=seed,
            max_steps=500,
            blue_cls=SleepAgent,
            green_cls=EnterpriseGreenAgent,
            red_cls=FiniteStateRedAgent,
            sync_green_rng=True,
        )
        harness.reset()

        for t in range(max_steps_per_seed):
            result = harness.full_step()

            error_diffs = [d for d in result.diffs if d.field_name in _ERROR_FIELDS]
            if error_diffs:
                d = error_diffs[0]
                return MismatchReport(
                    seed=seed,
                    step=t,
                    field_name=d.field_name,
                    host_or_agent=d.host_or_agent,
                    cyborg_value=d.cyborg_value,
                    jax_value=d.jax_value,
                    all_diffs_str=format_diffs(result.diffs),
                )

        if verbose:
            elapsed = time.time() - seed_start
            print(f"  Seed {seed}: {max_steps_per_seed} steps clean ({elapsed:.1f}s)")

    if verbose:
        total = time.time() - wall_start
        print(f"\nTotal: {total:.1f}s")
    return None


if __name__ == "__main__":
    print("Running differential fuzzer...")
    report = run_differential_fuzz(verbose=True)
    if report:
        print(f"\nFOUND MISMATCH:\n{report}")
    else:
        print("\nNo mismatches found across all seeds!")

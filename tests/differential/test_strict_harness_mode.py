import inspect

from tests.differential.fuzzer import run_differential_fuzz
from tests.differential.harness import CC4DifferentialHarness


def test_harness_constructor_is_strict_only():
    params = inspect.signature(CC4DifferentialHarness.__init__).parameters
    assert "strict_differential" not in params


def test_fuzzer_api_is_strict_only():
    params = inspect.signature(run_differential_fuzz).parameters
    assert "strict_differential" not in params

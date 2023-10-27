import pytest
import numpy as np
from evolutionary_programming.objective_function import (
    RastriginFunction,
    BaseFunction,
)


@pytest.fixture
def rastrigin_fn() -> BaseFunction:
    return RastriginFunction(2)


def test_rastrigin_function_type(rastrigin_fn):
    assert isinstance(rastrigin_fn, BaseFunction)


def test_rastrigin_function_evaluation_low_value(rastrigin_fn):
    assert rastrigin_fn.evaluate(np.array([0, 0, 0], dtype='d')) == 0.0


def test_rastrigin_function_evaluation_high_value(rastrigin_fn):
    assert rastrigin_fn.evaluate(np.array([4.5, 4.5])) == 80.5

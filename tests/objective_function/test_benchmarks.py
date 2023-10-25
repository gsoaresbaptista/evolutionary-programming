import numpy as np
from evolutionary_programming.objective_function import (
    RastriginFunction,
    BaseFunction,
)


def test_rastrigin_function_type():
    fn = RastriginFunction(3)
    assert isinstance(fn, BaseFunction)


def test_rastrigin_function_creation():
    assert RastriginFunction(3)


def test_rastrigin_function_evaluation_low_value():
    fn = RastriginFunction(3)
    assert fn.evaluate(np.array([0, 0, 0], dtype='d')) == 0.0


def test_rastrigin_function_evaluation_high_value():
    fn = RastriginFunction(2)
    assert fn.evaluate(np.array([4.5, 4.5])) == 80.5

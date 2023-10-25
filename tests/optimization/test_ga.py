from evolutionary_programming.optimization import (
    GeneticAlgorithm,
    PopulationBasedOptimizer,
)
from evolutionary_programming.objective_function import RastriginFunction


def test_ga_inheritance():
    ga = GeneticAlgorithm(1, 1, [-1], [1])
    assert isinstance(ga, PopulationBasedOptimizer)


def test_ga_creation():
    assert GeneticAlgorithm(1, 1, [-1], [1])


def test_ga_optimization():
    dims = 5
    individuals = 20
    min_bounds = [-5.2 for _ in range(dims)]
    max_bounds = [5.2 for _ in range(dims)]
    ga = GeneticAlgorithm(individuals, dims, min_bounds, max_bounds)
    fn = RastriginFunction(dims)
    ga.optimize(1, fn)
    assert ga.best_fitness < 201.7664509

from evolutionary_programming.optimization import (
    ParticleSwarm,
    PopulationBasedOptimizer,
)
from evolutionary_programming.objective_function import RastriginFunction


def test_pso_inheritance():
    pso = ParticleSwarm(1, 1, [-1], [1])
    assert isinstance(pso, PopulationBasedOptimizer)


def test_pso_creation():
    assert ParticleSwarm(1, 1, [-1], [1])


def test_pso_optimization():
    dims = 5
    particles = 20
    min_bounds = [-5.2 for _ in range(dims)]
    max_bounds = [5.2 for _ in range(dims)]
    pso = ParticleSwarm(particles, dims, min_bounds, max_bounds)
    fn = RastriginFunction(dims)
    pso.optimize(1, fn)
    assert pso.best_value < 201.7664509

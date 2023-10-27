from evolutionary_programming.optimization import GeneticAlgorithm
from evolutionary_programming.objective_function import RastriginFunction

dim = 100000
min_bounds = [-5.2 for _ in range(dim)]
max_bounds = [5.2 for _ in range(dim)]
fn = RastriginFunction(dim)
ga = GeneticAlgorithm(1000, dim, min_bounds, max_bounds)
ga.optimize(20, fn)

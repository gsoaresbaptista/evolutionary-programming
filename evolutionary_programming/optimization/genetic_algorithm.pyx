import numpy as np
cimport numpy as np
from .base_optimizer cimport PopulationBasedOptimizer
from evolutionary_programming.objective_function.base_function cimport BaseFunction


np.import_array()


cdef class GeneticAlgorithm(PopulationBasedOptimizer):
    def __cinit__(
        self,
        int n_individuals,
        int n_dims,
        list min_bounds,
        list max_bounds,
        double mutation_probability = 0.01,
    ):
        super().__init__(n_individuals, n_dims, min_bounds, max_bounds)
        self._mutation_probability = mutation_probability
        self._children_shape = (self._n_individuals, self._n_dims)
        self._init_individuals()

    cpdef void _init_individuals(self) except *:
        # create individuals
        self._individuals_fitness = np.full(self._n_individuals, DBL_MAX)
        self._individuals = np.random.uniform(
            self._min_bounds, self._max_bounds, self._children_shape)
        # set the best individual, temporary
        self.best_individual = self._individuals[0]
        self.best_fitness = self._individuals_fitness[0]

    cpdef void _fitness_compute(self, BaseFunction function) except *:
        cdef double[:, :] individuals = self._individuals

        for i in range(self._n_individuals):
            self._individuals_fitness[i] = function.evaluate(individuals[i])
            # update particle best fitness
            if self._individuals_fitness[i] < self.best_fitness:
                self.best_fitness = self._individuals_fitness[i]
                self.best_individual = self._individuals[i]

    cpdef np.ndarray _select_fathers(self) except *:
        # randomly select fathers
        fathers_0 = np.random.choice(self._n_individuals, self._n_individuals//2)
        fathers_1 = np.random.choice(self._n_individuals, self._n_individuals//2)
        return self._individuals[
            np.where(
                self._individuals_fitness[fathers_0] < self._individuals_fitness[fathers_1],
                fathers_0, fathers_1
            )
        ]

    cpdef np.ndarray _crossover(self, np.ndarray fathers_a, np.ndarray fathers_b) except *:
        beta = np.random.random((self._n_individuals//2, self._n_dims))
        return np.concatenate([
            beta * fathers_a + (1 - beta) * fathers_b,
            (1 - beta) * fathers_a + beta * fathers_b,
        ])

    cpdef np.ndarray _mutation(self, np.ndarray children) except *:
        mutation_mask = np.random.random(self._children_shape) <= self._mutation_probability
        mutation_values = np.random.normal(0, 1, self._children_shape)
        children = children + mutation_mask * mutation_values
        return children

    cpdef void optimize(self, int iterations, BaseFunction function) except *:
        self._fitness_compute(function)

        for i in range(iterations):
            # create new individuals
            children = self._crossover(self._select_fathers(), self._select_fathers())
            children = self._mutation(children)
            self._individuals = np.clip(children, self._min_bounds, self._max_bounds)
            self._fitness_compute(function)
            print(f'[{i+1}] current min value: {self.best_fitness}')

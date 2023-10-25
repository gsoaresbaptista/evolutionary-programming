import numpy as np
cimport numpy as np
from .base_optimizer cimport PopulationBasedOptimizer
from evolutionary_programming.objective_function.base_function cimport BaseFunction

np.import_array()


cdef class ParticleSwarm(PopulationBasedOptimizer):
    def __cinit__(
        self,
        int n_individuals,
        int n_dims,
        list min_bounds,
        list max_bounds,
        double cognitive = 0.5,
        double social = 0.5,
        double inertia = 0.8,
        int max_stagnation_interval = 5,
        double scaling_factor = 1.1,
    ):
        super().__init__(n_individuals, n_dims, min_bounds, max_bounds)
        self._cognitive = cognitive
        self._social = social
        self._inertia = inertia
        self._max_stagnation_interval = max_stagnation_interval
        self._scaling_factor = scaling_factor
        self._cj = np.random.random()
        self._init_individuals()

    cpdef void _init_individuals(self) except *:
        self._individuals = []
        cdef np.ndarray velocity = np.zeros(self._n_dims)
        for _ in range(self._n_individuals):
            individual = np.random.uniform(self._min_bounds, self._max_bounds, self._n_dims)
            self._individuals.append([individual, DBL_MAX, velocity, individual, DBL_MAX, 0])

    cpdef void optimize(self, int iterations, BaseFunction function) except *:
        # find the best individual
        self.best_individual = self._individuals[0][0]
        self.best_value = self._individuals[0][1]

        for i in range(self._n_individuals):
            self._individuals[i][1] = function.evaluate(self._individuals[i][0])

            # update particle best fitness
            if self._individuals[i][1] < self._individuals[i][4]:
                self._individuals[i][4] = self._individuals[i][1]
                self._individuals[i][3] = self._individuals[i][0]

                # update best value
                if self._individuals[i][1] < self.best_value:
                    self.best_value = self._individuals[i][1]
                    self.best_individual = self._individuals[i][0]

        for i in range(iterations):
            # update all particles
            g_best = self.best_individual

            for j in range(self._n_individuals):
                position, _, velocity, p_best, _, stagnation = self._individuals[j]

                if stagnation <= self._max_stagnation_interval:
                    # default pso update
                    # compute new velocity
                    self._individuals[j][2] = self._inertia * velocity +\
                        self._cognitive * np.random.random(self._n_dims) * (p_best - position) +\
                        self._social * np.random.random(self._n_dims) * (g_best - position)
                    
                    # update position
                    self._individuals[j][0] = np.clip(
                        self._individuals[j][0] + self._individuals[j][2],
                        self._min_bounds, self._max_bounds)
                else:
                    # chaotic jump update
                    self._individuals[j][5] = 0
                    self._cj = 4*self._cj*(1 - self._cj)
                    self._individuals[j][0] = np.clip(
                        self._individuals[j][3] * (1 + 1.1 * (2*self._cj - 1)),
                        self._min_bounds, self._max_bounds)

                # update particle best fitness
                self._individuals[j][1] = function.evaluate(self._individuals[j][0])

                if self._individuals[j][1] < self._individuals[j][4]:
                    self._individuals[j][5] = 0
                    self._individuals[j][4] = self._individuals[j][1]
                    self._individuals[j][3] = self._individuals[j][0]

                    # update best value
                    if self._individuals[j][1] < self.best_value:
                        self.best_value = self._individuals[j][1]
                        self.best_individual = self._individuals[j][0]
                else:
                    self._individuals[j][5] += 1

            print(f'[{i+1}] current min value: {self.best_value}')

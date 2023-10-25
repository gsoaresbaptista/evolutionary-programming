import numpy as np
cimport numpy as np
from evolutionary_programming.objective_function.base_function cimport BaseFunction

np.import_array()


cdef class ParticleSwarm:
    def __cinit__(
        self,
        int num_particles,
        int n_dims,
        list min_bounds,
        list max_bounds,
        float cognitive = 0.5,
        float social = 0.5,
        float inertia = 0.8,
        int max_stagnation_interval = 5,
        float scaling_factor = 1.1,
    ):
        self._n_particles = num_particles
        self._min_bounds = min_bounds
        self._max_bounds = max_bounds
        self._n_dims = n_dims
        self._particles = []
        self._cognitive = cognitive
        self._social = social
        self._inertia = inertia
        self._max_stagnation_interval = max_stagnation_interval
        self._scaling_factor = scaling_factor
        self._cj = np.random.random()
        self._init_particles()

    cpdef void _init_particles(self) except *:
        cdef float value = FLT_MAX
        cdef np.ndarray velocity = np.zeros(self._n_dims)
        for _ in range(self._n_particles):
            individual = np.random.uniform(self._min_bounds, self._max_bounds, self._n_dims)
            self._particles.append([individual, value, velocity, individual, value, 0])

    cpdef void optimize(self, int iterations, BaseFunction function) except *:
        # find the best individual
        cdef np.ndarray best_individual = self._particles[0][0]
        cdef float best_value = self._particles[0][1]

        for i in range(self._n_particles):
            self._particles[i][1] = function.evaluate(self._particles[i][0])

            # update particle best fitness
            if self._particles[i][1] < self._particles[i][4]:
                self._particles[i][4] = self._particles[i][1]
                self._particles[i][3] = self._particles[i][0]

                # update best value
                if self._particles[i][1] < best_value:
                    best_value = self._particles[i][1]
                    best_individual = self._particles[i][0]

        self.best_individual = best_individual
        self.best_value = best_value

        for i in range(iterations):
            # update all particles
            g_best = self.best_individual

            for j in range(self._n_particles):
                position, _, velocity, p_best, _, stagnation = self._particles[j]

                if stagnation <= self._max_stagnation_interval:
                    # default pso update
                    # compute new velocity
                    self._particles[j][2] = self._inertia * velocity +\
                        self._cognitive * np.random.random(self._n_dims) * (p_best - position) +\
                        self._social * np.random.random(self._n_dims) * (g_best - position)
                    
                    # update position
                    self._particles[j][0] = np.clip(
                        self._particles[j][0] + self._particles[j][2],
                        self._min_bounds, self._max_bounds)
                else:
                    # chaotic jump update
                    self._particles[j][5] = 0
                    self._cj = 4*self._cj*(1 - self._cj)
                    self._particles[j][0] = np.clip(
                        self._particles[j][3] * (1 + 1.1 * (2*self._cj - 1)),
                        self._min_bounds, self._max_bounds)

                # update particle best fitness
                self._particles[j][1] = function.evaluate(self._particles[j][0])

                if self._particles[j][1] < self._particles[j][4]:
                    self._particles[j][5] = 0
                    self._particles[j][4] = self._particles[j][1]
                    self._particles[j][3] = self._particles[j][0]

                    # update best value
                    if self._particles[j][1] < self.best_value:
                        self.best_value = self._particles[j][1]
                        self.best_individual = self._particles[j][0]
                else:
                    self._particles[j][5] += 1

            print(f'[{i+1}] current min value: {self.best_value}')

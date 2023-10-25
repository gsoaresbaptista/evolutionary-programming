import numpy as np
cimport numpy as np
from .base_optimizer cimport PopulationBasedOptimizer

from evolutionary_programming.objective_function.base_function cimport BaseFunction


np.import_array()


cdef extern from "float.h":
    const double DBL_MAX


cdef class GeneticAlgorithm(PopulationBasedOptimizer):
    # does not access via python code
    cdef double _mutation_probability
    cdef tuple _children_shape
    cdef np.ndarray _individuals
    cdef np.ndarray _individuals_fitness

    cpdef void _fitness_compute(self, BaseFunction function) except  *
    cpdef np.ndarray _select_fathers(self) except *
    cpdef np.ndarray _crossover(self, np.ndarray fathers_a, np.ndarray fathers_b) except *
    cpdef np.ndarray _mutation(self, np.ndarray children) except *
    cpdef void optimize(self, int iterations, BaseFunction function) except *
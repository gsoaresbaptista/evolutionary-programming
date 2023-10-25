import numpy as np
cimport numpy as np
from .base_optimizer cimport PopulationBasedOptimizer

from evolutionary_programming.objective_function.base_function cimport BaseFunction


cdef extern from "float.h":
    const float FLT_MAX


cdef class ParticleSwarm(PopulationBasedOptimizer):
    # does not access via python code
    cdef int _max_stagnation_interval
    cdef float _scaling_factor
    cdef float _cj
    cdef float _cognitive
    cdef float _social
    cdef float _inertia

    cpdef void optimize(self, int iterations, BaseFunction function) except *

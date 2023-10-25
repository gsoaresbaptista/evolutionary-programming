import numpy as np
cimport numpy as np
from evolutionary_programming.objective_function.base_function cimport BaseFunction


np.import_array()


cdef extern from "float.h":
    const double DBL_MAX


cdef class ParticleSwarm:
    cdef readonly np.ndarray best_individual
    cdef readonly double best_value
    # does not access via python code
    cdef list _min_bounds
    cdef list _max_bounds
    cdef list _particles
    cdef int _n_particles
    cdef int _n_dims
    cdef double _cognitive
    cdef double _social
    cdef double _inertia

    cpdef void _init_particles(self) except *
    cpdef void optimize(self, int iterations, BaseFunction function) except *

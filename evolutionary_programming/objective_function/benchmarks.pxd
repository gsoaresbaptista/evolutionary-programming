import numpy as np
cimport numpy as np
from .base_function cimport BaseFunction

np.import_array()


cdef class RastriginFunction(BaseFunction):
    cdef int _dimension
    cpdef float evaluate(self, np.ndarray individual) except *

import numpy as np
cimport numpy as np
from .base_function cimport BaseFunction

np.import_array()


cdef class RastriginFunction(BaseFunction):
    def __cinit__(self, int dimension):
        self._dimension = dimension

    cpdef double evaluate(self, np.ndarray individual) except *:
        return 10 * self._dimension + np.sum(
            individual**2 - 10 * np.cos(np.deg2rad(2*np.pi*individual))
        )

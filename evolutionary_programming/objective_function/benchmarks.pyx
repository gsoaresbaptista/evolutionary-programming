from libc.math cimport cos, pi
from .base_function cimport BaseFunction


cdef class RastriginFunction(BaseFunction):
    def __cinit__(self, int dimension):
        self._dimension = dimension

    cpdef double evaluate(self, double[:] individual) noexcept:
        cdef double value = 10 * self._dimension

        for i in range(self._dimension):
            value += individual[i]**2 - 10*cos(2*pi*individual[i])

        return value

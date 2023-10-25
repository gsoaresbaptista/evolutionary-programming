cimport cython
from cython.cimports.libc.math import cos, pi
from .base_function cimport BaseFunction


@cython.final
cdef class RastriginFunction(BaseFunction):
    def __cinit__(self, int dimension):
        self._dimension = dimension

    cpdef double evaluate(self, double[:] individual) noexcept nogil:
        cdef double value = 10 * self._dimension

        for i in range(self._dimension):
            value += individual[i]**2 - 10*cos(2*pi*individual[i])

        return value

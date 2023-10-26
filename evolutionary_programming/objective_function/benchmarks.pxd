from .base_function cimport BaseFunction


cdef class RastriginFunction(BaseFunction):
    cdef int _dimension
    cpdef double evaluate(self, double[:] individual) noexcept

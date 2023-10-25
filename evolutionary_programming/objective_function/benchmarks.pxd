from .base_function cimport BaseFunction


cdef class RastriginFunction(BaseFunction):
    cdef int _dimension
    cpdef float evaluate(self, double[:] individual) noexcept nogil

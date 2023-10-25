cimport cython

@cython.final
cdef class BaseFunction:
    cpdef float evaluate(self, double[:] individual) noexcept nogil:
        ...

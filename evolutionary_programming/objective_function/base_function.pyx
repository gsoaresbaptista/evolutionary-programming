cimport cython

@cython.final
cdef class BaseFunction:
    cpdef double evaluate(self, double[:] individual) noexcept nogil:
        ...

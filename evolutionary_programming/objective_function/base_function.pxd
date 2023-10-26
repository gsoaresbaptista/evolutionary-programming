cdef class BaseFunction:
    @classmethod
    cpdef double evaluate(self, double[:] individual) noexcept

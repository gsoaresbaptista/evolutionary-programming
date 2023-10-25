cdef class BaseFunction:
    @classmethod
    cpdef float evaluate(self, double[:] individual) noexcept nogil

import numpy as np
cimport numpy as np


np.import_array()


cdef class BaseFunction:
    cpdef float evaluate(self, np.ndarray individual) except *:
        ...

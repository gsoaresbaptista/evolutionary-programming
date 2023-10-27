import numpy as np
cimport numpy as np
from .base_function cimport BaseFunction
from evolutionary_programming.neural_network.coding cimport decode_neural_network


cdef class MeanSquaredErrorForNN(BaseFunction):
    cdef np.ndarray x_data
    cdef np.ndarray y_data
    cdef list[tuple] decode_guide
    cdef double l2_regularization

    cpdef double evaluate(self, np.ndarray individual) noexcept

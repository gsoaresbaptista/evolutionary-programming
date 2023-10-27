import numpy as np
cimport numpy as np
from .base_function cimport BaseFunction
from evolutionary_programming.neural_network.coding cimport decode_neural_network


cdef class RootMeanSquaredErrorForNN(BaseFunction):
    def __init__(
        self,
        np.ndarray x_data,
        np.ndarray y_data,
        list[tuple] decode_guide,
        double l2_regularization = 0.0,
    ):
        super().__init__()
        self._x_data = x_data
        self._y_data = y_data
        self._decode_guide = decode_guide
        self._l2_regularization = l2_regularization

    cpdef double evaluate(self, np.ndarray individual) noexcept:
        # decode network
        nn = decode_neural_network(individual, self._decode_guide)

        with np.errstate(all='raise'):
            try:
                y_hat = nn.predict(self._x_data)
            except FloatingPointError:
                return float('inf')

        # compute error
        error = np.sqrt(np.mean((y_hat - self._y_data) ** 2))
        weights = [layer._weights.squeeze()**2 for layer in nn._layers]
        error += self._l2_regularization * np.sum([w.sum() for w in weights])

        return error

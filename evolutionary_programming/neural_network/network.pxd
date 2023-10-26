import numpy as np
cimport numpy as np


np.import_array()


cdef extern from "limits.h":
    const int INT_MAX


cdef extern from "float.h":
    const double DBL_MAX


cdef class DenseLayer:
    """
    A fully connected dense layer for neural networks.

    This layer connects the input to the output with a set of learnable parameters
    including weights and biases. It supports various activation functions,
    weight initialization methods, and regularization techniques.

    Parameters:
    ----------
    input_size : int
        The number of input features.

    output_size : int
        The number of output units.

    activation : str, optional (default: 'tanh')
        The activation function to apply to the layer's output.

    weights_init : str, optional (default: 'glorot_uniform')
        The weight initialization method for the layer.

    biases_init : str, optional (default: 'glorot_uniform')
        The bias initialization method for the layer.

    regularization : str, optional (default: 'l2')
        The regularization technique to apply to the weights.

    regularization_strength : float, optional (default: 0.0)
        The strength of regularization applied to the weights.

    dropout_probability : float, optional (default: 0.0)
        The dropout probability to apply during training.

    batch_norm : bool, optional (default: False)
        Whether to use batch normalization.

    batch_decay : float, optional (default: 0.9)
        The decay rate for updating batch normalization statistics.

    Attributes:
    ----------
    _weights : ndarray
        The weights of the layer.

    _biases : ndarray
        The biases of the layer.

    _gamma : ndarray
        The gamma parameter for batch normalization.

    _beta : ndarray
        The beta parameter for batch normalization.

    _activation : function
        The activation function applied to the layer's output.

    _regularization : function
        The regularization function applied to the weights.

    _regularization_strength : float
        The strength of regularization applied to the weights.

    _dropout_probability : float
        The dropout probability applied during training.

    _batch_norm : bool
        Whether batch normalization is enabled.

    _batch_decay : float
        The decay rate for updating batch normalization statistics.

    _input : ndarray
        The input data to the layer.

    _dropout_mask : ndarray
        The dropout mask applied during training.

    _activation_input : ndarray
        The input to the activation function.

    _activation_output : ndarray
        The output of the activation function.

    _dweights : ndarray
        The gradients with respect to weights.

    _dbias : ndarray
        The gradients with respect to biases.

    _dgamma : ndarray
        The gradients with respect to gamma (batch normalization).

    _dbeta : ndarray
        The gradients with respect to beta (batch normalization).

    _prev_dweights : ndarray
        The previous gradients with respect to weights (for momentum-based optimizers).

    _population_mean : ndarray
        The population mean for batch normalization.

    _population_var : ndarray
        The population variance for batch normalization.

    _batch_norm_cache : list
        Cache for batch normalization values during forward and backward passes.
    """
    cdef public np.ndarray _weights
    cdef public np.ndarray _biases
    cdef public np.ndarray _gamma
    cdef public np.ndarray _beta

    # hyper params
    cdef public object _activation
    cdef public object _regularization
    cdef public double _regularization_strength
    cdef public double _dropout_probability
    cdef public double _batch_decay
    cdef public bint _batch_norm

    # intermediary values
    cdef public np.ndarray _input
    cdef public np.ndarray _dropout_mask
    cdef public np.ndarray _activation_input
    cdef public np.ndarray _activation_output
    cdef public np.ndarray _dweights
    cdef public np.ndarray _dbias
    cdef public np.ndarray _dgamma
    cdef public np.ndarray _dbeta
    cdef public np.ndarray _prev_dweights
    cdef public np.ndarray _population_mean
    cdef public np.ndarray _population_var
    cdef public list _batch_norm_cache


cdef class NeuralNetwork:
    cdef public list _layers
    cdef public list _best_model
    cdef double _learning_rate
    cdef object _lr_decay_fn
    cdef double _lr_decay_rate
    cdef int _lr_decay_steps
    cdef object _loss_function
    cdef double _momentum
    cdef int _patience
    cdef int _waiting
    cdef double _best_loss

    cdef np.ndarray _feedforward(self, np.ndarray x, bint training=*) except *
    cdef np.ndarray _backpropagation(self, np.ndarray y, np.ndarray y_hat) except *
    cpdef void add_layer(self, layer) except *
    cpdef double evaluate(self, np.ndarray x, np.ndarray y, str loss_name=*) except *
    cpdef void save(self, str file_path) except *
    cpdef NeuralNetwork load(self, str file_path) except *
    cpdef np.ndarray predict(self, np.ndarray x) except *
    cpdef void fit(self, np.ndarray x_train, np.ndarray y_train, np.ndarray x_val=*,
        np.ndarray y_val=*, int epochs=*, object batch_generator=*,
        int batch_size=*, int verbose=*) except *


cpdef np.ndarray batch_normalization_forward(DenseLayer layer, np.ndarray x, bint training=*) except *


cpdef np.ndarray batch_normalization_backward(DenseLayer layer, np.ndarray dactivation) except *

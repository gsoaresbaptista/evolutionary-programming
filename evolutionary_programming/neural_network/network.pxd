import numpy as np
cimport numpy as np


np.import_array()


cdef extern from "limits.h":
    const int INT_MAX


cdef extern from "float.h":
    const double DBL_MAX


cdef class DenseLayer:
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
    cpdef void save(self, str file_path) except *
    cpdef NeuralNetwork load(self, str file_path) except *
    cpdef np.ndarray predict(self, np.ndarray x) except *
    cpdef void fit(self, np.ndarray x_train, np.ndarray y_train, np.ndarray x_val=*,
        np.ndarray y_val=*, int epochs=*, object batch_generator=*,
        int batch_size=*, int verbose=*) except *


cpdef np.ndarray batch_normalization_forward(DenseLayer layer, np.ndarray x, bint training=*) except *


cpdef np.ndarray batch_normalization_backward(DenseLayer layer, np.ndarray dactivation) except *

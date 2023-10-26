import numpy as np
cimport numpy as np
import pickle
from copy import deepcopy
from .utils import batch_sequential
from itertools import zip_longest
from .loss_functions cimport LOSS_FUNCTIONS
from .activation_functions cimport ACTIVATION_FUNCTIONS
from .regularization cimport REGULARIZATION_FUNCTIONS, LR_DECAY_FUNCTIONS
from .utils cimport WEIGHT_INITIALIZERS


np.import_array()


cdef class DenseLayer:
    """
    TESTE 2
    """
    def __init__(
        self,
        int input_size,
        int output_size,
        str activation = 'tanh',
        str weights_init = 'glorot_uniform',
        str biases_init = 'glorot_uniform',
        str regularization = 'l2',
        double regularization_strength = 0.0,
        double dropout_probability = 0.0,
        bint batch_norm = False,
        double batch_decay = 0.9,
    ):
        # parameters
        self._weights = WEIGHT_INITIALIZERS.get(weights_init)(output_size, input_size)
        self._biases = WEIGHT_INITIALIZERS.get(biases_init)(1, output_size)
        self._gamma = WEIGHT_INITIALIZERS.get('ones')(1, output_size)
        self._beta = WEIGHT_INITIALIZERS.get('zeros')(1, output_size)

        # hyper parameters
        self._activation = ACTIVATION_FUNCTIONS.get(activation)
        self._regularization = REGULARIZATION_FUNCTIONS.get(regularization)
        self._regularization_strength = regularization_strength
        self._dropout_probability = dropout_probability
        self._batch_norm = batch_norm
        self._batch_decay = batch_decay

        # intermediary values
        self._input = np.zeros(0)
        self._dropout_mask = np.zeros(0)
        self._activation_input, self._activation_output = np.zeros(0), np.zeros(0)
        self._dweights, self._dbias = np.zeros(0), np.zeros(0)
        self._dgamma, self._dbeta = np.zeros(0), np.zeros(0)
        self._prev_dweights = np.zeros((output_size, input_size))
        self._population_mean = np.zeros((1, output_size))
        self._population_var = np.zeros((1, output_size))
        self._batch_norm_cache = []

    def __deepcopy__(self, memo):
        cls = self.__class__
        layer = cls.__new__(cls)
        layer._weights = deepcopy(self._weights)
        layer._biases = deepcopy(self._biases)
        layer._gamma = deepcopy(self._gamma)
        layer._beta = deepcopy(self._beta)
        layer._activation = self._activation
        layer._regularization = self._regularization
        layer._regularization_strength = self._regularization_strength
        layer._dropout_probability = self._dropout_probability
        layer._batch_norm = self._batch_norm
        layer._batch_decay = self._batch_decay
        return layer


cdef class NeuralNetwork:
    def __cinit__(
        self,
        double learning_rate,
        str lr_decay_fn = 'none',
        double lr_decay_rate = 0.0,
        int lr_decay_steps = 1,
        str loss_function = 'mse',
        double momentum = 0.0,
        int patience = INT_MAX,
    ):
        self._layers = []
        self._learning_rate = learning_rate
        self._lr_decay_fn = LR_DECAY_FUNCTIONS.get(lr_decay_fn)
        self._lr_decay_rate = lr_decay_rate
        self._lr_decay_steps = lr_decay_steps
        self._loss_function = LOSS_FUNCTIONS.get(loss_function)
        self._momentum = momentum
        self._patience = patience
        self._waiting = 0
        self._best_loss = float('inf')

    cpdef void add_layer(self, layer) except *:
        if isinstance(layer, list):
            self._layers.extend(layer)
        elif isinstance(layer, DenseLayer):
            self._layers.append(layer)
        else:
            raise TypeError('Only support DenseLayer or list[DenseLayer] types')

    cpdef void save(self, str file_path) except *:
        pickle.dump(self, open(file_path, 'wb'), -1)

    cpdef NeuralNetwork load(self, str file_path) except *:
        return pickle.load(open(file_path, 'rb'))

    cpdef np.ndarray predict(self, np.ndarray x) except *:
        return self._feedforward(x, training=False)

    cdef np.ndarray _feedforward(self, np.ndarray x, bint training = True) except *:
        self._layers[0]._input = x
        layer_pairs = zip_longest(self._layers, self._layers[1:])

        # process each layer
        for cur_layer, next_layer in layer_pairs:
            y = cur_layer._input.dot(cur_layer._weights.T) + cur_layer._biases

            # apply batch normalization
            if cur_layer._batch_norm:
                y = batch_normalization_forward(cur_layer, y, training)

            # dropout mask
            cur_layer._dropout_mask = np.random.binomial(
                1, 1.0 - cur_layer._dropout_probability, y.shape
            ) / (1.0 - cur_layer._dropout_probability)

            # save values
            cur_layer._activation_input = y
            cur_layer._activation_output = cur_layer._activation(y, derivative=False) * (
                cur_layer._dropout_mask if training else 1.0
            )

            if next_layer:
                next_layer._input = cur_layer._activation_output

        return self._layers[len(self._layers) - 1]._activation_output

    cdef np.ndarray _backpropagation(self, np.ndarray y, np.ndarray y_hat) except *:
        last_delta = self._loss_function(y, y_hat, derivative=True)

        # calculate in reverse
        for layer in reversed(self._layers):
            dactivation = (
                layer._activation(layer._activation_input, derivative=True)
                * last_delta
                * layer._dropout_mask
            )

            # compute derivation for batch normalization
            if layer._batch_norm:
                dactivation = batch_normalization_backward(layer, dactivation)

            last_delta = dactivation.dot(layer._weights)
            layer._dweights = dactivation.T.dot(layer._input)
            layer._dbias = 1.0 * dactivation.sum(axis=0, keepdims=True)

        for layer in reversed(self._layers):
            # apply regularization
            layer._dweights = layer._dweights + (
                1.0 / y.shape[0]
            ) * layer._regularization_strength * layer._regularization(
                layer._weights, derivative=True
            )

            # apply momentum
            layer._prev_dweights = (
                -self._learning_rate * layer._dweights
                + self._momentum * layer._prev_dweights
            )

            # update weights and biases
            layer._weights = layer._weights + layer._prev_dweights
            layer._biases = layer._biases - self._learning_rate * layer._dbias

            # update batch normalization
            if layer._batch_norm:
                layer._beta = layer._beta - self._learning_rate * layer._dbeta
                layer._gamma = (
                    layer._gamma - self._learning_rate * layer._dgamma
                )

    cpdef void fit(
        self,
        np.ndarray x_train,
        np.ndarray y_train,
        np.ndarray x_val = None,
        np.ndarray y_val = None,
        int epochs = 100,
        object batch_generator = batch_sequential,
        int batch_size = -1,
        int verbose = 10,
    ) except *:
        # saves initial learning rate for restoration after fit
        learning_rate = self._learning_rate

        # initial settings for patience to work
        self._best_model = deepcopy(self._layers)
        self._best_loss = float('inf')

        # update x_val and y_val
        if x_val is None or y_val is None:
            print(
                '* The X_val or Y_val set was not informed, '
                'the training sets will be used'
            )
            x_val, y_val = x_train, y_train

        for epoch in range(epochs):
            self._learning_rate = self._lr_decay_fn(
                learning_rate,
                epoch,
                self._lr_decay_rate,
                self._lr_decay_steps,
            )
            batches = batch_generator(x_train, y_train, batch_size)

            for x_batch, y_batch in batches:
                y_pred = self._feedforward(x_batch)
                self._backpropagation(y_batch, y_pred)

            # check early stop
            loss_val = self._loss_function(y_val, self.predict(x_val), derivative=False)[0]

            if loss_val < self._best_loss:
                self._best_model = deepcopy(self._layers)
                self._best_loss, self._waiting = loss_val, 0
            else:
                self._waiting += 1
                if self._waiting >= self._patience:
                    print(f'* early stopping at epoch {epoch + 1}')
                    self._layers = self._best_model
                    break

            # print loss
            if (epoch + 1) % verbose == 0:
                # compute regularization loss
                loss_reg = (1.0 / y_train.shape[0]) * np.sum(
                    [
                        layer._regularization_strength
                        * layer._regularization(layer._weights, derivative=False)
                        for layer in self._layers
                    ]
                )

                # compute train loss
                loss_train = self._loss_function(
                    y_train, self.predict(x_train), derivative=False
                )[0]

                # get information to format output
                d_length = len(str(epochs))

                print(
                    f'epoch: {epoch + 1:{d_length}d}/{epochs:{d_length}d} | '
                    f'loss train: {loss_train:.4f} | '
                    f'loss reg.: {loss_reg:.4f} | '
                    f'sum: {loss_train + loss_reg:.4f} '
                )

        # restore initial settings
        self._learning_rate = learning_rate

    cpdef double evaluate(self, np.ndarray x, np.ndarray y, str loss_name = None) except *:
        loss_fn = self._loss_function if loss_name is None else LOSS_FUNCTIONS.get(loss_name)
        y_hat = self.predict(x)
        return loss_fn(y, y_hat, derivative=False)[0]


cpdef np.ndarray batch_normalization_forward(DenseLayer layer, np.ndarray x, bint training = True) except *:
    mu = np.mean(x, axis=0) if training else layer._population_mean
    var = np.var(x, axis=0) if training else layer._population_var
    x_norm = (x - mu) / np.sqrt(var + 1e-8)
    out = layer._gamma * x_norm + layer._beta

    if training:
        # mean average
        layer._population_mean = (
            layer._batch_decay * layer._population_mean
            + (1 - layer._batch_decay) * mu
        )
        # mean var
        layer._population_var = (
            layer._batch_decay * layer._population_var
            + (1 - layer._batch_decay) * var
        )
        # update batch norm cache
        layer._batch_norm_cache = [x, x_norm, mu, var]

    return out


cpdef np.ndarray batch_normalization_backward(DenseLayer layer, np.ndarray dactivation) except *:
    # extract cached values from the layer, and batch size from input
    x, x_norm, mu, var = layer._batch_norm_cache
    m = layer._activation_input.shape[0]

    # compute gradients
    x_mu = x - mu
    std_inv = 1.0 / np.sqrt(var + 1e-8)
    dx_norm = dactivation * layer._gamma
    dvar = np.sum(dx_norm * x_mu, axis=0) * -0.5 * (std_inv**3)
    dmu = np.sum(dx_norm * -std_inv, axis=0) + dvar * np.sum(-2 * x_mu, axis=0)
    dx = (dx_norm * std_inv) + (dvar * 2 * x_mu / m) + (dmu / m)
    layer._dgamma = np.sum(dactivation * x_norm, axis=0)
    layer._dbeta = np.sum(dactivation, axis=0)

    return dx

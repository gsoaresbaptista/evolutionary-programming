import numpy as np
import os
from tempfile import TemporaryDirectory
from evolutionary_programming.neural_network import (
    NeuralNetwork,
    DenseLayer,
    NoLayersError,
)


def test_neural_network_initialization():
    nn = NeuralNetwork(1e-3, momentum=0.9)
    nn.add_layer([DenseLayer(1, 10), DenseLayer(10, 10), DenseLayer(10, 3)])
    assert len(nn._layers) == 3


def test_neural_network_feedforward_shape_1d():
    nn = NeuralNetwork(1e-3, momentum=0.9)
    nn.add_layer([DenseLayer(1, 10), DenseLayer(10, 10), DenseLayer(10, 3)])
    y_hat = nn.predict(np.array([1]))
    assert y_hat.shape == (1, 3)


def test_neural_network_feedforward_shape_3d():
    nn = NeuralNetwork(1e-3, momentum=0.9)
    nn.add_layer([DenseLayer(1, 10), DenseLayer(10, 10), DenseLayer(10, 3)])
    y_hat = nn.predict(np.array([[1], [3], [3]]))
    assert y_hat.shape == (3, 3)


def test_neural_network_minimizing_loss():
    x = np.array([[0.1, 0.2, 0.7]])
    y = np.array([[1, 0, 0]])
    nn = NeuralNetwork(
        1e-3, momentum=0.9, loss_function='softmax_neg_log_likelihood')
    nn.add_layer([DenseLayer(3, 3), DenseLayer(3, 3), DenseLayer(3, 3)])

    loss_before = nn.evaluate(x, y)
    nn.fit(x, y, epochs=100, verbose=101)
    loss_after = nn.evaluate(x, y)

    assert loss_after < loss_before


def test_neural_network_empty_layers():
    x = np.array([[0.1, 0.2, 0.7]])
    y = np.array([[1, 0, 0]])
    nn = NeuralNetwork(
        1e-3, momentum=0.9, loss_function='softmax_neg_log_likelihood')

    # check if neural network is throw the error
    caught = False
    try:
        nn.fit(x, y, epochs=100, verbose=101)
    except NoLayersError:
        caught = True

    assert caught


def test_neural_network_save():
    x = np.array([[0.1, 0.2, 0.7]])
    y = np.array([[1, 0, 0]])
    nn = NeuralNetwork(
        1e-3, momentum=0.9, loss_function='softmax_neg_log_likelihood')
    nn.add_layer([DenseLayer(3, 3), DenseLayer(3, 3), DenseLayer(3, 3)])
    nn.fit(x, y, epochs=100, verbose=101)

    # evaluate and save network
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir) + 'test.pkl'
        nn.save(path)
        assert os.path.exists(path)


def test_neural_network_load():
    x = np.array([[0.1, 0.2, 0.7]])
    y = np.array([[1, 0, 0]])
    nn = NeuralNetwork(
        1e-3, momentum=0.9, loss_function='softmax_neg_log_likelihood')
    nn.add_layer([DenseLayer(3, 3), DenseLayer(3, 3), DenseLayer(3, 3)])
    nn.fit(x, y, epochs=100, verbose=101)

    # evaluate and save network
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir) + 'test.pkl'
        nn.save(path)
        nn2 = NeuralNetwork.load(path)
        assert isinstance(nn2, NeuralNetwork)


def test_neural_network_load_equal_weights():
    x = np.array([[0.1, 0.2, 0.7]])
    y = np.array([[1, 0, 0]])
    nn = NeuralNetwork(
        1e-3, momentum=0.9, loss_function='softmax_neg_log_likelihood')
    nn.add_layer([DenseLayer(3, 3), DenseLayer(3, 3), DenseLayer(3, 3)])
    nn.fit(x, y, epochs=100, verbose=101)

    # evaluate and save network
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir) + 'test.pkl'
        nn.save(path)
        nn2 = NeuralNetwork.load(path)
        assert nn.evaluate(x, y) == nn2.evaluate(x, y)

import os
import pytest
import numpy as np
from tempfile import TemporaryDirectory
from evolutionary_programming.neural_network import (
    NeuralNetwork,
    DenseLayer,
    NoLayersError,
)


@pytest.fixture
def module() -> NeuralNetwork:
    module = NeuralNetwork(1e-3, momentum=0.9)
    module.add_layer([
        DenseLayer(3, 10), DenseLayer(10, 10), DenseLayer(10, 3)])
    return module


@pytest.fixture
def x() -> np.ndarray:
    return np.array([[0.1, 0.2, 0.7]])


@pytest.fixture
def y() -> np.ndarray:
    return np.array([[1, 0, 0]])


def test_neural_network_initialization():
    nn = NeuralNetwork(1e-3, momentum=0.9)
    nn.add_layer([DenseLayer(1, 10), DenseLayer(10, 10), DenseLayer(10, 3)])
    assert len(nn._layers) == 3


def test_neural_network_feedforward_shape_1d():
    module = NeuralNetwork(1e-3)
    module.add_layer([DenseLayer(1, 5), DenseLayer(5, 5), DenseLayer(5, 3)])
    y_hat = module.predict(np.array([1]))
    assert y_hat.shape == (1, 3)


def test_neural_network_feedforward_shape_3d(module: NeuralNetwork):
    module = NeuralNetwork(1e-3)
    module.add_layer([DenseLayer(1, 5), DenseLayer(5, 5), DenseLayer(5, 3)])
    y_hat = module.predict(np.array([[1], [3], [3]]))
    assert y_hat.shape == (3, 3)


def test_neural_network_minimizing_loss(
    module: NeuralNetwork, x: np.ndarray, y: np.ndarray
):
    loss_before = module.evaluate(x, y)
    assert True
    module.fit(x, y, epochs=100, verbose=101)
    loss_after = module.evaluate(x, y)

    assert loss_after < loss_before


def test_neural_network_empty_layers(
    module: NeuralNetwork, x: np.ndarray, y: np.ndarray
):
    module = NeuralNetwork(1e-3)
    caught = False

    try:
        module.fit(x, y, epochs=100, verbose=101)
    except NoLayersError:
        caught = True

    assert caught


def test_neural_network_save(
    module: NeuralNetwork, x: np.ndarray, y: np.ndarray
):
    module.fit(x, y, epochs=100, verbose=101)

    # evaluate and save network
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir) + 'temp_module.pkl'
        module.save(path)
        assert os.path.exists(path)


def test_neural_network_load(
    module: NeuralNetwork, x: np.ndarray, y: np.ndarray
):
    module.fit(x, y, epochs=100, verbose=101)

    # evaluate and save network
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir) + 'temp_module.pkl'
        module.save(path)
        module2 = NeuralNetwork.load(path)
        assert isinstance(module2, NeuralNetwork)


def test_neural_network_load_equal_weights(
    module: NeuralNetwork, x: np.ndarray, y: np.ndarray
):
    module.fit(x, y, epochs=100, verbose=101)

    # evaluate and save network
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir) + 'temp_module.pkl'
        module.save(path)
        module2 = NeuralNetwork.load(path)
        assert module.evaluate(x, y) == module2.evaluate(x, y)

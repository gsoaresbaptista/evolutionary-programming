import numpy as np
from evolutionary_programming.neural_network import (
    NeuralNetwork,
    DenseLayer,
)


def test_neural_network_initialization():
    nn = NeuralNetwork(1e-3, momentum=0.9)
    nn.add_layer([DenseLayer(1, 10), DenseLayer(10, 10), DenseLayer(10, 3)])
    assert len(nn._layers) == 3


def test_neural_network_feedforward_shape():
    nn = NeuralNetwork(1e-3, momentum=0.9)
    nn.add_layer([DenseLayer(1, 10), DenseLayer(10, 10), DenseLayer(10, 3)])
    y_hat = nn.predict(np.array([1]))

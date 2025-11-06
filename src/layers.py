"""
Layer definitions, activation functions, and weight initialization.
"""
import numpy as np
from typing import Tuple


class Activation:
    """Base class for activation functions."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        raise NotImplementedError

    def backward(self, dA: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute gradient with respect to pre-activation."""
        raise NotImplementedError


class Sigmoid(Activation):
    """Sigmoid activation function."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid activation.

        Args:
            z: Pre-activation values

        Returns:
            Activated values
        """
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def backward(self, dA: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Compute gradient of sigmoid.

        Args:
            dA: Gradient from next layer
            z: Pre-activation values

        Returns:
            Gradient with respect to z
        """
        a = self.forward(z)
        return dA * a * (1 - a)


class ReLU(Activation):
    """ReLU activation function."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Compute ReLU activation.

        Args:
            z: Pre-activation values

        Returns:
            Activated values
        """
        return np.maximum(0, z)

    def backward(self, dA: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Compute gradient of ReLU.

        Args:
            dA: Gradient from next layer
            z: Pre-activation values

        Returns:
            Gradient with respect to z
        """
        return dA * (z > 0)


class Tanh(Activation):
    """Tanh activation function."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Compute tanh activation.

        Args:
            z: Pre-activation values

        Returns:
            Activated values
        """
        return np.tanh(z)

    def backward(self, dA: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Compute gradient of tanh.

        Args:
            dA: Gradient from next layer
            z: Pre-activation values

        Returns:
            Gradient with respect to z
        """
        a = self.forward(z)
        return dA * (1 - a ** 2)


class Softmax(Activation):
    """Softmax activation function."""

    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Compute softmax activation.

        Args:
            z: Pre-activation values (batch_size, num_classes)

        Returns:
            Probabilities for each class
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def backward(self, dA: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Compute gradient of softmax (used with cross-entropy).

        Args:
            dA: Gradient from loss
            z: Pre-activation values

        Returns:
            Gradient with respect to z
        """
        return dA


class DenseLayer:
    """Fully connected (dense) layer."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = 'sigmoid',
        initialization: str = 'xavier'
    ):
        """
        Initialize dense layer.

        Args:
            input_size: Number of input features
            output_size: Number of neurons in this layer
            activation: Activation function name
            initialization: Weight initialization method
        """
        self.input_size = input_size
        self.output_size = output_size

        self.weights = self._initialize_weights(initialization)
        self.bias = np.zeros((1, output_size))

        self.activation = self._get_activation(activation)

        self.z = None
        self.a = None
        self.input = None

        self.dW = None
        self.db = None

    def _initialize_weights(self, method: str) -> np.ndarray:
        """
        Initialize weights using specified method.

        Args:
            method: Initialization method ('xavier', 'he', 'random')

        Returns:
            Initialized weight matrix
        """
        if method == 'xavier':
            limit = np.sqrt(6 / (self.input_size + self.output_size))
            return np.random.uniform(-limit, limit, (self.input_size, self.output_size))
        elif method == 'he':
            std = np.sqrt(2 / self.input_size)
            return np.random.randn(self.input_size, self.output_size) * std
        elif method == 'random':
            return np.random.randn(self.input_size, self.output_size) * 0.01
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def _get_activation(self, name: str) -> Activation:
        """Get activation function by name."""
        activations = {
            'sigmoid': Sigmoid(),
            'relu': ReLU(),
            'tanh': Tanh(),
            'softmax': Softmax(),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.

        Args:
            x: Input data (batch_size, input_size)

        Returns:
            Output after activation (batch_size, output_size)
        """
        self.input = x
        self.z = x @ self.weights + self.bias
        self.a = self.activation.forward(self.z)
        return self.a

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Backward pass through the layer.

        Args:
            dA: Gradient from next layer

        Returns:
            Gradient to pass to previous layer
        """
        m = self.input.shape[0]

        dZ = self.activation.backward(dA, self.z)

        self.dW = (self.input.T @ dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m

        dA_prev = dZ @ self.weights.T

        return dA_prev

    def get_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return weights and biases."""
        return self.weights, self.bias

    def get_gradients(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return gradients for weights and biases."""
        return self.dW, self.db

    def set_params(self, weights: np.ndarray, bias: np.ndarray) -> None:
        """Set weights and biases."""
        self.weights = weights
        self.bias = bias


class Dropout:
    """Dropout layer for regularization."""

    def __init__(self, dropout_rate: float = 0.5):
        """
        Initialize dropout layer.

        Args:
            dropout_rate: Probability of dropping a unit (0 to 1)
        """
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply dropout during training.

        Args:
            x: Input data

        Returns:
            Output with dropout applied
        """
        if self.training and self.dropout_rate > 0:
            self.mask = (np.random.rand(*x.shape) > self.dropout_rate).astype(np.float32)
            return x * self.mask / (1 - self.dropout_rate)
        return x

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Backward pass through dropout.

        Args:
            dA: Gradient from next layer

        Returns:
            Gradient for previous layer
        """
        if self.training and self.dropout_rate > 0:
            return dA * self.mask / (1 - self.dropout_rate)
        return dA

    def set_training(self, training: bool) -> None:
        """Set training mode."""
        self.training = training


def get_activation_function(name: str) -> Activation:
    """
    Factory function to get activation by name.

    Args:
        name: Activation function name

    Returns:
        Activation instance
    """
    activations = {
        'sigmoid': Sigmoid(),
        'relu': ReLU(),
        'tanh': Tanh(),
        'softmax': Softmax(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]

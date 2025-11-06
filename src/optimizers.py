"""
Optimization algorithms for neural network training.
"""
import numpy as np
from typing import List, Tuple


class Optimizer:
    """Base class for optimizers."""

    def update(
        self,
        params: List[Tuple[np.ndarray, np.ndarray]],
        grads: List[Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """
        Update parameters using gradients.

        Args:
            params: List of (weights, bias) tuples for each layer
            grads: List of (dW, db) tuples for each layer
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset optimizer state."""
        pass


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with optional momentum."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0
    ):
        """
        Initialize SGD optimizer.

        Args:
            learning_rate: Learning rate
            momentum: Momentum factor (0 for no momentum)
            weight_decay: L2 regularization factor
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.velocity_w = []
        self.velocity_b = []

    def update(
        self,
        params: List[Tuple[np.ndarray, np.ndarray]],
        grads: List[Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """
        Update parameters using SGD with momentum.

        Args:
            params: List of (weights, bias) tuples
            grads: List of (dW, db) tuples
        """
        if len(self.velocity_w) == 0:
            self.velocity_w = [np.zeros_like(w) for w, _ in params]
            self.velocity_b = [np.zeros_like(b) for _, b in params]

        for i, ((w, b), (dW, db)) in enumerate(zip(params, grads)):
            if self.weight_decay > 0:
                dW = dW + self.weight_decay * w

            self.velocity_w[i] = self.momentum * self.velocity_w[i] - self.learning_rate * dW
            self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.learning_rate * db

            w += self.velocity_w[i]
            b += self.velocity_b[i]

    def reset(self) -> None:
        """Reset optimizer state."""
        self.velocity_w = []
        self.velocity_b = []


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0
    ):
        """
        Initialize Adam optimizer.

        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
            weight_decay: L2 regularization factor
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        self.m_w = []
        self.m_b = []
        self.v_w = []
        self.v_b = []
        self.t = 0

    def update(
        self,
        params: List[Tuple[np.ndarray, np.ndarray]],
        grads: List[Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """
        Update parameters using Adam algorithm.

        Args:
            params: List of (weights, bias) tuples
            grads: List of (dW, db) tuples
        """
        if len(self.m_w) == 0:
            self.m_w = [np.zeros_like(w) for w, _ in params]
            self.m_b = [np.zeros_like(b) for _, b in params]
            self.v_w = [np.zeros_like(w) for w, _ in params]
            self.v_b = [np.zeros_like(b) for _, b in params]

        self.t += 1

        for i, ((w, b), (dW, db)) in enumerate(zip(params, grads)):
            if self.weight_decay > 0:
                dW = dW + self.weight_decay * w

            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dW
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db

            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (dW ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db ** 2)

            m_w_corrected = self.m_w[i] / (1 - self.beta1 ** self.t)
            m_b_corrected = self.m_b[i] / (1 - self.beta1 ** self.t)

            v_w_corrected = self.v_w[i] / (1 - self.beta2 ** self.t)
            v_b_corrected = self.v_b[i] / (1 - self.beta2 ** self.t)

            w -= self.learning_rate * m_w_corrected / (np.sqrt(v_w_corrected) + self.epsilon)
            b -= self.learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)

    def reset(self) -> None:
        """Reset optimizer state."""
        self.m_w = []
        self.m_b = []
        self.v_w = []
        self.v_b = []
        self.t = 0


class RMSprop(Optimizer):
    """RMSprop optimizer."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        rho: float = 0.9,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0
    ):
        """
        Initialize RMSprop optimizer.

        Args:
            learning_rate: Learning rate
            rho: Decay rate for moving average
            epsilon: Small constant for numerical stability
            weight_decay: L2 regularization factor
        """
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        self.cache_w = []
        self.cache_b = []

    def update(
        self,
        params: List[Tuple[np.ndarray, np.ndarray]],
        grads: List[Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """
        Update parameters using RMSprop algorithm.

        Args:
            params: List of (weights, bias) tuples
            grads: List of (dW, db) tuples
        """
        if len(self.cache_w) == 0:
            self.cache_w = [np.zeros_like(w) for w, _ in params]
            self.cache_b = [np.zeros_like(b) for _, b in params]

        for i, ((w, b), (dW, db)) in enumerate(zip(params, grads)):
            if self.weight_decay > 0:
                dW = dW + self.weight_decay * w

            self.cache_w[i] = self.rho * self.cache_w[i] + (1 - self.rho) * (dW ** 2)
            self.cache_b[i] = self.rho * self.cache_b[i] + (1 - self.rho) * (db ** 2)

            w -= self.learning_rate * dW / (np.sqrt(self.cache_w[i]) + self.epsilon)
            b -= self.learning_rate * db / (np.sqrt(self.cache_b[i]) + self.epsilon)

    def reset(self) -> None:
        """Reset optimizer state."""
        self.cache_w = []
        self.cache_b = []


def get_optimizer(name: str, **kwargs) -> Optimizer:
    """
    Factory function to get optimizer by name.

    Args:
        name: Optimizer name ('sgd', 'adam', 'rmsprop')
        **kwargs: Optimizer-specific parameters

    Returns:
        Optimizer instance
    """
    optimizers = {
        'sgd': SGD,
        'adam': Adam,
        'rmsprop': RMSprop,
    }

    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")

    return optimizers[name](**kwargs)

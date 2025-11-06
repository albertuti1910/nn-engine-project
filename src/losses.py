"""
Loss functions for neural network training.
"""
import numpy as np


class Loss:
    """Base class for loss functions."""

    def compute(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute loss value."""
        raise NotImplementedError

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute gradient of loss with respect to predictions."""
        raise NotImplementedError


class MeanSquaredError(Loss):
    """Mean Squared Error loss for regression."""

    def compute(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute MSE loss.

        Args:
            y_pred: Predicted values (batch_size, output_dim)
            y_true: True values (batch_size, output_dim)

        Returns:
            Scalar loss value
        """
        return np.mean((y_pred - y_true) ** 2)

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute gradient of MSE.

        Args:
            y_pred: Predicted values
            y_true: True values

        Returns:
            Gradient with respect to predictions
        """
        return 2 * (y_pred - y_true) / y_true.shape[0]


class CategoricalCrossEntropy(Loss):
    """Categorical Cross-Entropy loss for multi-class classification."""

    def __init__(self, epsilon: float = 1e-15):
        """
        Initialize cross-entropy loss.

        Args:
            epsilon: Small value to avoid log(0)
        """
        self.epsilon = epsilon

    def compute(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute categorical cross-entropy loss.

        Args:
            y_pred: Predicted probabilities (batch_size, num_classes)
            y_true: True labels one-hot encoded (batch_size, num_classes)

        Returns:
            Scalar loss value
        """
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

        return loss

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute gradient of cross-entropy.

        When used with softmax activation, this simplifies to (y_pred - y_true).

        Args:
            y_pred: Predicted probabilities
            y_true: True labels one-hot encoded

        Returns:
            Gradient with respect to predictions
        """
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return (y_pred - y_true)


class BinaryCrossEntropy(Loss):
    """Binary Cross-Entropy loss for binary classification."""

    def __init__(self, epsilon: float = 1e-15):
        """
        Initialize binary cross-entropy loss.

        Args:
            epsilon: Small value to avoid log(0)
        """
        self.epsilon = epsilon

    def compute(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss.

        Args:
            y_pred: Predicted probabilities (batch_size, 1)
            y_true: True labels (batch_size, 1)

        Returns:
            Scalar loss value
        """
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )

        return loss

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute gradient of binary cross-entropy.

        Args:
            y_pred: Predicted probabilities
            y_true: True labels

        Returns:
            Gradient with respect to predictions
        """
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return (y_pred - y_true)


def get_loss_function(name: str) -> Loss:
    """
    Factory function to get loss by name.

    Args:
        name: Loss function name ('mse', 'categorical_crossentropy', 'binary_crossentropy')

    Returns:
        Loss instance
    """
    losses = {
        'mse': MeanSquaredError(),
        'mean_squared_error': MeanSquaredError(),
        'categorical_crossentropy': CategoricalCrossEntropy(),
        'cross_entropy': CategoricalCrossEntropy(),
        'binary_crossentropy': BinaryCrossEntropy(),
    }

    if name not in losses:
        raise ValueError(f"Unknown loss function: {name}")

    return losses[name]

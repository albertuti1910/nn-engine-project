"""
Neural Network implementation with training functionality.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from src.layers import DenseLayer, Dropout
from src.losses import get_loss_function
from src.optimizers import Optimizer
from src.utils import create_mini_batches, accuracy_score


class NeuralNetwork:
    """Fully connected neural network."""

    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[str],
        initialization: str = 'xavier',
        dropout_rates: Optional[List[float]] = None
    ):
        """
        Initialize neural network.

        Args:
            layer_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            activations: List of activation functions for each layer
            initialization: Weight initialization method
            dropout_rates: List of dropout rates for each layer (None for no dropout)
        """
        assert len(layer_sizes) >= 2, "Need at least input and output layer"
        assert len(activations) == len(layer_sizes) - 1, \
            "Number of activations must match number of layers"

        self.layer_sizes = layer_sizes
        self.layers = []
        self.dropout_layers = []

        for i in range(len(layer_sizes) - 1):
            layer = DenseLayer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activations[i],
                initialization=initialization
            )
            self.layers.append(layer)

            if dropout_rates is not None and i < len(dropout_rates):
                dropout = Dropout(dropout_rate=dropout_rates[i])
                self.dropout_layers.append(dropout)
            else:
                self.dropout_layers.append(None)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            X: Input data (batch_size, input_size)
            training: Whether in training mode (affects dropout)

        Returns:
            Network output (batch_size, output_size)
        """
        activation = X
        for i, layer in enumerate(self.layers):
            activation = layer.forward(activation)

            if self.dropout_layers[i] is not None:
                self.dropout_layers[i].set_training(training)
                activation = self.dropout_layers[i].forward(activation)

        return activation

    def backward(self, loss_grad: np.ndarray) -> None:
        """
        Backward pass through the network.

        Args:
            loss_grad: Gradient of loss with respect to network output
        """
        grad = loss_grad
        for i in reversed(range(len(self.layers))):
            if self.dropout_layers[i] is not None:
                grad = self.dropout_layers[i].backward(grad)

            grad = self.layers[i].backward(grad)

    def get_params(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get all network parameters.

        Returns:
            List of (weights, bias) tuples for each layer
        """
        return [layer.get_params() for layer in self.layers]

    def get_gradients(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get all network gradients.

        Returns:
            List of (dW, db) tuples for each layer
        """
        return [layer.get_gradients() for layer in self.layers]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.

        Args:
            X: Input data

        Returns:
            Predictions
        """
        return self.forward(X, training=False)

    def set_training(self, training: bool) -> None:
        """Set training mode for all dropout layers."""
        for dropout in self.dropout_layers:
            if dropout is not None:
                dropout.set_training(training)

    def __repr__(self) -> str:
        """String representation of the network."""
        return f"NeuralNetwork(layers={self.layer_sizes})"


class Trainer:
    """Trainer class for managing the training process."""

    def __init__(
        self,
        network: NeuralNetwork,
        optimizer: Optimizer,
        loss_function: str = 'categorical_crossentropy',
        scheduler: Optional[object] = None
    ):
        """
        Initialize trainer.

        Args:
            network: Neural network to train
            optimizer: Optimizer instance
            loss_function: Loss function name
            scheduler: Learning rate scheduler (optional)
        """
        self.network = network
        self.optimizer = optimizer
        self.loss_fn = get_loss_function(loss_function)
        self.scheduler = scheduler

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

    def train_epoch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size: int,
        shuffle: bool = True
    ) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            X_train: Training features
            y_train: Training labels (one-hot encoded)
            batch_size: Mini-batch size
            shuffle: Whether to shuffle data

        Returns:
            Average loss and accuracy for the epoch
        """
        self.network.set_training(True)
        batches = create_mini_batches(X_train, y_train, batch_size, shuffle)

        epoch_loss = 0.0
        epoch_acc = 0.0

        for X_batch, y_batch in batches:
            y_pred = self.network.forward(X_batch, training=True)

            loss = self.loss_fn.compute(y_pred, y_batch)
            epoch_loss += loss * X_batch.shape[0]

            acc = accuracy_score(y_batch, y_pred)
            epoch_acc += acc * X_batch.shape[0]

            loss_grad = self.loss_fn.gradient(y_pred, y_batch)

            self.network.backward(loss_grad)

            params = self.network.get_params()
            grads = self.network.get_gradients()
            self.optimizer.update(params, grads)

        avg_loss = epoch_loss / X_train.shape[0]
        avg_acc = epoch_acc / X_train.shape[0]

        return avg_loss, avg_acc

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 128
    ) -> Tuple[float, float]:
        """
        Evaluate network on given data.

        Args:
            X: Features
            y: Labels (one-hot encoded)
            batch_size: Batch size for evaluation

        Returns:
            Loss and accuracy
        """
        self.network.set_training(False)
        batches = create_mini_batches(X, y, batch_size, shuffle=False)

        total_loss = 0.0
        total_acc = 0.0

        for X_batch, y_batch in batches:
            y_pred = self.network.predict(X_batch)

            loss = self.loss_fn.compute(y_pred, y_batch)
            total_loss += loss * X_batch.shape[0]

            acc = accuracy_score(y_batch, y_pred)
            total_acc += acc * X_batch.shape[0]

        avg_loss = total_loss / X.shape[0]
        avg_acc = total_acc / X.shape[0]

        return avg_loss, avg_acc

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1,
        early_stopping: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train the neural network.

        Args:
            X_train: Training features
            y_train: Training labels (one-hot encoded)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Mini-batch size
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
            early_stopping: Stop if validation loss doesn't improve for N epochs

        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(
                X_train, y_train, batch_size, shuffle=True
            )

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            current_lr = self.optimizer.learning_rate if hasattr(self.optimizer, 'learning_rate') else 0
            self.history['learning_rate'].append(current_lr)

            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val, batch_size)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                if verbose >= 2:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - "
                          f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - "
                          f"lr: {current_lr:.6f}")

                if early_stopping is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping:
                            if verbose >= 1:
                                print(f"Early stopping at epoch {epoch+1}")
                            break
            else:
                if verbose >= 2:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - "
                          f"lr: {current_lr:.6f}")

            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step'):
                    if 'Plateau' in self.scheduler.__class__.__name__ and val_loss is not None:
                        new_lr = self.scheduler.step(epoch, val_loss)
                    else:
                        new_lr = self.scheduler.step(epoch)

                    if hasattr(self.optimizer, 'learning_rate'):
                        self.optimizer.learning_rate = new_lr

        if verbose == 1 and epochs > 0:
            print(f"Training completed. Final train_acc: {train_acc:.4f}")

        return self.history

    def get_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        return self.history

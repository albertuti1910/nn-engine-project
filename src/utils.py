"""
Utility functions for data preprocessing, batching, and metrics.
"""
import numpy as np
from typing import Tuple, List, Optional


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        X: Features (n_samples, n_features)
        y: Labels (n_samples, n_classes) or (n_samples,)
        train_size: Fraction for training set
        val_size: Fraction for validation set
        test_size: Fraction for test set
        random_seed: Random seed for reproducibility

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "train_size + val_size + test_size must equal 1.0"

    if random_seed is not None:
        np.random.seed(random_seed)

    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)

    train_end = int(n_samples * train_size)
    val_end = train_end + int(n_samples * val_size)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_mini_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create mini-batches from dataset.

    Args:
        X: Features (n_samples, n_features)
        y: Labels (n_samples, n_classes)
        batch_size: Size of each mini-batch
        shuffle: Whether to shuffle data before batching

    Returns:
        List of (X_batch, y_batch) tuples
    """
    n_samples = X.shape[0]

    if shuffle:
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

    batches = []
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        X_batch = X[i:end_idx]
        y_batch = y[i:end_idx]
        batches.append((X_batch, y_batch))

    return batches


def one_hot_encode(y: np.ndarray, num_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert class labels to one-hot encoded vectors.

    Args:
        y: Class labels (n_samples,) with integer values
        num_classes: Number of classes (if None, inferred from y)

    Returns:
        One-hot encoded array (n_samples, num_classes)
    """
    if num_classes is None:
        num_classes = int(np.max(y)) + 1

    n_samples = y.shape[0]
    y_one_hot = np.zeros((n_samples, num_classes))
    y_one_hot[np.arange(n_samples), y.astype(int)] = 1

    return y_one_hot


def normalize_features(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    method: str = 'standard'
) -> Tuple[np.ndarray, ...]:
    """
    Normalize features using training set statistics.

    Args:
        X_train: Training features
        X_val: Validation features (optional)
        X_test: Test features (optional)
        method: Normalization method ('standard' or 'minmax')

    Returns:
        Normalized arrays (X_train, X_val, X_test) or just X_train
    """
    if method == 'standard':
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-8

        X_train_norm = (X_train - mean) / std

        results = [X_train_norm]

        if X_val is not None:
            X_val_norm = (X_val - mean) / std
            results.append(X_val_norm)

        if X_test is not None:
            X_test_norm = (X_test - mean) / std
            results.append(X_test_norm)

        return tuple(results) if len(results) > 1 else results[0]

    elif method == 'minmax':
        min_val = np.min(X_train, axis=0)
        max_val = np.max(X_train, axis=0)
        range_val = max_val - min_val + 1e-8

        X_train_norm = (X_train - min_val) / range_val

        results = [X_train_norm]

        if X_val is not None:
            X_val_norm = (X_val - min_val) / range_val
            results.append(X_val_norm)

        if X_test is not None:
            X_test_norm = (X_test - min_val) / range_val
            results.append(X_test_norm)

        return tuple(results) if len(results) > 1 else results[0]

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate classification accuracy.

    Args:
        y_true: True labels (one-hot or class indices)
        y_pred: Predicted probabilities (n_samples, n_classes)

    Returns:
        Accuracy score
    """
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)

    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    return np.mean(y_true == y_pred)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels (class indices)
        y_pred: Predicted labels (class indices)
        num_classes: Number of classes

    Returns:
        Confusion matrix (num_classes, num_classes)
    """
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)

    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1

    return cm


def k_fold_indices(n_samples: int, k: int, random_seed: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate indices for k-fold cross-validation.

    Args:
        n_samples: Number of samples
        k: Number of folds
        random_seed: Random seed for reproducibility

    Returns:
        List of (train_indices, val_indices) tuples for each fold
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    indices = np.random.permutation(n_samples)
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_indices, val_indices))
        current = stop

    return folds

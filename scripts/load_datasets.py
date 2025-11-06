"""
Dataset loading utilities for various datasets (MNIST, IRIS, synthetic, etc.).
Provides robust hybrid loaders that work offline or via OpenML.
"""

import os
import numpy as np
from typing import Tuple
from urllib import request
from sklearn.datasets import fetch_openml


def download_file(url: str, filename: str) -> None:
    """Download file from URL if it doesn't exist."""
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        print(f"Downloading {filename}...")
        request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")


# ====================
# MNIST Loader
# ====================
def load_mnist(data_dir: str = "../data/mnist", cache: bool = True) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load MNIST dataset using hybrid strategy:
        1. Load from local cache (npz)
        2. Fetch from OpenML
        3. Fallback to Kaggle CSV (if manually downloaded)
    """
    os.makedirs(data_dir, exist_ok=True)
    cache_file = os.path.join(data_dir, "mnist_cached.npz")

    # 1️⃣ Try local cache
    if cache and os.path.exists(cache_file):
        print("Loading MNIST from local cache...")
        with np.load(cache_file) as data:
            return (data["X_train"], data["y_train"]), (data["X_test"], data["y_test"])

    # Try OpenML
    try:
        print("Downloading MNIST from OpenML...")
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        X = X.astype(np.float32) / 255.0
        y = y.astype(np.int32)

        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]

        if cache:
            np.savez_compressed(cache_file,
                                X_train=X_train, y_train=y_train,
                                X_test=X_test, y_test=y_test)
            print(f"Saved cached dataset to {cache_file}")

        return (X_train, y_train), (X_test, y_test)

    except Exception as e:
        print(f"OpenML MNIST fetch failed: {e}")

    # Try Kaggle CSV fallback
    csv_train = os.path.join(data_dir, "mnist_train.csv")
    csv_test = os.path.join(data_dir, "mnist_test.csv")

    if os.path.exists(csv_train) and os.path.exists(csv_test):
        print("Loading MNIST from Kaggle CSV files...")
        train_data = np.loadtxt(csv_train, delimiter=",", skiprows=1)
        test_data = np.loadtxt(csv_test, delimiter=",", skiprows=1)

        y_train = train_data[:, 0].astype(np.int32)
        X_train = train_data[:, 1:].astype(np.float32) / 255.0

        y_test = test_data[:, 0].astype(np.int32)
        X_test = test_data[:, 1:].astype(np.float32) / 255.0

        if cache:
            np.savez_compressed(cache_file,
                                X_train=X_train, y_train=y_train,
                                X_test=X_test, y_test=y_test)
            print(f"Saved cached dataset to {cache_file}")

        return (X_train, y_train), (X_test, y_test)

    raise RuntimeError(
        "MNIST dataset not found. "
        "Tried local cache, OpenML, and Kaggle CSV files."
    )


# ====================
# IRIS Loader
# ====================
def load_iris_dataset(data_dir: str = "../data/iris", cache: bool = True) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """
    Load IRIS dataset using hybrid strategy:
        1. Load from local cache (npz)
        2. Fetch from sklearn.datasets (preferred)
        3. Fallback to OpenML

    Returns:
        X, y, feature_names, class_names
    """
    from sklearn.datasets import load_iris as sklearn_load_iris
    from sklearn.datasets import fetch_openml

    os.makedirs(data_dir, exist_ok=True)
    cache_file = os.path.join(data_dir, "iris_cached.npz")

    # Try local cache (data only)
    if cache and os.path.exists(cache_file):
        print("Loading IRIS from local cache...")
        with np.load(cache_file) as data:
            X, y = data["X"], data["y"]
        # names not stored in cache
        feature_names = ["sepal length", "sepal width", "petal length", "petal width"]
        class_names = ["setosa", "versicolor", "virginica"]
        return X, y, feature_names, class_names

    # Try sklearn.datasets
    try:
        print("Loading IRIS from sklearn.datasets...")
        iris = sklearn_load_iris(as_frame=False)
        X = iris.data.astype(np.float32)
        y = iris.target.astype(np.int32)
        feature_names = list(iris.feature_names)
        class_names = list(iris.target_names)

        if cache:
            np.savez_compressed(cache_file, X=X, y=y)
            print(f"Saved cached dataset to {cache_file}")

        return X, y, feature_names, class_names

    except Exception as e:
        print(f"sklearn IRIS load failed: {e}")

    # Fallback to OpenML
    try:
        print("Downloading IRIS from OpenML...")
        data = fetch_openml("iris", version=1, as_frame=True)
        X = data.data.to_numpy(dtype=np.float32)
        y = data.target.astype("category").cat.codes.to_numpy(dtype=np.int32)
        feature_names = list(data.feature_names)
        class_names = list(data.categories["class"])

        if cache:
            np.savez_compressed(cache_file, X=X, y=y)
            print(f"Saved cached dataset to {cache_file}")

        return X, y, feature_names, class_names

    except Exception as e:
        raise RuntimeError(f"IRIS dataset could not be loaded from any source: {e}")

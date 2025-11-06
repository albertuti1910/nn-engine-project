"""
Neural Network Engine - A from-scratch implementation of fully connected neural networks.
"""

from src.network import NeuralNetwork, Trainer, cross_validate
from src.layers import DenseLayer, Dropout, Sigmoid, ReLU, Tanh, Softmax
from src.optimizers import Adam, SGD, RMSprop
from src.losses import MeanSquaredError, CategoricalCrossEntropy, BinaryCrossEntropy
from src.schedulers import (
    StepDecayLR,
    ExponentialDecayLR,
    CosineAnnealingLR,
    PolynomialDecayLR,
    ReduceLROnPlateau
)
from src.utils import (
    train_val_test_split,
    create_mini_batches,
    one_hot_encode,
    normalize_features,
    accuracy_score,
    confusion_matrix,
    k_fold_indices
)

__version__ = '0.1.0'
__all__ = [
    'NeuralNetwork',
    'Trainer',
    'DenseLayer',
    'Dropout',
    'Sigmoid',
    'ReLU',
    'Tanh',
    'Softmax',
    'Adam',
    'SGD',
    'RMSprop',
    'MeanSquaredError',
    'CategoricalCrossEntropy',
    'BinaryCrossEntropy',
    'StepDecayLR',
    'ExponentialDecayLR',
    'CosineAnnealingLR',
    'PolynomialDecayLR',
    'ReduceLROnPlateau',
    'train_val_test_split',
    'create_mini_batches',
    'one_hot_encode',
    'normalize_features',
    'accuracy_score',
    'confusion_matrix',
    'k_fold_indices',
    'cross_validate'
]
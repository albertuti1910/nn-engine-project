"""
Unit tests for the neural network engine.
"""
import sys
import numpy as np

sys.path.append('..')

from src import NeuralNetwork, Trainer, Adam, SGD
from src.utils import one_hot_encode, normalize_features, train_val_test_split
from src.losses import get_loss_function


def test_one_hot_encoding():
    """Test one-hot encoding function."""
    print("Testing one-hot encoding...")
    y = np.array([0, 1, 2, 0, 1])
    y_encoded = one_hot_encode(y, num_classes=3)

    expected = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0]
    ])

    assert np.allclose(y_encoded, expected), "One-hot encoding failed"
    print("✓ One-hot encoding passed")


def test_normalization():
    """Test feature normalization."""
    print("Testing normalization...")
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    X_test = np.array([[2, 3], [4, 5]])

    X_train_norm, X_test_norm = normalize_features(X_train, X_test=X_test, method='standard')

    assert np.abs(np.mean(X_train_norm, axis=0)).sum() < 1e-10, "Mean should be ~0"
    assert np.abs(np.std(X_train_norm, axis=0) - 1.0).sum() < 1e-1, "Std should be ~1"
    print("✓ Normalization passed")


def test_train_val_test_split():
    """Test dataset splitting."""
    print("Testing train/val/test split...")
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, size=100)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_seed=42
    )

    assert X_train.shape[0] == 70, "Train set size incorrect"
    assert X_val.shape[0] == 15, "Val set size incorrect"
    assert X_test.shape[0] == 15, "Test set size incorrect"
    assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == 100, "Total samples incorrect"
    print("✓ Train/val/test split passed")


def test_network_forward():
    """Test forward pass."""
    print("Testing forward pass...")
    network = NeuralNetwork(
        layer_sizes=[10, 5, 3],
        activations=['relu', 'softmax'],
        initialization='xavier'
    )

    X = np.random.randn(32, 10)
    output = network.forward(X)

    assert output.shape == (32, 3), "Output shape incorrect"
    assert np.allclose(output.sum(axis=1), 1.0), "Softmax should sum to 1"
    assert np.all(output >= 0) and np.all(output <= 1), "Softmax should be in [0, 1]"
    print("✓ Forward pass passed")


def test_network_backward():
    """Test backward pass."""
    print("Testing backward pass...")
    network = NeuralNetwork(
        layer_sizes=[5, 4, 3],
        activations=['sigmoid', 'softmax'],
        initialization='xavier'
    )

    X = np.random.randn(10, 5)
    y = one_hot_encode(np.random.randint(0, 3, size=10), num_classes=3)

    y_pred = network.forward(X)

    loss_fn = get_loss_function('categorical_crossentropy')
    loss_grad = loss_fn.gradient(y_pred, y)

    network.backward(loss_grad)

    grads = network.get_gradients()
    assert len(grads) == 2, "Should have gradients for 2 layers"

    for dW, db in grads:
        assert dW is not None and db is not None, "Gradients should not be None"
        assert not np.isnan(dW).any() and not np.isnan(db).any(), "Gradients contain NaN"

    print("✓ Backward pass passed")


def test_optimizer_update():
    """Test optimizer parameter updates."""
    print("Testing optimizer updates...")
    network = NeuralNetwork(
        layer_sizes=[5, 3],
        activations=['sigmoid'],
        initialization='xavier'
    )

    optimizer = Adam(learning_rate=0.01)

    params_before = [(w.copy(), b.copy()) for w, b in network.get_params()]

    X = np.random.randn(10, 5)
    y = one_hot_encode(np.random.randint(0, 3, size=10), num_classes=3)

    y_pred = network.forward(X)
    loss_fn = get_loss_function('categorical_crossentropy')
    loss_grad = loss_fn.gradient(y_pred, y)
    network.backward(loss_grad)

    params = network.get_params()
    grads = network.get_gradients()
    optimizer.update(params, grads)

    params_after = network.get_params()

    for (w_before, b_before), (w_after, b_after) in zip(params_before, params_after):
        assert not np.allclose(w_before, w_after), "Weights should have changed"
        assert not np.allclose(b_before, b_after), "Biases should have changed"

    print("✓ Optimizer update passed")


def test_training_decreases_loss():
    """Test that training decreases loss."""
    print("Testing training process...")
    np.random.seed(42)

    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, size=100)
    y = one_hot_encode(y, num_classes=3)

    X_train, X_val, _, y_train, y_val, _ = train_val_test_split(
        X, y, train_size=0.7, val_size=0.2, test_size=0.1, random_seed=42
    )

    network = NeuralNetwork(
        layer_sizes=[5, 10, 3],
        activations=['relu', 'softmax'],
        initialization='xavier'
    )

    optimizer = Adam(learning_rate=0.01)
    trainer = Trainer(network, optimizer, loss_function='categorical_crossentropy')

    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=20,
        batch_size=16,
        verbose=0
    )

    initial_loss = history['train_loss'][0]
    final_loss = history['train_loss'][-1]

    assert final_loss < initial_loss, "Training should decrease loss"
    print(f"  Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
    print("✓ Training decreases loss passed")


def test_gradient_numerical_check():
    """Numerical gradient checking."""
    print("Testing numerical gradient check...")

    def compute_loss(network, X, y, loss_fn):
        y_pred = network.forward(X)
        return loss_fn.compute(y_pred, y)

    network = NeuralNetwork(
        layer_sizes=[3, 4, 2],
        activations=['sigmoid', 'sigmoid'],
        initialization='random'
    )

    X = np.random.randn(5, 3)
    y = one_hot_encode(np.array([0, 1, 0, 1, 0]), num_classes=2)

    loss_fn = get_loss_function('categorical_crossentropy')

    y_pred = network.forward(X)
    loss_grad = loss_fn.gradient(y_pred, y)
    network.backward(loss_grad)

    analytical_grads = network.get_gradients()

    epsilon = 1e-5
    params = network.get_params()

    numerical_grads = []
    for w, b in params:
        num_dW = np.zeros_like(w)
        num_db = np.zeros_like(b)

        for i in range(min(3, w.shape[0])):
            for j in range(min(3, w.shape[1])):
                w[i, j] += epsilon
                loss_plus = compute_loss(network, X, y, loss_fn)
                w[i, j] -= 2 * epsilon
                loss_minus = compute_loss(network, X, y, loss_fn)
                w[i, j] += epsilon

                num_dW[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

        numerical_grads.append((num_dW, num_db))

    for (analytical_dW, _), (numerical_dW, _) in zip(analytical_grads, numerical_grads):
        for i in range(min(3, analytical_dW.shape[0])):
            for j in range(min(3, analytical_dW.shape[1])):
                diff = abs(analytical_dW[i, j] - numerical_dW[i, j])
                if numerical_dW[i, j] != 0:
                    rel_diff = diff / abs(numerical_dW[i, j])
                    assert rel_diff < 1e-2, f"Gradient check failed: rel_diff={rel_diff}"

    print("✓ Numerical gradient check passed")


def run_all_tests():
    """Run all unit tests."""
    print("="*60)
    print("Running Unit Tests for Neural Network Engine")
    print("="*60)
    print()

    try:
        test_one_hot_encoding()
        test_normalization()
        test_train_val_test_split()
        test_network_forward()
        test_network_backward()
        test_optimizer_update()
        test_training_decreases_loss()
        test_gradient_numerical_check()

        print()
        print("="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        return True

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

"""
Learning rate schedules for optimizers.
"""
import numpy as np
from typing import Optional


class LearningRateScheduler:
    """Base class for learning rate schedulers."""

    def __init__(self, initial_lr: float):
        """
        Initialize scheduler.

        Args:
            initial_lr: Initial learning rate
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr

    def step(self, epoch: int) -> float:
        """
        Update learning rate based on epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Updated learning rate
        """
        raise NotImplementedError

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr


class ConstantLR(LearningRateScheduler):
    """Constant learning rate (no decay)."""

    def step(self, epoch: int) -> float:
        """Return constant learning rate."""
        return self.current_lr


class StepDecayLR(LearningRateScheduler):
    """Step decay learning rate scheduler."""

    def __init__(
        self,
        initial_lr: float,
        decay_rate: float = 0.5,
        decay_steps: int = 10
    ):
        """
        Initialize step decay scheduler.

        Args:
            initial_lr: Initial learning rate
            decay_rate: Factor to multiply LR by
            decay_steps: Decay every N epochs
        """
        super().__init__(initial_lr)
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def step(self, epoch: int) -> float:
        """
        Update learning rate with step decay.

        Args:
            epoch: Current epoch number

        Returns:
            Updated learning rate
        """
        self.current_lr = self.initial_lr * (self.decay_rate ** (epoch // self.decay_steps))
        return self.current_lr


class ExponentialDecayLR(LearningRateScheduler):
    """Exponential decay learning rate scheduler."""

    def __init__(
        self,
        initial_lr: float,
        decay_rate: float = 0.95
    ):
        """
        Initialize exponential decay scheduler.

        Args:
            initial_lr: Initial learning rate
            decay_rate: Decay rate per epoch
        """
        super().__init__(initial_lr)
        self.decay_rate = decay_rate

    def step(self, epoch: int) -> float:
        """
        Update learning rate with exponential decay.

        Args:
            epoch: Current epoch number

        Returns:
            Updated learning rate
        """
        self.current_lr = self.initial_lr * (self.decay_rate ** epoch)
        return self.current_lr


class CosineAnnealingLR(LearningRateScheduler):
    """Cosine annealing learning rate scheduler."""

    def __init__(
        self,
        initial_lr: float,
        T_max: int,
        eta_min: float = 0.0
    ):
        """
        Initialize cosine annealing scheduler.

        Args:
            initial_lr: Initial learning rate
            T_max: Maximum number of epochs
            eta_min: Minimum learning rate
        """
        super().__init__(initial_lr)
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self, epoch: int) -> float:
        """
        Update learning rate with cosine annealing.

        Args:
            epoch: Current epoch number

        Returns:
            Updated learning rate
        """
        self.current_lr = self.eta_min + (self.initial_lr - self.eta_min) * \
                          (1 + np.cos(np.pi * epoch / self.T_max)) / 2
        return self.current_lr


class PolynomialDecayLR(LearningRateScheduler):
    """Polynomial decay learning rate scheduler."""

    def __init__(
        self,
        initial_lr: float,
        max_epochs: int,
        power: float = 1.0,
        end_lr: float = 0.0001
    ):
        """
        Initialize polynomial decay scheduler.

        Args:
            initial_lr: Initial learning rate
            max_epochs: Maximum number of epochs
            power: Polynomial power
            end_lr: Final learning rate
        """
        super().__init__(initial_lr)
        self.max_epochs = max_epochs
        self.power = power
        self.end_lr = end_lr

    def step(self, epoch: int) -> float:
        """
        Update learning rate with polynomial decay.

        Args:
            epoch: Current epoch number

        Returns:
            Updated learning rate
        """
        if epoch >= self.max_epochs:
            self.current_lr = self.end_lr
        else:
            decay = (1 - epoch / self.max_epochs) ** self.power
            self.current_lr = (self.initial_lr - self.end_lr) * decay + self.end_lr
        return self.current_lr


class ReduceLROnPlateau(LearningRateScheduler):
    """Reduce learning rate when metric plateaus."""

    def __init__(
        self,
        initial_lr: float,
        factor: float = 0.5,
        patience: int = 10,
        min_lr: float = 1e-6,
        threshold: float = 1e-4
    ):
        """
        Initialize plateau scheduler.

        Args:
            initial_lr: Initial learning rate
            factor: Factor to reduce LR by
            patience: Number of epochs to wait before reducing
            min_lr: Minimum learning rate
            threshold: Threshold for measuring improvement
        """
        super().__init__(initial_lr)
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold

        self.best_loss = float('inf')
        self.num_bad_epochs = 0

    def step(self, epoch: int, val_loss: Optional[float] = None) -> float:
        """
        Update learning rate based on validation loss.

        Args:
            epoch: Current epoch number
            val_loss: Validation loss

        Returns:
            Updated learning rate
        """
        if val_loss is None:
            return self.current_lr

        if val_loss < self.best_loss - self.threshold:
            self.best_loss = val_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.num_bad_epochs = 0

        return self.current_lr


def get_scheduler(name: str, initial_lr: float, **kwargs) -> LearningRateScheduler:
    """
    Factory function to get scheduler by name.

    Args:
        name: Scheduler name
        initial_lr: Initial learning rate
        **kwargs: Scheduler-specific parameters

    Returns:
        LearningRateScheduler instance
    """
    schedulers = {
        'constant': ConstantLR,
        'step': StepDecayLR,
        'exponential': ExponentialDecayLR,
        'cosine': CosineAnnealingLR,
        'polynomial': PolynomialDecayLR,
        'plateau': ReduceLROnPlateau,
    }

    if name not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}")

    return schedulers[name](initial_lr, **kwargs)

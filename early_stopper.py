class EarlyStopping:
    """Stop after a minimized validation metric stops improving.

    Args:
        patience: Consecutive unimproved epochs to allow. Zero disables stopping.
        min_delta: Minimum metric decrease that counts as an improvement.

    Call the instance once per epoch. It returns ``True`` when training should
    stop and resets its counter whenever sufficient improvement is observed.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-4) -> None:
        if patience < 0:
            raise ValueError("patience must be non-negative")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0

    def __call__(self, metric: float) -> bool:
        """Update state with the latest metric and report whether to stop."""
        if metric < self.best - self.min_delta:
            self.best = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.patience > 0 and self.bad_epochs >= self.patience

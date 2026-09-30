import numpy as np


def make_dataset(
    n_samples: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic binary classification dataset."""

    rng = np.random.default_rng(seed)

    # Four numerical features.
    X = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_samples, 4),
    ).astype(np.float32)

    # Create a deterministic binary target.
    score = (
        0.8 * X[:, 0]
        + 0.5 * X[:, 1]
        - 0.6 * X[:, 2]
        + 0.3 * X[:, 3]
    )

    y = (score > 0.0).astype(np.float32)

    # Shuffle before splitting.
    indices = rng.permutation(n_samples)

    split_index = int(n_samples * 0.8)

    train_indices = indices[:split_index]
    validation_indices = indices[split_index:]

    X_train = X[train_indices]
    X_val = X[validation_indices]

    y_train = y[train_indices]
    y_val = y[validation_indices]

    return X_train, X_val, y_train, y_val
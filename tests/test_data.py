import numpy as np

from ml.data import make_dataset


def test_make_dataset_is_reproducible():
    first = make_dataset(
        n_samples=100,
        seed=42,
    )

    second = make_dataset(
        n_samples=100,
        seed=42,
    )

    for first_array, second_array in zip(first, second):
        np.testing.assert_array_equal(
            first_array,
            second_array,
        )


def test_make_dataset_split_sizes():
    X_train, X_val, y_train, y_val = make_dataset(
        n_samples=100,
        seed=42,
    )

    assert len(X_train) == 80
    assert len(X_val) == 20
    assert len(y_train) == 80
    assert len(y_val) == 20


def test_make_dataset_has_four_features():
    X_train, X_val, _, _ = make_dataset(
        n_samples=100,
        seed=42,
    )

    assert X_train.shape[1] == 4
    assert X_val.shape[1] == 4
import numpy as np


def _validate_inputs(X_left, X_right):
    X_left = np.asarray(X_left, dtype=float)
    X_right = np.asarray(X_right, dtype=float)

    if X_left.ndim != 2 or X_right.ndim != 2:
        raise ValueError(
            "X_left and X_right must be 2D arrays."
        )

    if X_left.shape[1] != X_right.shape[1]:
        raise ValueError(
            "X_left and X_right must have the same number of features."
        )

    if len(X_left) == 0 or len(X_right) == 0:
        raise ValueError(
            "Both groups must contain at least one sample."
        )

    return X_left, X_right


def _euclidean_distances(X, Y):
    squared = (
        np.sum(X ** 2, axis=1)[:, None]
        + np.sum(Y ** 2, axis=1)[None, :]
        - 2.0 * X @ Y.T
    )

    squared = np.maximum(
        squared,
        0.0,
    )

    return np.sqrt(
        squared
    )


def compute_energy(
    X_left,
    X_right,
):
    """
    Compute the multivariate energy distance.

    D(X, Y) =
        2 E||X - Y||
        - E||X - X'||
        - E||Y - Y'||
    """

    X_left, X_right = _validate_inputs(
        X_left,
        X_right,
    )

    D_lr = _euclidean_distances(
        X_left,
        X_right,
    )

    D_ll = _euclidean_distances(
        X_left,
        X_left,
    )

    D_rr = _euclidean_distances(
        X_right,
        X_right,
    )

    value = (
        2.0 * D_lr.mean()
        - D_ll.mean()
        - D_rr.mean()
    )

    # Protect against tiny negative values caused
    # by floating-point precision.
    return max(
        float(value),
        0.0,
    )
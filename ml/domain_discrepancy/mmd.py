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


def _squared_distances(X, Y):
    return (
        np.sum(X ** 2, axis=1)[:, None]
        + np.sum(Y ** 2, axis=1)[None, :]
        - 2.0 * X @ Y.T
    )


def _median_bandwidth(
    X_left,
    X_right,
    max_samples=1000,
):
    """
    Estimate RBF bandwidth using the median heuristic.
    """

    X = np.vstack([
        X_left,
        X_right,
    ])

    if len(X) > max_samples:
        indices = np.linspace(
            0,
            len(X) - 1,
            max_samples,
            dtype=int,
        )
        X = X[indices]

    distances = _squared_distances(
        X,
        X,
    )

    distances = distances[
        np.triu_indices(
            len(X),
            k=1,
        )
    ]

    distances = distances[
        distances > 0
    ]

    if len(distances) == 0:
        return 1.0

    median_distance = np.median(
        distances
    )

    return np.sqrt(
        median_distance / 2.0
    )


def _rbf_kernel(
    X,
    Y,
    sigma,
):
    distances = _squared_distances(
        X,
        Y,
    )

    return np.exp(
        -distances
        / (2.0 * sigma ** 2)
    )


def compute_mmd(
    X_left,
    X_right,
    sigma=None,
    squared=True,
    max_bandwidth_samples=1000,
):
    """
    Compute Maximum Mean Discrepancy using an RBF kernel.

    Parameters
    ----------
    X_left, X_right:
        Arrays with shape (n_samples, n_features).

    sigma:
        RBF bandwidth. If None, the median heuristic
        is used.

    squared:
        If True, return MMD^2.
        Otherwise return MMD.

    max_bandwidth_samples:
        Maximum number of samples used by the
        median-bandwidth heuristic.
    """

    X_left, X_right = _validate_inputs(
        X_left,
        X_right,
    )

    if sigma is None:
        sigma = _median_bandwidth(
            X_left,
            X_right,
            max_samples=max_bandwidth_samples,
        )

    if sigma <= 0:
        raise ValueError(
            "sigma must be greater than zero."
        )

    K_ll = _rbf_kernel(
        X_left,
        X_left,
        sigma,
    )

    K_rr = _rbf_kernel(
        X_right,
        X_right,
        sigma,
    )

    K_lr = _rbf_kernel(
        X_left,
        X_right,
        sigma,
    )

    # Biased empirical estimator.
    # It is stable and non-negative apart from
    # small floating-point errors.
    mmd_squared = (
        K_ll.mean()
        + K_rr.mean()
        - 2.0 * K_lr.mean()
    )

    mmd_squared = max(
        float(mmd_squared),
        0.0,
    )

    if squared:
        return mmd_squared

    return float(
        np.sqrt(mmd_squared)
    )
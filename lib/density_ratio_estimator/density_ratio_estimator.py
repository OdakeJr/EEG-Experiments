import numpy as np


class DensityRatioEstimator:
    """Base interface for w(x) = p_target(x) / p_source(x)."""

    def fit(self, X_source, X_target):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def fit_predict(self, X_source, X_target):
        self.fit(X_source, X_target)
        return self.predict(X_source)


def _rbf_kernel(X, centers, sigma):
    dist2 = (
        np.sum(X**2, axis=1)[:, None]
        + np.sum(centers**2, axis=1)[None, :]
        - 2 * X @ centers.T
    )
    return np.exp(-dist2 / (2 * sigma**2))


def _median_sigma(X, max_samples=1000, random_state=42):
    rng = np.random.default_rng(random_state)

    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]

    dist2 = (
        np.sum(X**2, axis=1)[:, None]
        + np.sum(X**2, axis=1)[None, :]
        - 2 * X @ X.T
    )

    distances = np.sqrt(np.maximum(dist2[np.triu_indices(len(X), 1)], 0))
    distances = distances[distances > 0]

    return np.median(distances) if len(distances) else 1.0


class KLIEP(DensityRatioEstimator):
    """Kullback-Leibler Importance Estimation Procedure."""

    def __init__(
        self,
        sigma=None,
        n_centers=100,
        learning_rate=0.1,
        max_iter=500,
        tol=1e-6,
        random_state=42,
    ):
        self.sigma = sigma
        self.n_centers = n_centers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X_source, X_target):
        X_source = np.asarray(X_source)
        X_target = np.asarray(X_target)

        rng = np.random.default_rng(self.random_state)
        n_centers = min(self.n_centers, len(X_target))

        idx = rng.choice(len(X_target), n_centers, replace=False)
        self.centers_ = X_target[idx]

        self.sigma_ = self.sigma or _median_sigma(
            np.vstack([X_source, X_target]),
            random_state=self.random_state
        )

        Ks = _rbf_kernel(X_source, self.centers_, self.sigma_)
        Kt = _rbf_kernel(X_target, self.centers_, self.sigma_)

        source_mean = Ks.mean(axis=0)
        alpha = np.ones(n_centers)
        alpha /= np.dot(source_mean, alpha)

        previous_objective = -np.inf

        for _ in range(self.max_iter):
            weights_target = np.maximum(Kt @ alpha, 1e-12)

            gradient = (
                Kt.T @ (1.0 / weights_target)
            ) / len(X_target)

            alpha += self.learning_rate * gradient
            alpha = np.maximum(alpha, 0)

            normalization = np.dot(source_mean, alpha)
            if normalization <= 0:
                break

            alpha /= normalization

            objective = np.mean(
                np.log(np.maximum(Kt @ alpha, 1e-12))
            )

            if abs(objective - previous_objective) < self.tol:
                break

            previous_objective = objective

        self.alpha_ = alpha
        return self

    def predict(self, X):
        K = _rbf_kernel(
            np.asarray(X),
            self.centers_,
            self.sigma_
        )
        return np.maximum(K @ self.alpha_, 0)


class ULSIF(DensityRatioEstimator):
    """Unconstrained Least-Squares Importance Fitting."""

    def __init__(
        self,
        sigma=None,
        regularization=1e-3,
        n_centers=100,
        random_state=42,
    ):
        self.sigma = sigma
        self.regularization = regularization
        self.n_centers = n_centers
        self.random_state = random_state

    def fit(self, X_source, X_target):
        X_source = np.asarray(X_source)
        X_target = np.asarray(X_target)

        rng = np.random.default_rng(self.random_state)
        n_centers = min(self.n_centers, len(X_target))

        idx = rng.choice(len(X_target), n_centers, replace=False)
        self.centers_ = X_target[idx]

        self.sigma_ = self.sigma or _median_sigma(
            np.vstack([X_source, X_target]),
            random_state=self.random_state
        )

        Ks = _rbf_kernel(X_source, self.centers_, self.sigma_)
        Kt = _rbf_kernel(X_target, self.centers_, self.sigma_)

        H = Ks.T @ Ks / len(X_source)
        h = Kt.mean(axis=0)

        self.alpha_ = np.linalg.solve(
            H + self.regularization * np.eye(n_centers),
            h
        )

        # uLSIF uses the unconstrained solution, followed by
        # non-negativity correction for the estimated ratio.
        self.alpha_ = np.maximum(self.alpha_, 0)

        return self

    def predict(self, X):
        K = _rbf_kernel(
            np.asarray(X),
            self.centers_,
            self.sigma_
        )
        return np.maximum(K @ self.alpha_, 0)
    
def get_density_ratio_estimator(name, **params):
    estimators = {
        "kliep": KLIEP,
        "ulsif": ULSIF,
    }

    if name not in estimators:
        raise ValueError(f"Unknown density-ratio estimator: {name}")

    return estimators[name](**params)





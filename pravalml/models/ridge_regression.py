import numpy as np

from pravalml.models.linear_regression import LinearRegression
from pravalml.validation import check_X_y, check_X

class RidgeRegression(LinearRegression):
    def __init__(
            self,
            method: str = "normal",
            alpha: float = 1.0,
            fit_intercept: bool = True,
            lr: float = 1e-2,
            epochs: int = 1000
    ):
        super().__init__(fit_intercept=fit_intercept, method=method, lr=lr, epochs=epochs)
        self.alpha = float(alpha)
    
    def fit(self, X, y):
        X, y = check_X_y(X, y, X_dtype=float, y_dtype=float)

        if self.fit_intercept:
            X = self._add_intercept(X)

        if self.method == "normal":
            self._fit_normal(X, y)
        elif self.method == "gd":
            self._fit_gd(X, y)
        else:
            raise ValueError("method must be 'normal' or 'gd'")
        
        return self
    
    def _fit_normal(self, X, y):
        n_features = X.shape[1]

        reg = self.alpha * np.eye(n_features)
        if self.fit_intercept:
            reg[0, 0] = 0.0

        A = X.T @ X + reg
        b = X.T @ y

        self._weights = np.linalg.solve(A, b)

        if self.fit_intercept:
            self.intercept_ = float(self._weights[0])
            self.coef_ = self._weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self._weights

        self._unpack_weights()
        
    
    def _fit_gd(self, X, y):
        n_samples, n_features = X.shape

        self._weights = np.zeros(n_features, dtype=float)

        reg_mask = np.ones(n_features, dtype=float)
        if self.fit_intercept:
            reg_mask[0] = 0.0
        
        self.loss_history_ = []

        alpha_eff = self.alpha / n_samples

        for _ in range(self.epochs):
            y_pred = X @ self._weights
            residual = y_pred - y

            mse = (residual @ residual) / n_samples

            penalty = alpha_eff * np.sum((self._weights * reg_mask) ** 2)

            loss = mse + penalty
            self.loss_history_.append(float(loss))

            grad = (2.0 / n_samples) * (X.T @ residual)

            grad += 2.0 * alpha_eff * (self._weights * reg_mask)

            self._weights -= self.lr * grad
         
        if self.fit_intercept:
            self.intercept_ = float(self._weights[0])
            self.coef_ = self._weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self._weights

        self._unpack_weights()


        

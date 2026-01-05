import numpy as np

from pravalml.models.linear_regression import LinearRegression
from pravalml.validation import check_X_y

class LassoRegression(LinearRegression):
    def __init__(
            self, 
            alpha: float = 1.0,
            fit_intercept: bool = True,
            lr: float = 1e-2, 
            epochs: int = 1000
    ):    
        super().__init__(fit_intercept=fit_intercept,lr= lr,epochs= epochs, method="gd")
        self.alpha = alpha

    def fit(self, X, y):
        X, y = check_X_y(X, y, X_dtype=float, y_dtype=float)

        if self.fit_intercept:
            X = self._add_intercept(X)

        self._fit_gd(X, y)
        return self
            
    def _fit_normal(self, X, y):
        raise NotImplementedError("Lasso has no closed-form normal equation solution.")

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

            l1 = alpha_eff * np.sum(np.abs(self._weights * reg_mask))
            loss = mse + l1
            self.loss_history_.append(float(loss))

            grad = (2.0 / n_samples) * (X.T @ residual)

            subgrad = np.sign(self._weights)
            subgrad[self._weights == 0] = 0.0
            subgrad *= reg_mask

            grad += alpha_eff * subgrad

            self._weights -= self.lr * grad
        
        if self.fit_intercept:
            self.intercept_ = float(self._weights[0])
            self.coef_ = self._weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self._weights

        self._unpack_weights()
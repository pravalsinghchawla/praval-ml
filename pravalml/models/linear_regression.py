import numpy as np 

class LinearRegression:
    def __init__(
            self,
            fit_intercept: bool = True,
            method: str = "normal", # "normal" or "gd"
            lr: float = 1e-2,
            epochs: int = 1000,
    ):  
        self.fit_intercept = fit_intercept
        self.method = method
        self.lr = lr
        self.epochs = epochs

        self._weights = None
        self.coef_ = None
        self.intercept_ = None
        self.loss_history_ = []

        
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be 1D (n_samples,)")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        if self.fit_intercept:
            X = self._add_intercept(X)

        if self.method == "normal":
            self._fit_normal(X, y)
        elif self.method == "gd":
            self._fit_gd(X, y)
        else:
            raise ValueError("method must be 'normal' or 'gd'")

        return self
    
    def predict(self, X):
        if self._weights is None:
            raise ValueError("Model is not fitted yet. Call fit() first")
        
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        
        if self.fit_intercept:
            X = self._add_intercept(X)
        
        return X @ self._weights

        
    def _fit_normal(self, X, y):
        self._weights = np.linalg.pinv(X) @ y
        self._unpack_weights()
        
    def _fit_gd(self, X, y):
        n_samples, n_features = X.shape
        self._weights = np.zeros(n_features, dtype=float)
        self.loss_history_.clear()
        
        for _ in range(self.epochs):
            y_pred = X @ self._weights
            error = y_pred - y

            grad = (2 / n_samples) * (X.T @ error)
            self._weights -= self.lr * grad

            self.loss_history_.append(np.mean(error ** 2))

        self._unpack_weights()
    
    def _add_intercept(self, X):
        return np.c_[np.ones(X.shape[0]), X]
    
    def _unpack_weights(self):
        if self.fit_intercept:
            self.intercept_ = float(self._weights[0])
            self.coef_ = self._weights[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = self._weights.copy()

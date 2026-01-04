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

        self.coef = None
        self.intercept = None
        self.loss_history = []

    def fit(self, X, y):
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
        return X @ self._weights

        
    def _fit_normal(self, X, y):
        XtX = X.T @ X

        # Pseudoinverse for numerical stability
        self._weights = np.linalg.pinv(XtX) @ X.T @ y

        self._unpack_weights()
        
    def _fit_gd(self, X, y):
        n_samples, n_features = X.shape
        self._weights = np.zeros(n_features)
        
        for _ in range(self.epochs):
            y_pred = X @ self._weights
            error = y - y_pred

            grad = (2 / n_samples) * (X.T @ error)
            self._weights -= self.lr * grad

            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

        self._unpack_weights()
    
    def _add_intercept(self, X):
        return np.c_[np.ones(X.shape[0]), X]
    
    def _unpack_weights(self):
        if self.fit_intercept:
            self.intercept_ = self._weights[0]
            self.coef_ = self._weights[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = self._weights

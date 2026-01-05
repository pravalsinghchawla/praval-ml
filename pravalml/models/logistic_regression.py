import numpy as np

from pravalml.base import BaseEstimator, ClassifierMixin
from pravalml.validation import check_X_y, check_X

class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            fit_intercept: bool = True,
            lr: float = 1e-2,
            epochs: int = 1000,
            l2: float = 0.0,
            threshold: float = 0.5
    ):
        self.fit_intercept = fit_intercept
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.threshold = threshold

        self.coef_ = None
        self.intercept_ = None
        self.loss_history_ = []

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        y = self._encode_binary_labels(y)

        if self.fit_intercept:
            X = self._add_intercept(X)

        n_samples, n_features = X.shape
        self._weights = np.zeros(n_features, dtype=float)
        self.loss_history_ = []

        for _ in range(self.epochs):
            p = self._predict_proba_internal(X)

            error = p - y
            grad = (X.T @ error) / n_samples

            if self.l2 > 0.0:
                reg = self._weights.copy()
                if self.fit_intercept:
                    reg[0] = 0
                grad += self.l2 * reg

            self._weights -= self.lr * grad
            
            loss = self._log_loss(y, p)
            if self.l2 > 0.0:
                loss += 0.5 * self.l2 * np.sum(reg ** 2)
            self.loss_history_.append(loss)

        self._unpack_weights()
        return self
    
    def predict_proba(self, X):
        X = check_X(X)
        if self.fit_intercept:
            X = self._add_intercept(X)
        p = self._predict_proba_internal(X)
        
        return np.column_stack([1.0 - p, p])
    
    def predict(self, X):
        proba_pos = self.predict_proba(X)[:, 1]
        return (proba_pos >= self.threshold).astype(int)
    
    def _predict_proba_internal(self, X):
        z = X @ self._weights
        return self._sigmoid(z)
    
    @staticmethod
    def _sigmoid(z):
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        expz = np.exp(z[neg])
        out[neg] = expz / (1.0 + expz)

        return out
    
    @staticmethod
    def _log_loss(y, p):
        eps = 1e-15
        p = np.clip(p, eps, 1.0 - eps)
        return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    
    @staticmethod
    def _add_intercept(X):
        return np.c_[np.ones(X.shape[0]), X]
    
    def _unpack_weights(self):
        if self.fit_intercept:
            self.intercept_ = float(self._weights[0])
            self.coef_ = self._weights[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = self._weights.copy()

    @staticmethod
    def _encode_binary_labels(y):
        y = np.asarray(y).reshape(-1)

        uniq = np.unique(y)
        if uniq.size != 2:
            raise ValueError("LogisticRegression supports only binary classification (2 unique labels).")

        y01 = (y == uniq.max()).astype(int)
        return y01
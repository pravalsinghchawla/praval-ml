import numpy as np 

class LinearRegression:
    def __init__(
            self,
            method: str = "normal", # "normal" or "gd"
            lr: float = 1e-2,
            epochs: int = 1000,
    ):
        self.method = method
        self.lr = lr
        self.epochs = epochs

        self.coef = None
        self.loss_history = []

    def fit(self, X, y):
        
        if self.method == "normal":
            self._fit_normal(X, y)
        elif self.method == "gd":
            self._fit_gd(X, y)
        else:
            raise ValueError("method must be 'normal' or 'gd'")

        return self
    
    def predict(self, X):
        return X @ self._weights

        
        
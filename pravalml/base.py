import numpy as np
from abc import ABC, abstractmethod

class BaseEstimator:
    def get_params(self):
        return {k: v for k, v in vars(self).items() if not k.endswith("_")}
    
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
    
    def fit(self, X, y=None):
        raise NotImplementedError("Subclasses must define their own fit method")
    
    def predict(self, X):
        raise NotImplementedError("Subclasses must define their own predict method")
    
class RegressorMixin:
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)

        if ss_tot == 0:
            return 0
        
        return 1 - ss_res / ss_tot
    
class ClassifierMixin:
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred==y)
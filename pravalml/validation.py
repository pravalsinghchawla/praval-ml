import numpy as np

def check_X(X, *, dtype=float, ensure_2d=True):
    X = np.asarray(X, dtype=dtype)

    if ensure_2d:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        
    if X.size == 0:
        raise ValueError("X must not be empty.")
    
    return X

def check_y(y, *, dtype=None):
    y = np.asarray(y if dtype is None else np.asarray(y, dtype=dtype))

    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    elif y.ndim != 1:
        raise ValueError("y must be a 1D array of shape(n_samples,).")
    
    if y.size == 0:
        raise ValueError("y must not be empty.")
    
    return y

def check_X_y(X, y, *, X_dtype=float, y_dtype=None):
    X = check_X(X, dtype=X_dtype, ensure_2d=True)
    y = check_y(y, dtype=y_dtype)

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples. Got {X.shape[0]} and {y.shape[0]}."
        )

    return X, y
import numpy as np
from abc import ABC, abstractmethod

from pravalml.base import BaseEstimator, RegressorMixin, ClassifierMixin
from pravalml.validation import check_X_y, check_X

class MLPBase(BaseEstimator, ABC):
    def __init__(
            self,
            hidden_layer_sizes=(100,),
            activation: str = "relu",
            lr: float = 1e-3,
            epochs: int = 200,
            batch_size = None,
            l2: float = 0.0,
            fit_intercept: bool = True,
            random_state = None,
            shuffle = True
    ):
        if isinstance(hidden_layer_sizes, int):
            hidden_layer_sizes = (hidden_layer_sizes,)
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.activation = activation
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.shuffle = shuffle

        self.loss_history_ = []
        self.n_features_in_ = None
        self.n_outputs_ = None
        self.weights_ = None  
        self.biases_ = None    

    def fit(self, X, y):
        X, y = check_X_y(X, y, X_dtype=float, y_dtype=float)
        
        y_2d = self._ensure_2d_targets(y)
        
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = y_2d.shape[1]

        rng = np.random.default_rng(self.random_state)
        self._init_params(rng)

        self.loss_history_ = []
        self._train(X, y_2d, rng)

        return self
    
    def predict(self, X):
        X = check_X(X, dtype=float)
        A_out, _cache = self._forward(X)
        return self._predict_from_output(A_out)
    
    def _train(self, X, y, rng):
        n_samples = X.shape[0]
        batch_size = self.batch_size or n_samples
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer or None.")

        for epoch in range(self.epochs):
            X_epoch, y_epoch = X, y
            if self.shuffle and batch_size < n_samples:
                idx = rng.permutation(n_samples)
                X_epoch, y_epoch = X[idx], y[idx]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                Xb = X_epoch[start:end]
                yb = y_epoch[start:end]

                A_out, cache = self._forward(Xb)
                loss, dZ_out = self._loss_and_grad(A_out, yb)

                if self.l2 > 0:
                    loss += 0.5 * self.l2 * sum(np.sum(W * W) for W in self.weights_)
                
                grads_W, grads_b = self._backward(dZ_out, cache)

                if self.l2 > 0:
                    grads_W = [gW + self.l2 * W for gW, W in zip(grads_W, self.weights_)]
                
                self._step(grads_W, grads_b)

                epoch_loss += loss
                n_batches += 1

            self.loss_history_.append(epoch_loss / max(n_batches, 1))

    def _step(self, grads_W, grads_b):
        for i in range(len(self.weights_)):
            self.weights_[i] -= self.lr * grads_W[i]
            if self.fit_intercept:
                self.biases_[i] -= self.lr * grads_b[i]

    def _forward(self, X):
        Zs, As = [], [X]
        A = X
        for i in range(len(self.hidden_layer_sizes)):
            W = self.weights_[i]
            b = self.biases_[i] if self.fit_intercept else 0.0

            Z = A @ W + b
            A = self._hidden_activation(Z)

            Zs.append(Z)
            As.append(A)

        W = self.weights_[-1]
        b = self.biases_[-1] if self.fit_intercept else 0.0

        Z_out = A @ W + b
        A_out = self._output_activation(Z_out)

        Zs.append(Z_out)
        As.append(A_out)

        cache = {"Zs": Zs, "As": As}
        return A_out, cache
    
    def _backward(self, dZ_out, cache):
        Zs = cache["Zs"]
        As = cache["As"]

        L = len(self.weights_)
        grads_W = [None] * L
        grads_b = [None] * L

        A_prev = As[-2]
        grads_W[-1] = (A_prev.T @ dZ_out) / A_prev.shape[0]
        grads_b[-1] = np.mean(dZ_out, axis=0, keepdims=True)

        dA_prev = dZ_out @ self.weights_[-1].T

        for layer in range(L - 2, -1, -1):
            Z = Zs[layer]
            dZ = dA_prev * self._hidden_activation_grad(Z)

            A_prev = As[layer]
            grads_W[layer] = (A_prev.T @ dZ) / A_prev.shape[0]
            grads_b[layer] = np.mean(dZ, axis=0, keepdims=True)

            if layer > 0:
                dA_prev = dZ @ self.weights_[layer].T
            else:
                pass

        return grads_W, grads_b

    def _init_params(self, rng):
        layer_sizes = [self.n_features_in_, *self.hidden_layer_sizes, self.n_outputs_]
        self.weights_ = []
        self.biases_ = [] if self.fit_intercept else None

        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            W = self._init_weight_matrix(rng, fan_in, fan_out)
            self.weights_.append(W)
            if self.fit_intercept:
                self.biases_.append(np.zeros((1, fan_out), dtype=float))

    def _init_weight_matrix(self, rng, fan_in, fan_out):
        if self.activation == "relu":
            scale = np.sqrt(2.0 / fan_in)
        else:
            scale = np.sqrt(1.0 / fan_in)
        return rng.normal(loc=0.0, scale=scale, size=(fan_in, fan_out)).astype(float)
    
    def _hidden_activation(self, Z):
        if self.activation == "relu":
            return np.maximum(0.0, Z)
        if self.activation == "tanh":
            return np.tanh(Z)
        if self.activation == "sigmoid":
            Z = np.clip(Z, -500, 500)
            return 1.0 / (1.0 + np.exp(-Z))
        raise ValueError(f"Unknown activation: {self.activation}")
    
    def _hidden_activation_grad(self, Z):
        if self.activation == "relu":
            return (Z > 0.0).astype(float)
        if self.activation == "tanh":
            A = np.tanh(Z)
            return 1.0 - A * A
        if self.activation == "sigmoid":
            Z = np.clip(Z, -500, 500)
            A = 1.0 / (1.0 + np.exp(-Z))
            return A * (1.0 - A)
        raise ValueError(f"Unknown activation: {self.activation}")

    def _ensure_2d_targets(self, y):
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            return y.reshape(-1, 1)
        return y
    
    @abstractmethod
    def _output_activation(self, Z_out):
        pass
    
    @abstractmethod
    def _loss_and_grad(self, A_out, y_true):
        pass

    @abstractmethod
    def _predict_from_output(self, A_out):
        pass


class MLPRegressor(MLPBase, RegressorMixin):
    def _output_activation(self, Z_out):
        return Z_out
    
    def _loss_and_grad(self, A_out, y_true):
        diff = A_out - y_true
        loss = 0.5 * np.mean(diff ** 2)
        dZ_out = diff
        return loss, dZ_out
    
    def _predict_from_output(self, A_out):
        if A_out.shape[1] == 1:
            return A_out.ravel()
        return A_out
    

class MLPClassifier(MLPBase, ClassifierMixin):
    def __init__(
            self, 
            hidden_layer_sizes=(100, ), 
            activation: str = "relu", 
            lr: float = 0.001, 
            epochs: int = 200, 
            batch_size=None, 
            l2: float = 0.0, 
            fit_intercept: bool = True, 
            random_state=None, 
            shuffle=True,
            threshold: float = 0.5
    ):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            l2=l2,
            fit_intercept=fit_intercept,
            random_state=random_state,
            shuffle=shuffle,
        )
        self.threshold = threshold

        self.classes_ = None

    def _ensure_2d_targets(self, y):
        y = np.asarray(y)

        if y.ndim == 2 and y.shape[1] > 1:
            self.classes_ = np.arange(y.shape[1])
            return y.astype(float)

        if y.ndim != 1:
            y = y.ravel()
        
        self.classes_ = np.unique(y)

        if self.classes_.size == 2:
            y01 = (y == self.classes_[1]).astype(float).reshape(-1, 1)
            return y01
        
        n = y.shape[0]
        k = self.classes_.size
        y_onehot = np.zeros((n, k), dtype=float)

        class_to_index = {c: i for i, c in enumerate(self.classes_)}
        idx = np.array([class_to_index[val] for val in y], dtype=int)
        y_onehot[np.arange(n), idx] = 1.0
        return y_onehot
    
    def _output_activation(self, Z_out):
        if self.n_outputs_ == 1:
            Z = np.clip(Z_out, -500, 500)
            return 1.0 / (1.0 + np.exp(-Z))

        Z = Z_out - np.max(Z_out, axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=1, keepdims=True)
    
    def _loss_and_grad(self, A_out, y_true):
        eps = 1e-12

        if self.n_outputs_ == 1:
            A = np.clip(A_out, eps, 1.0 - eps)
            loss = -np.mean(y_true * np.log(A) + (1.0 - y_true) * np.log(1.0 - A))

            dZ_out = (A_out - y_true)
            return loss, dZ_out

        A = np.clip(A_out, eps, 1.0 - eps)
        loss = -np.mean(np.sum(y_true * np.log(A), axis=1))

        dZ_out = (A_out - y_true)
        return loss, dZ_out
    
    def _predict_from_output(self, A_out):
        if self.n_outputs_ == 1:
            y01 = (A_out.ravel() >= self.threshold).astype(int)
            return self.classes_[y01]

        idx = np.argmax(A_out, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        X = check_X(X, dtype=float)  
        A_out, _ = self._forward(X)

        if self.n_outputs_ == 1:
            p1 = A_out.ravel()
            p0 = 1.0 - p1
            return np.column_stack([p0, p1])

        return A_out
import numpy as np
import pytest
from pravalml import LinearRegression

def test_fit_returns_self():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 6, 9, 12])
    
    model = LinearRegression()
    returned = model.fit(X, y)

    assert returned is model

def test_linear_regression_normal_fit_predict():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 6, 9, 12])

    model = LinearRegression(method="normal", epochs=10000)
    model.fit(X, y)

    preds = model.predict(X)

    assert np.allclose(preds, y, atol=1e-6)


def test_gd_converges_to_known_solution():
    rng = np.random.default_rng(67)
    X = rng.normal(size=(200, 1))
    y = 3.0 * X.squeeze() + 1.0

    model = LinearRegression(method="gd", lr=0.1, epochs=1000)
    model.fit(X, y)

    assert abs(model.coef_[0] - 3.0) < 0.05
    assert abs(model.intercept_ - 1.0) < 0.05


def test_gd_mechanism_reduces_loss():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(300, 1))
    y = 2.5 * X.squeeze() - 0.7

    model = LinearRegression(method="gd", lr=0.05, epochs=300)
    model.fit(X, y)

    assert hasattr(model, "loss_history_")
    assert len(model.loss_history_) == model.epochs
    assert np.isfinite(model.loss_history_).all()

    assert model.loss_history_[0] > model.loss_history_[-1]

    assert model.loss_history_[-1] < 0.1 * model.loss_history_[0]

def test_multiple_features():
    X = np.array([
        [1, 2],
        [2, 1],
        [3, 0],
        [0, 3],
    ])
    y = np.array([5, 5, 6, 6])

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == y.shape


def test_normal_and_gd_agree():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(300, 1))
    y = 2.5 * X.squeeze() + 0.7

    model_normal = LinearRegression(method="normal")
    model_gd = LinearRegression(method="gd", lr=0.1, epochs=1000)

    model_normal.fit(X, y)
    model_gd.fit(X, y)

    assert abs(model_normal.coef_[0] - model_gd.coef_[0]) < 0.05
    assert abs(model_normal.intercept_ - model_gd.intercept_) < 0.05


def test_invalid_method():
    X = np.array([[1], [3]])
    y = np.array([6, 18])

    model = LinearRegression(method="not a valid method")

    with pytest.raises(ValueError):
        model.fit(X, y)

def test_fit_without_intercept():
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)

    assert abs(model.intercept_) < 1e-8
    assert abs(model.coef_[0] - 2) < 1e-6


def test_predict_before_fit_raises():
    model = LinearRegression()
    X = np.array([[1], [2]])

    with pytest.raises(Exception):
        model.predict(X)


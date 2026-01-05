import numpy as np
import pytest

from pravalml import RidgeRegression, LinearRegression

def make_data(seed=0, n=200):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    true_w = np.array([0.5, -2.0])
    y = 10.0 + X @ true_w + rng.normal(scale=0.1, size=n)
    return X, y

def test_predict_before_fit_raises():
    X, _ = make_data()
    model = RidgeRegression()
    with pytest.raises(ValueError):
        model.predict(X)

def test_invalid_method_raises():
    X, y = make_data()
    model = RidgeRegression(method="something")
    with pytest.raises(ValueError):
        model.fit(X, y)

def test_alpha_zero_matches_ols_normal():
    X, y = make_data()
    ols = LinearRegression(method="normal", fit_intercept=True).fit(X, y)
    ridge0 = RidgeRegression(method="normal", alpha=0.0, fit_intercept=True).fit(X, y)

    np.testing.assert_allclose(ridge0.intercept_, ols.intercept_, atol=1e-8)
    np.testing.assert_allclose(ridge0.coef_, ols.coef_, atol=1e-8)

def test_intercept_not_regularized():
    X, y = make_data()
    model = RidgeRegression(method="normal", alpha=1e5, fit_intercept=True).fit(X, y)

    assert np.linalg.norm(model.coef_) < 0.1
    assert abs(model.intercept_ - 10.0) < 0.5

def test_gd_close_to_normal():
    X, y = make_data(seed=1)
    normal = RidgeRegression(method="normal", alpha=1.0, fit_intercept=True).fit(X, y)
    gd = RidgeRegression(method="gd", alpha=1.0, fit_intercept=True, lr=1e-2, epochs=5000).fit(X, y)

    np.testing.assert_allclose(gd.intercept_, normal.intercept_, atol=1e-2)
    np.testing.assert_allclose(gd.coef_, normal.coef_, atol=1e-2)


def test_y_column_vector_ok():
    X, y = make_data(seed=3)
    y_col = y.reshape(-1, 1)

    m1 = RidgeRegression(method="normal", alpha=1.0, fit_intercept=True).fit(X, y)
    m2 = RidgeRegression(method="normal", alpha=1.0, fit_intercept=True).fit(X, y_col)

    np.testing.assert_allclose(m1.intercept_, m2.intercept_, atol=1e-8)
    np.testing.assert_allclose(m1.coef_, m2.coef_, atol=1e-8)


def test_larger_alpha_shrinks_coeff_norm():
    X, y = make_data(seed=4)

    small = RidgeRegression(method="normal", alpha=0.1, fit_intercept=True).fit(X, y)
    big = RidgeRegression(method="normal", alpha=100.0, fit_intercept=True).fit(X, y)

    assert np.linalg.norm(big.coef_) < np.linalg.norm(small.coef_)
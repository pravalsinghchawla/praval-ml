import numpy as np
from pravalml.models.logistic_regression import LogisticRegression


def test_logistic_regression_linearly_separable():
    X = np.array([[-2], [-1], [-0.5], [0.5], [1], [2]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegression(lr=0.5, epochs=2000)
    model.fit(X, y)

    preds = model.predict(X)
    assert (preds == y).mean() >= 0.95


def test_predict_proba_shape():
    X = np.array([[-1], [1], [2]])
    y = np.array([0, 1, 1])

    model = LogisticRegression(lr=0.5, epochs=500)
    model.fit(X, y)

    proba = model.predict_proba(X)
    assert proba.shape == (3, 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)

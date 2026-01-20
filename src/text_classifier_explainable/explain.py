"""Explain model prediction: build background for explainer, explainer itself and explain the model."""

import shap
import numpy as np
from scipy import sparse
from sklearn.pipeline import Pipeline
from typing import Sequence

masker = shap.maskers.Independent


def build_shap_background(model: Pipeline, X_train: Sequence[str], max_samples: int = 500) -> sparse.spmatrix:
    """
    Build background data for SHAP explainer.
    """

    tfidf = model.named_steps['tfidf']
    rng = np.random.default_rng(42)
    n_samples = min(len(X_train), max_samples)
    idx = rng.choice(len(X_train), size=n_samples, replace=False)
    X_bg = tfidf.transform([X_train[i] for i in idx])

    return X_bg


def build_explainer(model: Pipeline, X_bg: sparse.spmatrix, masker) -> shap.LinearExplainer:
    """
    Build a SHAP LinearExplainer for a trained logistic regression model.

    Args:
        model: Fitted sklearn Pipeline containing 'lr' classifier.
        X_bg: Sparse matrix representing background data.

    Returns:
        shap.LinearExplainer
    """

    lr = model.named_steps['lr']

    explainer = shap.LinearExplainer(model=lr, masker=masker, data=X_bg)
    return explainer


def explain(explainer: shap.LinearExplainer, model: Pipeline, X: sparse.spmatrix) -> shap.Explanation:
    """
    Compute SHAP values for given TF-IDF features and attach feature names.

    Args:
        explainer: Fitted shap.LinearExplainer
        model: sklearn Pipeline (lr + tfidf)
        X: Sparse matrix of TF-IDF features to explain

    Returns:
        shap.Explanation object
    """

    if X is None or X.shape[0] == 0:
        raise ValueError("Input X is empty")

    tfidf = model.named_steps['tfidf']
    feature_names = list(tfidf.get_feature_names_out())

    shap_values = explainer(X)
    shap_values.feature_names = feature_names

    return shap_values


    
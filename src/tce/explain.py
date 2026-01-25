"""Model explanation utilities for text classification models (SHAP)."""

import shap
import logging
import numpy as np
from scipy import sparse
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def build_explainer(model: Pipeline, X_background: sparse.spmatrix) -> shap.LinearExplainer:
    """
    Build a SHAP LinearExplainer for a trained logistic regression model.

    Args:
        model (Pipeline):
            A fitted sklearn Pipeline containing:
            - 'tfidf': TfidfVectorizer
            - 'lr': LogisticRegression
        X_background (sparse.spmatrix):
            Background feature matrix used to estimate expected values.
            Must be a TF-IDF-transformed sparse matrix.

    Returns:
        A fitted shap.LinearExplainer instance.
    """

    logger.info('Building SHAP LinearExplainer')

    lr = model.named_steps['lr']
    masker = shap.maskers.Independent(X_background)
    explainer = shap.LinearExplainer(lr, masker=masker)

    logger.info('SHAP explainer built with background shape %s', X_background.shape)

    return explainer


def explain(explainer: shap.LinearExplainer, model: Pipeline, text: str, top_k: int = 10) -> list[tuple[str, float, float]]:
    """
    Explain a single text prediction using SHAP values.

    The function computes SHAP values for TF-IDF features corresponding
    to the input text and returns the most important features ranked by
    absolute contribution.

    Args:
        explainer:
            A fitted shap.LinearExplainer.
        model:
            A fitted sklearn Pipeline with a 'tfidf' step.
        text (str):
            Raw input text to be explained.
        top_k (int):
            Number of features with the largest absolute SHAP values to return.

    Returns:
        A list of tuples (feature_name, shap_value, abs_shap_value), sorted by descending absolute SHAP value.

    Raises:
        TypeError:
            If `text` is not a string.
        ValueError:
            If `top_k` is not positive.
    """
    if not isinstance(text, str):
        raise TypeError('`text` must be a string')

    if top_k <= 0:
        raise ValueError('`top_k` must be a positive integer')

    logger.info('Explaining text with length %d words', len(text.split()))

    tfidf = model.named_steps['tfidf']
    feature_names = tfidf.get_feature_names_out()

    X = tfidf.transform([text])

    shap_values = explainer(X)
    values = shap_values.values[0]

    logger.debug('Computed SHAP values for %d features', values.shape[0])

    top_k = min(top_k, values.shape[0])
    top_indices = np.argsort(np.abs(values))[::-1][:top_k]

    explanation = [
        (
            feature_names[i],
            float(values[i]),
            float(abs(values[i])),
        )
        for i in top_indices
    ]

    logger.debug('Returning top %d SHAP features', len(explanation))

    return explanation

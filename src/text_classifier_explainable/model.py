"""Build and train text classification model pipeline and use it for inference."""

import numpy as np
from typing import Dict, Any, Sequence, Union
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def build_model(params: Dict[str, Dict[str, Any]]) -> Pipeline:
    """
    Build text classification model pipeline.

    Pipeline consists of:
    - TF-IDF vectorizer
    - Logistic Regression classifier
    
    Args:
        params: Dictionary with model parametrs.
        Expected structure:
            {
                "tfidf": {...},
                "lr": {...}
            }
    Returns: 
        sklearn.pipeline.Pipeline: Configured model pipeline.
    """

    required_keys = {'tfidf', 'lr'}
    missing = required_keys - params.keys()
    if missing:
        raise KeyError(f"Missing required params: {missing}")
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer(**params['tfidf'])),
        ('lr', LogisticRegression(**params['lr']))
    ])

    return model


def _validate_labels(y: Sequence) -> None:
    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError("y must be a 1D array")

    if np.isnan(y).any():
        raise ValueError("y contains NaN values")

    if not np.issubdtype(y.dtype, np.integer):
        raise TypeError(
            f"Labels must be integers, got dtype={y.dtype}"
        )


def train_model(model: Pipeline, X_train: Sequence[str], y_train: Sequence[int]) -> Pipeline:
    """
    Train a sklearn pipeline on training data.

    Args:
        model: Unfitted sklearn pipeline.
        X_train: Training texts.
        y_train: Training labels.
    
    Returns:
        Fitted Pipeline.

    Raises:
        ValueError: If `y_train` has invalid shape or contains NaN values.
        TypeError: If `y_train` is not integer-encoded.
    """
    _validate_labels(y_train)
    return model.fit(X_train, y_train)


def predict(model: Pipeline, texts: Union[str, Sequence[str]]) -> tuple[list[int], list[list[float]]]:
    """
    Predict labels and probabilities for one or more texts.

    Args:
        model: Fitted sklearn pipeline.
        texts: Single text string or a sequence of text strings.

    Returns:
        Tuple containing:
            - predictions: List of predicted labels.
            - probabilities: List of lists with predicted class probabilities.
              Each inner list corresponds to a sample and contains probabilities
              for each class in the order returned by `model.classes_`.
    """

    if isinstance(texts, str):
        texts = [texts]
    
    predictions = model.predict(texts).tolist()
    probabilities = model.predict_proba(texts).tolist()

    return predictions, probabilities

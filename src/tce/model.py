"""Build, train, save/load, and predict using a text classification model pipeline."""

import numpy as np
from typing import Any, Sequence, Callable
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import joblib
from pathlib import Path
import shap
import logging
from .data import clean_text
from scipy import sparse

PREPROCESSORS: dict[str, Callable[[str], str]] = {
    'clean_text': clean_text
}

logger = logging.getLogger(__name__)


def _process_pipeline_params(params: dict) -> dict:
    """
    Normalize model parameters: ensure dictionaries, convert lists to tuples,
    and map 'preprocessor' string to callable function.
    """

    if not isinstance(params, dict):
        raise TypeError('Expected params to be a dictionary.')

    for step, values in params.items():
        if not isinstance(values, dict):
            raise TypeError(f'Parameters for step "{step}" must be a dictionary.')
        for param, value in values.items():
            if isinstance(value, list):
                params[step][param] = tuple(value)
            if param == 'preprocessor':
                if isinstance(value, str) and value in PREPROCESSORS:
                    params[step][param] = PREPROCESSORS[value]
                elif callable(value):
                    params[step][param] = value
                else:
                    raise ValueError(f'Invalid preprocessor value: {value}')

    logger.info('Pipeline parameters processed successfully')

    return params


def build_model(params: dict[str, dict[str, Any]]) -> Pipeline:
    """
    Build text classification model pipeline.

    Pipeline consists of:
    - TF-IDF vectorizer
    - Logistic Regression classifier
    
    Args:
        params (dict): Dictionary with model parametrs.
        Expected structure:
            {
                "tfidf": {...},
                "lr": {...}
            }
    Returns: 
        sklearn.pipeline.Pipeline: Configured model pipeline.
    Raises:
        TypeError: If some trouble with model parameters structure appears.
        ValueError: Raises problems with model parameters values.
        KeyError: If required steps are missed.
    """
    params = _process_pipeline_params(params)

    required_keys = {'tfidf', 'lr'}
    missing = required_keys - params.keys()
    if missing:
        raise KeyError(f'Missing required steps: {missing}')
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer(**params['tfidf'])),
        ('lr', LogisticRegression(**params['lr']))
    ])

    logger.info('Pipeline built successfully with steps: %s', list(model.named_steps.keys()))

    return model


def train_model(model: Pipeline, X_train: Sequence[str], y_train: Sequence[int] | np.ndarray) -> Pipeline:
    """
    Train a sklearn pipeline on training data.

    Args:
        model (Pipeline): Unfitted sklearn pipeline.
        X_train (Sequence[str]): Training (uncleaned) texts.
        y_train (Sequence[int] | np.ndarray): Training labels.
    
    Returns:
        Fitted Pipeline.
    """

    if not X_train or not y_train:
        raise ValueError('Training data X_train and y_train must not be empty')
    if len(X_train) != len(y_train):
        raise ValueError('X_train and y_train must have the same length')

    logger.info('Training model on %d samples', len(X_train))
    model.fit(X_train, y_train)
    logger.info('Model training complete')

    return model


def save_model(model: Pipeline | shap.LinearExplainer, path: Path) -> None:
    """
    Save model pipeline (or shap.LinearExplainer) to disk.

    Args:
        model (Pipeline | shap.LinearExplainer): Fitted model sklearn pipeline (or built shap linear explainer).
        path (Path): Destination path.
    
    Raises:
        Exception:
                Re-raises any exception that occurs during serialization.
                Errors are logged before being raised.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info('Saving model to %s', path)
    try:
        joblib.dump(model, path)
        logger.info('Model saved')
    except Exception:
        logger.exception('Failed to save model to %s', path)
        raise


def load_model(path: Path) -> Pipeline | shap.LinearExplainer:
    """
    Load a saved sklearn Pipeline or SHAP LinearExplainer from disk.

    Args:
        path (Path): Path to the serialized model file.

    Returns:
        Loaded sklearn Pipeline or shap.LinearExplainer instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception:
            Re-raises any exception that occurs during deserialization.
            Errors are logged before being raised.
    """

    logger.info('Loading model from %s', path)

    if not path.exists():
        logger.error('Model file not found: %s', path)
        raise FileNotFoundError(f'Model file not found: {path}')

    try:
        model = joblib.load(path)
        logger.info('Model successfully loaded from %s', path)
        return model

    except Exception:
        logger.exception('Failed to load model from %s', path)
        raise


def is_model_fitted(model: Pipeline) -> bool:
    try:
        check_is_fitted(model)
        return True
    except NotFittedError:
        return False


def predict(model: Pipeline, texts: str | Sequence[str]) -> tuple[list[int], list[list[float]], sparse.csr_matrix]:
    """
    Predict labels and probabilities for one or more texts.

    Args:
        model (Pipeline): Fitted sklearn pipeline.
        texts (str | Sequence[str]): Single text string or a sequence of text strings.

    Returns:
        Tuple containing:
            - predictions (list[int]): List of predicted labels.
            - probabilities (list[list[float]]): List of lists with predicted class probabilities.
              Each inner list corresponds to a sample and contains probabilities
              for each class in the order returned by `model.classes_`.
            - TF-IDF vectorized texts (sparse.csr_matrix).
    Raises:
        ValueError:
            If no texts are provided for prediction.
        TypeError:
            If `texts` is not a string or a sequence of strings.
        RuntimeError:
            If the provided model pipeline is not fitted.
    """

    if not texts:
        raise ValueError('No texts provided for prediction')
    if not isinstance(texts, (str, Sequence)):
        raise TypeError('texts must be a string or a sequence of strings')
    if isinstance(texts, str):
        texts = [texts]
    if not is_model_fitted(model):
        raise RuntimeError('Model is not fitted yet')
    
    tfidf = model.named_steps['tfidf']

    predictions = model.predict(texts).tolist()
    probabilities = model.predict_proba(texts).tolist()
    vectorized = tfidf.transform(texts)

    return predictions, probabilities, vectorized

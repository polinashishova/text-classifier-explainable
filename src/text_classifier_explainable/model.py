"""Build and train text classification model pipeline and use it for inference."""

import numpy as np
from typing import Dict, Any, Sequence, Union
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path
import shap
import logging
from .data import clean_text
from scipy import sparse

logger = logging.getLogger(__name__)


def _validate_pipeline_params(params: Dict) -> Dict:
    if not isinstance(params, dict):
        try: 
            params = dict(params)
        except Exception:
            raise Exception('Expected model parametrs structure should be dictionary')
    for key, value in params.items():
        for par, v in value.items():
            if isinstance(v, list):
                params[key][par] = tuple(v)
            if par == 'preprocessor':
                params[key][par] = eval(v)
    
    return params



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
    params = _validate_pipeline_params(params)

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
        X_train: Training (uncleand) texts.
        y_train: Training labels.
    
    Returns:
        Fitted Pipeline.

    Raises:
        ValueError: If `y_train` has invalid shape or contains NaN values.
        TypeError: If `y_train` is not integer-encoded.
    """
    #_validate_labels(y_train)
    return model.fit(X_train, y_train)


def save_model(model: Union[Pipeline, shap.LinearExplainer], path: Path) -> None:
    """
    Save model pipeline (or shap.LinearExplainer) to disk.

    Args:
        model: Fitted model sklearn pipeline (or built shap linear explainer).
        path: Destination path.
    
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


def load_model(path: Path) -> Union[Pipeline, shap.LinearExplainer]:
    """
    Load a saved sklearn Pipeline or SHAP LinearExplainer from disk.

    Args:
        path: Path to the serialized model file.

    Returns:
        Loaded sklearn Pipeline or shap.LinearExplainer instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception:
            Re-raises any exception that occurs during deserialization.
            Errors are logged before being raised.
    """
    logger.info("Loading model from %s", path)

    if not path.exists():
        logger.error("Model file not found: %s", path)
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        model = joblib.load(path)
        logger.info("Model successfully loaded from %s", path)
        return model

    except Exception:
        logger.exception("Failed to load model from %s", path)
        raise


def predict(model: Pipeline, texts: Union[str, Sequence[str]]) -> tuple[list[int], list[list[float]], sparse.spmatrix]:
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
            - TF-IDF vectorized texts.
    """

    if isinstance(texts, str):
        texts = [texts]
    
    tfidf = model.named_steps['tfidf']

    predictions = model.predict(texts).tolist()
    probabilities = model.predict_proba(texts).tolist()
    vectorized = tfidf.transform(texts)

    return predictions, probabilities, vectorized

import numpy as np
from typing import Sequence
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

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
    

def compute_metrics(model: Pipeline, X: Sequence[str], y_true: Sequence[int]):
    """
    Compute classification quality metrics for a fitted model.

    The function evaluates a trained sklearn Pipeline on given data and
    calculates standard binary classification metrics.

    Metrics computed:
        - Accuracy
        - Precision
        - Recall
        - F1-score
        - ROC AUC
        - Average Precision

    Args:
        model: Fitted sklearn Pipeline with a classifier supporting
            `predict` and `predict_proba`.
        X: Sequence of input texts.
        y_true: True labels (binary, integer-encoded).

    Returns:
        Dict[str, float | str]: Dictionary containing computed metrics.
        Includes:
            - 'model': Name of the classifier used in the pipeline.
            - Metric values as floats.

    Raises:
        ValueError: If `y_true` has invalid shape or contains NaN values.
        TypeError: If `y_true` is not integer-encoded.
    """

    _validate_labels(y_true)
    
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]
    
    acs = accuracy_score(y_true, y_pred)
    prs = precision_score(y_true, y_pred)
    res = recall_score(y_true, y_pred)
    f1s = f1_score(y_true, y_pred)
    ras = roc_auc_score(y_true, y_score)
    aps = average_precision_score(y_true, y_score)

    metrics = {
        'model': list(model.named_steps.keys())[-1],
        'accuracy_score': acs, 
        'precision_score': prs, 
        'recall_score': res, 
        'f1_score': f1s, 
        'roc_auc_score': ras, 
        'average_precision_score': aps
    }

    return metrics
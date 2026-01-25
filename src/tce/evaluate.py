"""Evaluate binary classification models."""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)
from pathlib import Path
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def _check_binary(y: ArrayLike, name: str) -> np.ndarray:
    """
    Check that input is binary (0/1) and one-dimensional.

    Args:
        y (ArrayLike): Input array of labels.
        name (str): Name of the variable for error messages.

    Returns:
        np.ndarray: Checked one-dimensional array.
    """
    y_arr = np.asarray(y)

    if y_arr.ndim != 1:
        raise ValueError(f'{name} must be 1-dimensional, got shape {y_arr.shape}')

    if not np.all(np.isfinite(y_arr)):
        raise ValueError(f'{name} contains NaN or infinite values')

    unique_vals = np.unique(y_arr)

    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(f'{name} must be binary (0/1), got values {unique_vals}')

    return y_arr.astype(int)


def compute_metrics(y_true: ArrayLike, y_pred: ArrayLike, y_score: ArrayLike) -> dict[str, float]:
    """
    Compute binary classification metrics for a fitted model.

    Metrics computed:
        - Accuracy
        - Precision
        - Recall
        - F1-score
        - ROC AUC
        - Average Precision

    Args:
        y_true (ArrayLike): True binary labels.
        y_pred (ArrayLike): Predicted binary labels.
        y_score (ArrayLike): Predicted positive class scores (probabilities or decision function).

    Returns:
        dict[str, float]: Dictionary with all metrics as floats.
    
    Raises:
        ValueError: If `y_true`, `y_pred` or `y_score` is not one-dimensional;
                    if `y_true` or `y_pred` in not binary;
                    if `y_score` contains NaN or infinite values.
    """

    y_true = _check_binary(y_true, 'y_true')
    y_pred = _check_binary(y_pred, 'y_pred')
    y_score = np.asarray(y_score)

    if y_score.ndim != 1 or len(y_score) != len(y_true):
        raise ValueError(
            f'y_score must be 1-dimensional and same length as y_true ({len(y_true)}), '
            f'got shape {y_score.shape}'
        )

    if not np.all(np.isfinite(y_score)):
        raise ValueError('y_score contains NaN or infinite values')
    
    if len(np.unique(y_true)) < 2:
        raise ValueError('y_true must contain both classes to compute ROC AUC')

    metrics = {
        'accuracy_score': float(accuracy_score(y_true, y_pred)),
        'precision_score': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall_score': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc_score': float(roc_auc_score(y_true, y_score)),
        'average_precision_score': float(average_precision_score(y_true, y_score))
    }

    logger.info('Binary classification metrics are computed')
    return metrics


def plot_curve(y_true: ArrayLike, y_score: ArrayLike, path: Path, curve_type: str = 'roc') -> None:
    """
    Plot and save ROC or Precision-Recall (PR) curve.

    Args:
        y_true (ArrayLike): True binary labels.
        y_score (ArrayLike): Predicted positive class scores (probabilities or decision function).
        path (Path): Path to save the plot.
        curve_type (str): Type of curve to plot: 'roc' or 'pr'.

    Raises:
        ValueError: If curve_type is not 'roc' or 'pr' or something is wrong with `y_true` or `y_score`.
    """

    y_true = _check_binary(y_true, 'y_true')
    y_score = np.asarray(y_score)
    if y_score.ndim != 1 or len(y_score) != len(y_true):
        raise ValueError(f'y_score must be 1-dimensional and same length as y_true ({len(y_true)}), got shape {y_score.shape}')
    
    if not np.all(np.isfinite(y_score)):
        raise ValueError('y_score contains NaN or infinite values')
    
    if len(np.unique(y_true)) < 2:
        raise ValueError('y_true must contain both classes to plot ROC/PR curve')
    
    path.parent.mkdir(parents=True, exist_ok=True)
    curve_type = curve_type.lower()

    fig, ax = plt.subplots()
    
    if curve_type == 'roc':
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        
    elif curve_type == 'pr':
        precisions, recalls, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recalls, precisions, label=f'AP = {ap:.3f}')
        ax.set_title('Precision-Recall (PR) Curve')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')

    else:
        raise ValueError(f'Invalid curve_type {curve_type}, must be "roc" or "pr"')

    ax.grid(True)
    ax.legend()
    fig.savefig(path)
    plt.close(fig)
    logger.info('%s curve saved to %s', curve_type.upper(), path)

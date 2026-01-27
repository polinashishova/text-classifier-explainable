"""
evaluate_model.py

Script to evaluate a trained machine learning model pipeline (TF-IDF + LR) on the given dataset.
Computes metrics (accuracy, precision, recall, F1, ROC AUC, AP) and generates ROC and PR curves.

Key steps:
- Load trained model and test data
- Predict labels and probabilities
- Compute evaluation metrics and plot figures
- Save metrics and plots to artifacts directory

Usage:
    python scripts/evaluate_model.py
"""

from pathlib import Path
from tce.utils import save_json, load_json, setup_logging
from tce.model import predict, load_model
from tce.evaluate import compute_metrics, plot_curve
from tce.data import load_data

logger = setup_logging()

def main() -> None:
    paths_path = Path('config/paths.json')
    paths = load_json(paths_path)

    model_path = Path(paths['models_dir']) / paths['model']
    model = load_model(model_path)

    y_test_path = Path(paths['data_processed_dir']) / paths['test_labels']
    y_test = load_data(y_test_path)

    X_test_path = Path(paths['data_raw_dir']) / paths['raw_test']
    X_test = load_data(X_test_path)

    y_pred, y_score, _ = predict(model, X_test)
    y_score_pos = [x[1] for x in y_score]

    metrics = compute_metrics(y_test, y_pred, y_score_pos)
    artifacts_dir = Path(paths['artifacts_dir'])
    evaluation_report_dir = artifacts_dir / paths['evaluation_report_dir']
    metrics_path = evaluation_report_dir / paths['model_metrics']
    save_json(metrics, metrics_path)

    figures_dir = Path(evaluation_report_dir / paths['figures_dir'])

    for curve in ['roc', 'pr']:
        plot_curve(y_test, y_score_pos, figures_dir / paths[f'{curve}_curve_plot'], curve_type=curve)



if __name__ == '__main__':
    main()
"""
train_model.py

Script to train a machine learning model using TF-IDF features and a Logistic Regression classifier pipeline. 
Saves the trained model and transformed feature matrices.

Key steps:
- Load configuration and paths
- Load training raw and label data
- Build and train model pipeline
- Save trained model and feature matrices

Usage:
    python scripts/train_model.py
"""

from tce.model import build_model, train_model, save_model
from tce.data import load_data, save_data
from tce.utils import setup_logging, load_json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Sequence

logger = setup_logging()


def save_transformed(tfidf: TfidfVectorizer, raw_data: Sequence[str], transformed_path: Path) -> None:
    """
    Save transformed TF-IDF features.
    Skips transformation if output files already exist.

    Args:
        tfidf: Fitted TfidfVectorizer.
        raw_data: Sequence of raw texts needed to be transformed.
        transformed_path (Path): Path for saving transformed features.
    """
    if transformed_path.exists():
        logger.info('%s already exists. Skipping saving', transformed_path.name)
        return
    
    transformed = tfidf.transform(raw_data)
    save_data(transformed_path, transformed)


def main() -> None:

    paths_cfg_path = Path('config/paths.json')
    paths = load_json(paths_cfg_path)

    model_cfg_path = Path('config/model.json')
    model_cfg = load_json(model_cfg_path)
    model_path = Path(paths['models_dir']) / paths['model']

    raw_dir = Path(paths['data_raw_dir'])
    X_train = load_data(raw_dir / paths['raw_train'])
    X_test = load_data(raw_dir / paths['raw_test'])

    y_train = load_data(Path(paths['data_processed_dir']) / paths['train_labels'])

    model = build_model(model_cfg)
    train_model(model, X_train, y_train)
    save_model(model, model_path)

    X_train_features_path = Path(paths['data_features_dir']) / paths['features_train']
    X_test_features_path = Path(paths['data_features_dir']) / paths['features_test']

    tfidf = model.named_steps['tfidf']

    save_transformed(tfidf, X_train, X_train_features_path)
    save_transformed(tfidf, X_test, X_test_features_path)


if __name__ == '__main__':
    main()
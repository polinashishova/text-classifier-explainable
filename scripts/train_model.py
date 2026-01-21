from src.text_classifier_explainable.model import build_model, train_model, save_model
from src.text_classifier_explainable.data import load_json, load_data, clean_text, save_data
from src.text_classifier_explainable.utils import setup_logging
from pathlib import Path
import logging

setup_logging()

logger = logging.getLogger(__name__)

paths_cfg_path = Path('config/paths.json')
paths = load_json(paths_cfg_path)

model_cfg_path = Path('config/model.json')
model_cfg = load_json(model_cfg_path)
model_path = Path(f'{paths['models_dir']}/{paths['model']}')

X_train = load_data(Path(f'{paths['data_raw_dir']}/{paths['raw_train']}'))
X_test = load_data(Path(f'{paths['data_raw_dir']}/{paths['raw_test']}'))
y_train_raw = load_data(Path(f'{paths['data_processed_dir']}/{paths['train_labels']}'))
y_train = [int(i) for i in y_train_raw]

model = build_model(model_cfg)

train_model(model, X_train, y_train)

save_model(model, model_path)

X_train_features_path = Path(f'{paths['data_features_dir']}/{paths['features_train']}')
X_test_features_path = Path(f'{paths['data_features_dir']}/{paths['features_test']}')

tfidf = model.named_steps['tfidf']

if not X_train_features_path.exists():
    X_train_features = tfidf.transform(X_train)
    save_data(X_train_features_path, X_train_features)
else:
    logger.info('Train features file already exists.')

if not X_test_features_path.exists():
    X_test_features = tfidf.transform(X_test)
    save_data(X_test_features_path, X_test_features)
else:
    logger.info('Test features file already exists.')
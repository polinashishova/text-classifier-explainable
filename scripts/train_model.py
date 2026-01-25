from tce.model import build_model, train_model, save_model
from tce.data import load_data, clean_text, save_data
from tce.utils import setup_logging, load_json
from pathlib import Path

logger = setup_logging()

paths_cfg_path = Path('config/paths.json')
paths = load_json(paths_cfg_path)

model_cfg_path = Path('config/model.json')
model_cfg = load_json(model_cfg_path)
model_path = Path(f'{paths['models_dir']}/{paths['model']}')

X_train = load_data(Path(f'{paths['data_raw_dir']}/{paths['raw_train']}'))
X_test = load_data(Path(f'{paths['data_raw_dir']}/{paths['raw_test']}'))
y_train = load_data(Path(f'{paths['data_processed_dir']}/{paths['train_labels']}'))

model = build_model(model_cfg)

train_model(model, X_train, y_train)

save_model(model, model_path)

X_train_features_path = Path(f'{paths['data_features_dir']}/{paths['features_train']}')
X_test_features_path = Path(f'{paths['data_features_dir']}/{paths['features_test']}')

tfidf = model.named_steps['tfidf']

if not X_train_features_path.exists():
    X_train_features = tfidf.transform(X_train)
    save_data(X_train_features_path, X_train_features)

if not X_test_features_path.exists():
    X_test_features = tfidf.transform(X_test)
    save_data(X_test_features_path, X_test_features)
from src.text_classifier_explainable.utils import setup_logging
from src.text_classifier_explainable.explain import build_explainer
from src.text_classifier_explainable.model import save_model, load_model
from src.text_classifier_explainable.data import load_json, load_data
from pathlib import Path

setup_logging()

paths_cfg_path = Path('config/paths.json')
paths = load_json(paths_cfg_path)

model_path = Path(f'{paths['models_dir']}/{paths['model']}')
explainer_path = Path(f'{paths['models_dir']}/{paths['explainer']}')

X_train = load_data(Path(f'{paths['data_features_dir']}/{paths['features_train']}'))

model = load_model(model_path)

explainer = build_explainer(model, X_train)

save_model(explainer_path, explainer_path)

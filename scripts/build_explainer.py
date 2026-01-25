from tce.utils import setup_logging
from tce.explain import build_explainer
from tce.model import save_model, load_model
from tce.data import load_data
from tce.utils import load_json
from pathlib import Path

logger = setup_logging()

paths_cfg_path = Path('config/paths.json')
paths = load_json(paths_cfg_path)

model_path = Path(f'{paths['models_dir']}/{paths['model']}')
explainer_path = Path(f'{paths['models_dir']}/{paths['explainer']}')

X_train = load_data(Path(f'{paths['data_features_dir']}/{paths['features_train']}'))

model = load_model(model_path)

explainer = build_explainer(model, X_train)

save_model(explainer, explainer_path)

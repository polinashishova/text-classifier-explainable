from pathlib import Path
from tce.utils import save_json, load_json, setup_logging
from tce.model import predict, load_model
from tce.evaluate import compute_metrics, plot_curve
from tce.data import load_data

logger = setup_logging()

paths_path = Path('config/paths.json')
paths = load_json(paths_path)

model_path = Path(f'{paths['models_dir']}/{paths['model']}')
model = load_model(model_path)

y_test_path = Path(f'{paths['data_processed_dir']}/{paths['test_labels']}')
y_test = load_data(y_test_path)

X_test_path = Path(f'{paths['data_raw_dir']}/{paths['raw_test']}')
X_test = load_data(X_test_path)

y_pred, y_score, _ = predict(model, X_test)
y_score_pos = [x[1] for x in y_score]

metrics = compute_metrics(y_test, y_pred, y_score_pos)
artifacts_dir = Path(paths['artifacts_dir'])
evaluation_report_dir = Path(artifacts_dir / paths['evaluation_report_dir'])
metrics_path = Path(evaluation_report_dir / paths['model_metrics'])
save_json(metrics, metrics_path)

figures_dir = Path(evaluation_report_dir / paths['figures_dir'])

roc_curve_path = Path(figures_dir / paths['roc_curve_plot'])
plot_curve(y_test, y_score_pos, roc_curve_path, curve_type='roc')

pr_curve_path = Path(figures_dir / paths['pr_curve_plot'])
plot_curve(y_test, y_score_pos, pr_curve_path, curve_type='pr')
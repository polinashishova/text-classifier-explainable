from pathlib import Path
import streamlit as st
from tce.model import load_model, predict
from tce.utils import setup_logging, load_json
from tce.explain import explain

st.set_page_config(layout='centered')

logger = setup_logging()

CLASSES = {
    0: 'negative', 
    1: 'positive'
    }

paths_cfg_path = Path('config/paths.json')
paths = load_json(paths_cfg_path)

model_path = Path(paths['models_dir']) / paths['model']
explainer_path = Path(paths['models_dir']) / paths['explainer']


@st.cache_resource
def load_models(model_path: Path, explainer_path: Path):
    """Load ML model and explainer once and cache them."""
    model = load_model(model_path)
    explainer = load_model(explainer_path)
    return model, explainer


model, explainer = load_models(model_path, explainer_path)

st.title('Text classification with explanation')

st.sidebar.header("Settings")

top_k = st.sidebar.slider(
    "Number of influential words",
    min_value=1,
    max_value=10,
    value=5
)

text = st.text_area(
    '**Enter text**',
    height=150,
    placeholder='Write text for classification...'
)

if st.button(label='Classify'):
    if not text.strip():
        st.warning('Enter text')
    else:
        try:
            label, proba, _ = predict(model, text)
            shap_words = explain(explainer, model, text, top_k=top_k)
        except Exception as e:
            st.error("Prediction failed. See logs.")
            logger.exception(e)
            st.stop()

        name = CLASSES.get(label[0], 'unknown')

        st.subheader('Result:')
        if name == 'negative':
            st.error(f'Class: {name}')
        elif name == 'positive':
            st.success(f'Class: {name}')

        st.write(f'Model confidence: {max(proba[0]):.2f}')

        st.subheader('The most influential words and phrases')

        for item in shap_words:
            word, value = item[0], item[1]
            if value > 0:
                st.markdown(f"🟢 **{word}** → +{value:.3f}")
            else:
                st.markdown(f"🔴 **{word}** → {value:.3f}")
        
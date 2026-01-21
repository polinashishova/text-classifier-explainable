import streamlit as st
from src.text_classifier_explainable.model import load_model, predict
from src.text_classifier_explainable.utils import setup_logging
from src.text_classifier_explainable.data import load_json
from pathlib import Path

setup_logging()

paths_cfg_path = Path('config/paths.json')
paths = load_json(paths_cfg_path)

model_path = Path(f'{paths['models_dir']}/{paths['model']}')
explainer_path = Path(f'{paths['models_dir']}/{paths['explainer']}')

model = load_model(model_path)
explainer = load_model(explainer_path)

st.set_page_config(layout='centered')
st.title('Text classification with explanation')

text = st.text_area(
    '**Enter text**',
    height=150,
    placeholder='Write text for classification...'
)

if st.button(label='Classify'):
    if not text.strip():
        st.warning('Enter text')
    else:
        label, proba, vectorized_text = predict(model, text)
        if label[0] == 0:
            name = 'negative'
        elif label[0] == 1:
            name = 'positive'
        st.subheader('Result:')
        st.success(f'Class: {name}')
        st.write(f'Model confidence: {round(max(proba[0]), 4)}')
        
import streamlit as st
from tce.model import load_model, predict
from tce.utils import setup_logging, load_json
from tce.explain import explain
from pathlib import Path

logger = setup_logging()

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
        if name == 'negative':
            st.error(f'Class: {name}')
        elif name == 'positive':
            st.success(f'Class: {name}')
        st.write(f'Model confidence: {max(proba[0]):.2f}')

        st.subheader('The most infuential words and phrases')
        shap_words = explain(explainer, model, text, k_top=3)

        for value in shap_words:
            if value[1] > 0:
                st.markdown(f"🟢 **{value[0]}** → +{value[1]:.3f}")
            else:
                st.markdown(f"🔴 **{value[0]}** → {value[1]:.3f}")
        
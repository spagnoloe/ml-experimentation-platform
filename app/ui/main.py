"""The Streamlit UI: `streamlit run app/ui/main.py`. Pages call services and clients only."""

import streamlit as st

from app.ui import experiments

st.set_page_config(page_title="ADP Experimentation Platform", layout="wide")

navigation = st.navigation(
    [st.Page(experiments.render, title="Experiments", url_path="experiments", default=True)]
)
navigation.run()

"""The Streamlit UI: `streamlit run app/ui/main.py`. Pages call services and clients only."""

import streamlit as st

from app.ui import deployments, experiments, registry

st.set_page_config(page_title="ML Experimentation Platform", layout="wide")

navigation = st.navigation(
    [
        st.Page(experiments.render, title="Experiments", url_path="experiments", default=True),
        st.Page(registry.render, title="Registry", url_path="registry"),
        st.Page(deployments.render, title="Deployments", url_path="deployments"),
    ]
)
navigation.run()

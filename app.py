"""
DermaSense - Streamlit entry point.

Run locally:   streamlit run app.py
Deploy:        share.streamlit.io, main file path app.py

This file only sets up the page shell and the navigation. Each page lives in
views/, the model in dermasense_model.py, shared look and cached calls in ui.py.
"""

import streamlit as st

import ui

st.set_page_config(
    page_title="DermaSense | iGEM IIT Delhi 2026",
    page_icon=":material/spa:",
    layout="wide",
    initial_sidebar_state="auto",
)
ui.apply_theme()

pages = [
    st.Page("views/cover.py", title="Cover", icon=":material/spa:", url_path="cover",
            default=True),
    st.Page("views/home.py", title="Home", icon=":material/home:", url_path="home"),
    st.Page("views/simulator.py", title="Dose simulator", icon=":material/science:"),
    st.Page("views/controls.py", title="Lab vs controls", icon=":material/compare_arrows:"),
    st.Page("views/model.py", title="How the model works", icon=":material/functions:"),
]

st.navigation(pages, position="top").run()

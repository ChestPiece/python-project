import streamlit as st
import os

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def inject_custom_css():
    """Loads the main style.css for the application."""
    css_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'style.css')
    if os.path.exists(css_path):
        load_css(css_path)
    else:
        st.warning("CSS not found: assets/style.css")

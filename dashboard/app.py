"""Delivery Risk Intelligence: stakeholder dashboard (BAN6800 Module 5).

    streamlit run dashboard/app.py

Reads the Module 4 champion model (models/) and a de-identified, scored copy of
the held-out test period (dashboard/data/, built by src.models.export_dashboard).
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.set_page_config(page_title="Delivery Risk Intelligence", page_icon="🚚", layout="wide")
st.markdown("""<style>
div[data-testid="stMetricValue"] {font-size: 1.7rem;}
.block-container {padding-top: 2.2rem; max-width: 1250px;}
.note {background:#EEF5F4; border-left:4px solid #00927F; padding:.7rem 1rem; border-radius:4px; margin:.3rem 0 1rem 0;}
.warn {background:#FBF0E7; border-left:4px solid #D9691A; padding:.7rem 1rem; border-radius:4px; margin:.3rem 0 1rem 0;}
</style>""", unsafe_allow_html=True)

from dashboard import views

pg = st.navigation([
    st.Page(views.page_overview, title="Overview", icon="📊", default=True),
    st.Page(views.page_drivers, title="What drives risk", icon="🔍", url_path="drivers"),
    st.Page(views.page_explorer, title="Order explorer & what-if", icon="🧪", url_path="what-if"),
    st.Page(views.page_fairness, title="Ethical compliance", icon="⚖️", url_path="fairness"),
    st.Page(views.page_transparency, title="Transparency", icon="📄", url_path="transparency"),
])
st.sidebar.caption("BAN6800 Module 5 · Gain Tawodzera\n\nShadow mode: scores are not used for decisions.")
pg.run()

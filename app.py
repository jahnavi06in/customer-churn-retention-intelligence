import streamlit as st
from app.dashboard import render_dashboard

st.set_page_config(
    page_title="Customer Churn & Retention Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_dashboard()

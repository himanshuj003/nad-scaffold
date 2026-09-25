"""Streamlit dashboard. Run: streamlit run dashboard/app.py"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_generator import generate_dataset
from src.detectors import get_detector
from src.evaluator import evaluate

st.set_page_config(page_title="Network Anomaly Detector", page_icon="🛡️", layout="wide")
st.sidebar.title("🛡️ NAD Scaffold")
method = st.sidebar.selectbox("Method", ["hybrid","statistical","isolation_forest","rule_based"])
n = st.sidebar.slider("Flows", 1000, 10000, 3000, 500)
if st.button("Run", type="primary") or "df" not in st.session_state:
    st.session_state["df"] = generate_dataset(n, 0.05, 42)
df = st.session_state["df"]
result = get_detector(method).fit_predict(df)
df_view = df.copy(); df_view["anomaly"]=result.labels; df_view["score"]=result.scores
st.title("Network Anomaly Detector")
c1,c2,c3 = st.columns(3)
c1.metric("Flows", len(df)); c2.metric("Anomalies", result.n_anomalies)
if "is_anomaly" in df.columns:
    m = evaluate(df["is_anomaly"].values, result.labels, result.scores)
    c3.metric("F1", f"{m['f1']:.3f}")
st.plotly_chart(px.pie(names=["Normal","Anomaly"],
    values=pd.Series(result.labels).value_counts().reindex([0,1],fill_value=0).values,
    color_discrete_sequence=["#2ecc71","#e74c3c"]), use_container_width=True)
st.dataframe(df_view[df_view["anomaly"]==1].nlargest(10,"score"), use_container_width=True)

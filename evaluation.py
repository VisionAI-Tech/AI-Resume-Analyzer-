import streamlit as st
import pandas as pd
import os

st.title("📈 Resume Analyzer - Performance Evaluation")

log_file = "evaluation_log.csv"

if os.path.exists(log_file):
    df = pd.read_csv(log_file)

    st.subheader("📋 Evaluation Log")
    st.dataframe(df)

    st.subheader("📊 Metrics Summary")
    st.metric("Average Match Score", f"{df['match_score'].mean():.2f}%")
    st.line_chart(df["match_score"])
else:
    st.warning("No evaluation logs found yet. Run some analyses first.")

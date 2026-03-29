import streamlit as st

st.set_page_config(initial_sidebar_state="collapsed")
import pandas as pd
from utils.helpers import get_from_session, classify_columns
from utils.recommendations import (
    recommend_imputation,
    recommend_scaling,
    recommend_problem_types
)

st.title("2. Data Profiling")

df = get_from_session("data")

if df is None:
    st.warning("Please upload a dataset first from the Upload page.")
    st.stop()

column_info = classify_columns(df)

numeric_cols = column_info["numeric"]
categorical_cols = column_info["categorical"]
text_like_cols = column_info["text_like"]
datetime_cols = column_info["datetime"]

st.subheader("Column Classification")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Numeric Columns", len(numeric_cols))
col2.metric("Categorical Columns", len(categorical_cols))
col3.metric("Text-like Columns", len(text_like_cols))
col4.metric("Datetime Columns", len(datetime_cols))

st.markdown("### Numeric Columns")
st.write(numeric_cols if numeric_cols else "No numeric columns found.")

st.markdown("### Categorical Columns")
st.write(categorical_cols if categorical_cols else "No categorical columns found.")

st.markdown("### Text-like Columns")
st.write(text_like_cols if text_like_cols else "No text-like columns found.")

st.markdown("### Datetime Columns")
st.write(datetime_cols if datetime_cols else "No datetime columns found.")

st.subheader("Basic Statistical Summary")

if numeric_cols:
    st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
else:
    st.info("No numeric columns available for statistical summary.")

st.subheader("Target Suitability Suggestions")

regression_candidates = numeric_cols
classification_candidates = categorical_cols

col5, col6 = st.columns(2)

with col5:
    st.markdown("#### Regression Target Candidates")
    st.write(regression_candidates if regression_candidates else "No numeric target candidates found.")

with col6:
    st.markdown("#### Classification Target Candidates")
    st.write(classification_candidates if classification_candidates else "No categorical target candidates found.")

st.subheader("Recommendations")

imputation_rec = recommend_imputation(df, numeric_cols, categorical_cols)
scaling_rec = recommend_scaling(df, numeric_cols)
problem_recs = recommend_problem_types(column_info)

st.markdown("### Imputation Recommendation")
st.success(f"Recommended: {imputation_rec['recommended']}")
st.write(imputation_rec["reason"])

st.markdown("### Scaling Recommendation")
st.success(f"Recommended: {scaling_rec['recommended']}")
st.write(scaling_rec["reason"])

st.markdown("### Problem Type Suggestions")
for item in problem_recs:
    st.write(f"- {item}")
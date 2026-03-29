import streamlit as st
from utils.helpers import get_from_session, save_to_session, classify_columns
from utils.recommendations import recommend_model

st.set_page_config(initial_sidebar_state="expanded")

st.title("4. Target Configuration")

# ============================================================
# Load dataset
# ============================================================
df = get_from_session("processed_data")
if df is None:
    df = get_from_session("data")

if df is None:
    st.warning("Please upload a dataset first from the Upload page.")
    st.stop()

column_info = classify_columns(df)
numeric_cols = column_info["numeric"]
categorical_cols = column_info["categorical"]

problem_options = ["Regression", "Classification", "Clustering"]

# ============================================================
# Initialize saved values only once
# ============================================================
if "saved_problem_type" not in st.session_state:
    st.session_state["saved_problem_type"] = get_from_session("saved_problem_type", "Regression")

if "saved_target_column" not in st.session_state:
    st.session_state["saved_target_column"] = get_from_session("saved_target_column", None)

saved_problem_type = st.session_state["saved_problem_type"]
saved_target_column = st.session_state["saved_target_column"]

# ============================================================
# Sidebar controls
# ============================================================
st.sidebar.header("Target Configuration")

problem_index = problem_options.index(saved_problem_type) if saved_problem_type in problem_options else 0
problem_type = st.sidebar.radio(
    "Select Problem Type",
    problem_options,
    index=problem_index,
    key="problem_type_sidebar"
)

target_col = None
classification_candidates = [
    col for col in categorical_cols
    if 2 <= df[col].nunique(dropna=True) <= 15
]

if problem_type == "Regression":
    if not numeric_cols:
        st.sidebar.error("No numeric columns are available for regression target selection.")
    else:
        reg_default = saved_target_column if saved_target_column in numeric_cols else numeric_cols[0]
        target_col = st.sidebar.selectbox(
            "Select Numeric Target Column",
            numeric_cols,
            index=numeric_cols.index(reg_default),
            key="target_column_regression_sidebar"
        )

elif problem_type == "Classification":
    if not classification_candidates:
        st.sidebar.error("No suitable categorical columns are available for classification target selection.")
    else:
        cls_default = saved_target_column if saved_target_column in classification_candidates else classification_candidates[0]
        target_col = st.sidebar.selectbox(
            "Select Categorical Target Column",
            classification_candidates,
            index=classification_candidates.index(cls_default),
            key="target_column_classification_sidebar"
        )

elif problem_type == "Clustering":
    st.sidebar.info("No target column is required for clustering.")

if st.sidebar.button("Save Configuration", use_container_width=True):
    save_to_session("saved_problem_type", problem_type)
    st.session_state["saved_problem_type"] = problem_type

    if problem_type == "Clustering":
        save_to_session("saved_target_column", None)
        st.session_state["saved_target_column"] = None
        st.sidebar.success("Clustering configuration saved.")
    elif target_col is None:
        st.sidebar.error("Please select a valid target column.")
    else:
        save_to_session("saved_target_column", target_col)
        st.session_state["saved_target_column"] = target_col
        st.sidebar.success("Target configuration saved.")

# Refresh values after any save
current_problem_type = get_from_session("saved_problem_type", problem_type)
current_target_col = get_from_session("saved_target_column", target_col)

# ============================================================
# Main page summary only
# ============================================================
st.subheader("Current Selection")
col1, col2 = st.columns(2)
col1.metric("Problem Type", current_problem_type)
col2.metric("Target Column", current_target_col if current_target_col else "Not Required")

st.markdown("---")
st.subheader("Available Column Summary")
col3, col4 = st.columns(2)
col3.metric("Numeric Columns", len(numeric_cols))
col4.metric("Categorical Columns", len(categorical_cols))

with st.expander("View Numeric Columns", expanded=False):
    st.write(numeric_cols if numeric_cols else "No numeric columns found.")

with st.expander("View Categorical Columns", expanded=False):
    st.write(categorical_cols if categorical_cols else "No categorical columns found.")

if current_problem_type == "Regression":
    if not numeric_cols:
        st.error("No numeric columns are available for regression target selection.")
    else:
        st.success(f"Selected target column: {current_target_col}")
        rec = recommend_model(current_problem_type, feature_count=len(df.columns))
        st.info(f"Recommended Model: {rec['recommended']}")
        st.write(rec["reason"])

elif current_problem_type == "Classification":
    if not classification_candidates:
        st.error("No suitable categorical columns are available for classification target selection.")
    else:
        unique_count = df[current_target_col].nunique(dropna=True) if current_target_col in df.columns else 0
        st.success(f"Selected target column: {current_target_col}")
        rec = recommend_model(
            current_problem_type,
            feature_count=len(df.columns),
            target_unique_count=unique_count
        )
        st.info(f"Recommended Model: {rec['recommended']}")
        st.write(rec["reason"])

elif current_problem_type == "Clustering":
    st.success("No target column is required for clustering.")
    rec = recommend_model(current_problem_type, feature_count=len(df.columns))
    st.info(f"Recommended Model: {rec['recommended']}")
    st.write(rec["reason"])

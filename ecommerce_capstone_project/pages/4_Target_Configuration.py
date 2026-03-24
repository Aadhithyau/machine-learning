import streamlit as st
from utils.helpers import get_from_session, save_to_session, classify_columns
from utils.recommendations import recommend_model

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

# ============================================================
# Callbacks
# ============================================================
def on_problem_type_change():
    st.session_state["saved_problem_type"] = st.session_state["problem_type_widget"]
    save_to_session("saved_problem_type", st.session_state["saved_problem_type"])

def on_regression_target_change():
    st.session_state["saved_target_column"] = st.session_state["target_column_regression_widget"]
    save_to_session("saved_target_column", st.session_state["saved_target_column"])

def on_classification_target_change():
    st.session_state["saved_target_column"] = st.session_state["target_column_classification_widget"]
    save_to_session("saved_target_column", st.session_state["saved_target_column"])

# ============================================================
# Problem type widget
# ============================================================
saved_problem_type = st.session_state["saved_problem_type"]
problem_index = problem_options.index(saved_problem_type) if saved_problem_type in problem_options else 0

problem_type = st.radio(
    "Select Problem Type",
    problem_options,
    index=problem_index,
    horizontal=True,
    key="problem_type_widget",
    on_change=on_problem_type_change
)

# Keep latest value saved
st.session_state["saved_problem_type"] = problem_type
save_to_session("saved_problem_type", problem_type)

# ============================================================
# REGRESSION
# ============================================================
if problem_type == "Regression":
    if not numeric_cols:
        st.error("No numeric columns are available for regression target selection.")
        st.stop()

    saved_target = st.session_state.get("saved_target_column")

    if saved_target in numeric_cols:
        reg_index = numeric_cols.index(saved_target)
    else:
        reg_index = 0

    target_col = st.selectbox(
        "Select Numeric Target Column",
        numeric_cols,
        index=reg_index,
        key="target_column_regression_widget",
        on_change=on_regression_target_change
    )

    st.session_state["saved_target_column"] = target_col
    save_to_session("saved_target_column", target_col)

    st.success(f"Selected target column: {target_col}")

    rec = recommend_model(problem_type, feature_count=len(df.columns))
    st.info(f"Recommended Model: {rec['recommended']}")
    st.write(rec["reason"])

# ============================================================
# CLASSIFICATION
# ============================================================
elif problem_type == "Classification":
    classification_candidates = [
        col for col in categorical_cols
        if 2 <= df[col].nunique(dropna=True) <= 15
    ]

    if not classification_candidates:
        st.error("No suitable categorical columns are available for classification target selection.")
        st.stop()

    saved_target = st.session_state.get("saved_target_column")

    if saved_target in classification_candidates:
        cls_index = classification_candidates.index(saved_target)
    else:
        cls_index = 0

    target_col = st.selectbox(
        "Select Categorical Target Column",
        classification_candidates,
        index=cls_index,
        key="target_column_classification_widget",
        on_change=on_classification_target_change
    )

    st.session_state["saved_target_column"] = target_col
    save_to_session("saved_target_column", target_col)

    unique_count = df[target_col].nunique(dropna=True)

    st.success(f"Selected target column: {target_col}")

    rec = recommend_model(
        problem_type,
        feature_count=len(df.columns),
        target_unique_count=unique_count
    )
    st.info(f"Recommended Model: {rec['recommended']}")
    st.write(rec["reason"])

# ============================================================
# CLUSTERING
# ============================================================
elif problem_type == "Clustering":
    st.session_state["saved_target_column"] = None
    save_to_session("saved_target_column", None)

    st.success("No target column is required for clustering.")

    rec = recommend_model(problem_type, feature_count=len(df.columns))
    st.info(f"Recommended Model: {rec['recommended']}")
    st.write(rec["reason"])
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    PowerTransformer
)

from utils.helpers import get_from_session, save_to_session, classify_columns, missing_value_summary
from utils.recommendations import recommend_imputation, recommend_scaling

st.title("3. Preprocessing")

# ==============================
# Load dataset from session
# ==============================
base_df = get_from_session("processed_data")
if base_df is None:
    base_df = get_from_session("data")

if base_df is None:
    st.warning("Please upload a dataset first from the Upload page.")
    st.stop()

df = base_df.copy()

# Initialize preprocessing summary once
if "preprocessing_summary" not in st.session_state:
    st.session_state["preprocessing_summary"] = []

# ==============================
# Helper functions
# ==============================
def classify_skewness(skew_value):
    if pd.isna(skew_value):
        return "Not Available"
    if -0.5 <= skew_value <= 0.5:
        return "Symmetric"
    elif skew_value > 0.5:
        return "Right-Skewed"
    else:
        return "Left-Skewed"


def add_summary_entry(message):
    summary = get_from_session("preprocessing_summary", [])
    if message not in summary:
        summary.append(message)
        save_to_session("preprocessing_summary", summary)


def get_numeric_and_categorical_columns(dataframe):
    column_info = classify_columns(dataframe)
    numeric_cols = column_info["numeric"]
    categorical_cols = column_info["categorical"]
    return numeric_cols, categorical_cols


numeric_cols, categorical_cols = get_numeric_and_categorical_columns(df)

# ==============================
# Current dataset snapshot
# ==============================
st.subheader("Current Dataset Snapshot")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", int(df.isnull().sum().sum()))

st.dataframe(df.head(), use_container_width=True)

# ==============================
# Section 1: Missing Value Handling
# ==============================
st.markdown("---")
st.header("1. Missing Value Handling")

missing_summary_df = missing_value_summary(df)
st.subheader("Missing Value Summary")
st.dataframe(missing_summary_df, use_container_width=True)

imputation_rec = recommend_imputation(df, numeric_cols, categorical_cols)
st.markdown("### Recommendation")
st.success(f"Recommended: {imputation_rec['recommended']}")
st.write(imputation_rec["reason"])

imputer_choice = st.selectbox(
    "Select Imputation Technique",
    ["No Imputation", "Simple Imputer", "KNN Imputer", "Iterative Imputer"],
    key="imputer_choice"
)

numeric_strategy = None
categorical_strategy = None
constant_fill_value = None
knn_neighbors = None
iterative_max_iter = None
iterative_random_state = None

if imputer_choice == "Simple Imputer":
    numeric_strategy = st.selectbox(
        "Numeric Strategy",
        ["mean", "median", "most_frequent"],
        key="simple_numeric_strategy"
    )
    categorical_strategy = st.selectbox(
        "Categorical Strategy",
        ["most_frequent", "constant"],
        key="simple_categorical_strategy"
    )

    if categorical_strategy == "constant":
        constant_fill_value = st.text_input(
            "Fill value for categorical columns",
            value=st.session_state.get("simple_constant_fill", "Missing"),
            key="simple_constant_fill"
        )

elif imputer_choice == "KNN Imputer":
    knn_neighbors = st.slider(
        "Number of Neighbors (k)",
        min_value=1,
        max_value=15,
        value=st.session_state.get("knn_neighbors", 5),
        key="knn_neighbors"
    )

elif imputer_choice == "Iterative Imputer":
    iterative_max_iter = st.slider(
        "Maximum Iterations",
        min_value=5,
        max_value=50,
        value=st.session_state.get("iterative_max_iter", 10),
        key="iterative_max_iter"
    )
    iterative_random_state = st.number_input(
        "Random State",
        min_value=0,
        max_value=9999,
        value=st.session_state.get("iterative_random_state", 42),
        step=1,
        key="iterative_random_state"
    )

if st.button("Apply Imputation", key="apply_imputation_btn"):
    working_df = df.copy()

    if imputer_choice == "No Imputation":
        st.info("No imputation was applied.")

    elif imputer_choice == "Simple Imputer":
        if numeric_cols:
            num_imputer = SimpleImputer(strategy=numeric_strategy)
            working_df[numeric_cols] = num_imputer.fit_transform(working_df[numeric_cols])

        if categorical_cols:
            if categorical_strategy == "constant":
                cat_imputer = SimpleImputer(strategy="constant", fill_value=constant_fill_value)
            else:
                cat_imputer = SimpleImputer(strategy=categorical_strategy)

            working_df[categorical_cols] = cat_imputer.fit_transform(working_df[categorical_cols])

        save_to_session("processed_data", working_df)
        add_summary_entry(
            f"Applied Simple Imputer | Numeric: {numeric_strategy} | Categorical: {categorical_strategy}"
        )
        st.success("Simple Imputer applied successfully.")

    elif imputer_choice == "KNN Imputer":
        if not numeric_cols:
            st.error("KNN Imputer requires numeric columns.")
        else:
            if categorical_cols:
                cat_imputer = SimpleImputer(strategy="most_frequent")
                working_df[categorical_cols] = cat_imputer.fit_transform(working_df[categorical_cols])

            knn_imputer = KNNImputer(n_neighbors=knn_neighbors)
            working_df[numeric_cols] = knn_imputer.fit_transform(working_df[numeric_cols])

            save_to_session("processed_data", working_df)
            add_summary_entry(f"Applied KNN Imputer | k={knn_neighbors}")
            st.success("KNN Imputer applied successfully.")

    elif imputer_choice == "Iterative Imputer":
        if not numeric_cols:
            st.error("Iterative Imputer requires numeric columns.")
        else:
            if categorical_cols:
                cat_imputer = SimpleImputer(strategy="most_frequent")
                working_df[categorical_cols] = cat_imputer.fit_transform(working_df[categorical_cols])

            iter_imputer = IterativeImputer(
                max_iter=iterative_max_iter,
                random_state=int(iterative_random_state)
            )
            working_df[numeric_cols] = iter_imputer.fit_transform(working_df[numeric_cols])

            save_to_session("processed_data", working_df)
            add_summary_entry(
                f"Applied Iterative Imputer | max_iter={iterative_max_iter} | random_state={int(iterative_random_state)}"
            )
            st.success("Iterative Imputer applied successfully.")

# Refresh local df after any previous processing
df = get_from_session("processed_data", get_from_session("data")).copy()
numeric_cols, categorical_cols = get_numeric_and_categorical_columns(df)

# ==============================
# Section 2: Symmetry / Skewness
# ==============================
st.markdown("---")
st.header("2. Symmetry / Skewness Analysis")

if numeric_cols:
    stats_rows = []

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) == 0:
            mean_val = np.nan
            median_val = np.nan
            skew_val = np.nan
        else:
            mean_val = series.mean()
            median_val = series.median()
            skew_val = series.skew()

        stats_rows.append({
            "Feature": col,
            "Mean": mean_val,
            "Median": median_val,
            "Skewness": skew_val,
            "Classification": classify_skewness(skew_val)
        })

    skewness_df = pd.DataFrame(stats_rows)

    st.subheader("Mean, Median, Skewness, and Classification")
    st.dataframe(skewness_df, use_container_width=True)

    symmetric_features = skewness_df[skewness_df["Classification"] == "Symmetric"]["Feature"].tolist()
    right_skewed_features = skewness_df[skewness_df["Classification"] == "Right-Skewed"]["Feature"].tolist()
    left_skewed_features = skewness_df[skewness_df["Classification"] == "Left-Skewed"]["Feature"].tolist()

    col4, col5, col6 = st.columns(3)
    col4.metric("Symmetric Features", len(symmetric_features))
    col5.metric("Right-Skewed Features", len(right_skewed_features))
    col6.metric("Left-Skewed Features", len(left_skewed_features))

    st.markdown("### Feature Groups")
    st.write("**Symmetric:**", symmetric_features if symmetric_features else "None")
    st.write("**Right-Skewed:**", right_skewed_features if right_skewed_features else "None")
    st.write("**Left-Skewed:**", left_skewed_features if left_skewed_features else "None")

else:
    st.info("No numeric columns available for skewness analysis.")

# ==============================
# Section 3: Scaling
# ==============================
st.markdown("---")
st.header("3. Scaling")

df = get_from_session("processed_data", get_from_session("data")).copy()
numeric_cols, categorical_cols = get_numeric_and_categorical_columns(df)

scaling_rec = recommend_scaling(df, numeric_cols)
st.markdown("### Recommendation")
st.success(f"Recommended: {scaling_rec['recommended']}")
st.write(scaling_rec["reason"])

if numeric_cols:
    default_scale_cols = st.session_state.get("scale_columns", numeric_cols)
    default_scale_cols = [col for col in default_scale_cols if col in numeric_cols]

    scale_columns = st.multiselect(
        "Select Numeric Columns to Scale",
        options=numeric_cols,
        default=default_scale_cols,
        key="scale_columns"
    )

    scaler_choice = st.selectbox(
        "Select Scaling Technique",
        ["No Scaling", "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler"],
        key="scaler_choice"
    )

    if st.button("Apply Scaling", key="apply_scaling_btn"):
        working_df = df.copy()

        if scaler_choice == "No Scaling":
            st.info("No scaling was applied.")

        elif not scale_columns:
            st.error("Please select at least one numeric column to scale.")

        else:
            if working_df[scale_columns].isnull().sum().sum() > 0:
                st.error("Scaling cannot be applied while selected columns still contain missing values. Please apply imputation first.")
            else:
                if scaler_choice == "StandardScaler":
                    scaler = StandardScaler()
                elif scaler_choice == "MinMaxScaler":
                    scaler = MinMaxScaler()
                elif scaler_choice == "RobustScaler":
                    scaler = RobustScaler()
                else:
                    scaler = MaxAbsScaler()

                working_df[scale_columns] = scaler.fit_transform(working_df[scale_columns])

                save_to_session("processed_data", working_df)
                add_summary_entry(f"Applied {scaler_choice} on columns: {', '.join(scale_columns)}")
                st.success(f"{scaler_choice} applied successfully.")

else:
    st.info("No numeric columns available for scaling.")

# ==============================
# Section 4: Transformations
# ==============================
st.markdown("---")
st.header("4. Transformations")

df = get_from_session("processed_data", get_from_session("data")).copy()
numeric_cols, categorical_cols = get_numeric_and_categorical_columns(df)

if numeric_cols:
    default_transform_cols = st.session_state.get("transform_columns", [])
    default_transform_cols = [col for col in default_transform_cols if col in numeric_cols]

    transform_columns = st.multiselect(
        "Select Numeric Columns to Transform",
        options=numeric_cols,
        default=default_transform_cols,
        key="transform_columns"
    )

    transform_choice = st.selectbox(
        "Select Transformation",
        ["No Transformation", "Log", "Sqrt", "Box-Cox", "Yeo-Johnson"],
        key="transform_choice"
    )

    if st.button("Apply Transformation", key="apply_transformation_btn"):
        working_df = df.copy()

        if transform_choice == "No Transformation":
            st.info("No transformation was applied.")

        elif not transform_columns:
            st.error("Please select at least one numeric column to transform.")

        elif working_df[transform_columns].isnull().sum().sum() > 0:
            st.error("Transformations cannot be applied while selected columns still contain missing values. Please apply imputation first.")

        else:
            skipped_columns = []

            if transform_choice == "Log":
                for col in transform_columns:
                    if (working_df[col] < 0).any():
                        skipped_columns.append(col)
                    else:
                        working_df[col] = np.log1p(working_df[col])

            elif transform_choice == "Sqrt":
                for col in transform_columns:
                    if (working_df[col] < 0).any():
                        skipped_columns.append(col)
                    else:
                        working_df[col] = np.sqrt(working_df[col])

            elif transform_choice == "Box-Cox":
                valid_cols = []
                for col in transform_columns:
                    if (working_df[col] <= 0).any():
                        skipped_columns.append(col)
                    else:
                        valid_cols.append(col)

                if valid_cols:
                    transformer = PowerTransformer(method="box-cox")
                    working_df[valid_cols] = transformer.fit_transform(working_df[valid_cols])

            elif transform_choice == "Yeo-Johnson":
                transformer = PowerTransformer(method="yeo-johnson")
                working_df[transform_columns] = transformer.fit_transform(working_df[transform_columns])

            save_to_session("processed_data", working_df)
            add_summary_entry(f"Applied {transform_choice} transformation on columns: {', '.join(transform_columns)}")

            if skipped_columns:
                st.warning(
                    f"{transform_choice} was skipped for these columns because their values were not valid for this transformation: "
                    + ", ".join(skipped_columns)
                )

            st.success(f"{transform_choice} transformation applied successfully.")

else:
    st.info("No numeric columns available for transformation.")

# ==============================
# Section 5: Final preview and save status
# ==============================
st.markdown("---")
st.header("5. Processed Dataset Preview")

final_df = get_from_session("processed_data", get_from_session("data")).copy()

col7, col8, col9 = st.columns(3)
col7.metric("Rows", final_df.shape[0])
col8.metric("Columns", final_df.shape[1])
col9.metric("Missing Values After Processing", int(final_df.isnull().sum().sum()))

st.dataframe(final_df.head(), use_container_width=True)

summary_entries = get_from_session("preprocessing_summary", [])

st.subheader("Preprocessing Summary")
if summary_entries:
    for i, item in enumerate(summary_entries, start=1):
        st.write(f"{i}. {item}")
else:
    st.info("No preprocessing steps have been applied yet.")
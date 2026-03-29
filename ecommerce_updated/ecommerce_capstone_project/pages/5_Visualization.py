import streamlit as st

st.set_page_config(initial_sidebar_state="collapsed")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils.helpers import get_from_session

st.title("5. Visualization")
st.subheader("Module 4: Univariate Data Visualization")

# ============================================================
# Load dataset
# ============================================================
df = get_from_session("processed_data")
if df is None:
    df = get_from_session("data")

if df is None:
    st.warning("Please upload and preprocess a dataset first.")
    st.stop()

# ============================================================
# Helper: classify columns properly
# Treat ID-like numeric columns as categorical for bar plot
# ============================================================
def split_columns_for_visualization(dataframe):
    categorical_cols = []
    numeric_cols = []

    id_keywords = ["id", "invoice", "stock", "code", "number", "no"]

    for col in dataframe.columns:
        col_lower = col.lower()

        # If object/string column => categorical
        if dataframe[col].dtype == "object":
            categorical_cols.append(col)

        # If numeric but column name looks like identifier => categorical
        elif pd.api.types.is_numeric_dtype(dataframe[col]):
            if any(keyword in col_lower for keyword in id_keywords):
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)

    return categorical_cols, numeric_cols


categorical_cols, numeric_cols = split_columns_for_visualization(df)

# ============================================================
# Page settings for speed and readability
# ============================================================
MAX_CATEGORIES_TO_SHOW = 20
MAX_ROWS_FOR_PLOTTING = 10000

if len(df) > MAX_ROWS_FOR_PLOTTING:
    plot_df = df.sample(MAX_ROWS_FOR_PLOTTING, random_state=42)
    st.info(f"Dataset is large, so visualizations are generated using a random sample of {MAX_ROWS_FOR_PLOTTING} rows for faster performance.")
else:
    plot_df = df.copy()

# ============================================================
# BAR PLOTS FOR ALL CATEGORICAL FEATURES
# ============================================================
st.markdown("---")
st.header("Bar Plots for Categorical Features")

if not categorical_cols:
    st.info("No categorical features available for bar plots.")
else:
    for col in categorical_cols:
        st.subheader(f"Bar Plot - {col}")

        value_counts = plot_df[col].astype(str).value_counts().head(MAX_CATEGORIES_TO_SHOW)

        if value_counts.empty:
            st.warning(f"No valid values available for {col}.")
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)

        ax.set_title(f"Top {min(MAX_CATEGORIES_TO_SHOW, len(value_counts))} Categories in {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        st.pyplot(fig)

        total_unique = plot_df[col].astype(str).nunique(dropna=True)
        st.write(f"Unique values in **{col}**: {total_unique}")

        if total_unique > MAX_CATEGORIES_TO_SHOW:
            st.caption(f"Showing only top {MAX_CATEGORIES_TO_SHOW} categories to avoid overlapping labels and slow rendering.")

# ============================================================
# HISTOGRAMS FOR ALL NUMERIC FEATURES
# ============================================================
st.markdown("---")
st.header("Histograms for Numeric Features")

if not numeric_cols:
    st.info("No numeric features available for histograms.")
else:
    for col in numeric_cols:
        st.subheader(f"Histogram - {col}")

        series = plot_df[col].dropna()

        if series.empty:
            st.warning(f"No valid numeric values available for {col}.")
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(series, bins=20, kde=True, ax=ax)

        ax.set_title(f"Histogram of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        plt.tight_layout()

        st.pyplot(fig)

# ============================================================
# BOX PLOTS FOR ALL NUMERIC FEATURES
# ============================================================
st.markdown("---")
st.header("Box Plots for Numeric Features")

if not numeric_cols:
    st.info("No numeric features available for box plots.")
else:
    for col in numeric_cols:
        st.subheader(f"Box Plot - {col}")

        series = plot_df[col].dropna()

        if series.empty:
            st.warning(f"No valid numeric values available for {col}.")
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        median = series.median()
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]

        fig, ax = plt.subplots(figsize=(10, 3))
        sns.boxplot(x=series, ax=ax)

        ax.set_title(f"Box Plot of {col}")
        plt.tight_layout()

        st.pyplot(fig)

        st.write(f"**Median:** {median:.2f}")
        st.write(f"**IQR:** {iqr:.2f}")
        st.write(f"**Outliers:** {len(outliers)}")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)

from utils.helpers import get_from_session

st.title("7. Metrics")

problem_type = get_from_session("saved_problem_type", "Regression")
model_name = get_from_session("model_name", "Model")

st.subheader("Current Evaluation Setup")
st.write(f"**Problem Type:** {problem_type}")
st.write(f"**Model Name:** {model_name}")

# ============================================================
# REGRESSION
# ============================================================
if problem_type == "Regression":
    y_test = get_from_session("y_test")
    y_pred = get_from_session("y_pred")

    if y_test is None or y_pred is None:
        st.warning("No regression results found. Please train a regression model first.")
        st.stop()

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    st.markdown("## Regression Metrics")
    c1, c2 = st.columns(2)
    c1.metric("MAE", f"{mae:.4f}")
    c2.metric("MSE", f"{mse:.4f}")

    c3, c4 = st.columns(2)
    c3.metric("RMSE", f"{rmse:.4f}")
    c4.metric("R² Score", f"{r2:.4f}")

    st.markdown("## Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

    residuals = y_test - y_pred
    st.markdown("## Residual Plot")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.scatter(y_pred, residuals)
    ax2.axhline(y=0, linestyle="--")
    ax2.set_xlabel("Predicted Values")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residual Plot")
    st.pyplot(fig2)

# ============================================================
# CLASSIFICATION
# ============================================================
elif problem_type == "Classification":
    y_test = get_from_session("y_test")
    y_pred = get_from_session("y_pred")
    class_names = get_from_session("class_names", None)

    if y_test is None or y_pred is None:
        st.warning("No classification results found. Please train a classification model first.")
        st.stop()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    st.markdown("## Classification Metrics")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{accuracy:.4f}")
    c2.metric("Precision", f"{precision:.4f}")

    c3, c4 = st.columns(2)
    c3.metric("Recall", f"{recall:.4f}")
    c4.metric("F1 Score", f"{f1:.4f}")

    st.markdown("## Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(14, 10))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names if class_names else None
    )
    disp.plot(ax=ax, xticks_rotation=60, colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("## Prediction Summary")
    result_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })
    st.dataframe(result_df.head(20), use_container_width=True)

# ============================================================
# CLUSTERING
# ============================================================
elif problem_type == "Clustering":
    cluster_labels = get_from_session("cluster_labels")
    cluster_input_data = get_from_session("cluster_input_data")

    if cluster_labels is None or cluster_input_data is None:
        st.warning("No clustering results found. Please train a clustering model first.")
        st.stop()

    cluster_labels = np.array(cluster_labels)
    unique_labels = np.unique(cluster_labels)

    valid_mask = cluster_labels != -1
    valid_labels = cluster_labels[valid_mask]

    sil_score = None

    if len(np.unique(valid_labels)) >= 2 and valid_mask.sum() >= 2:
        sil_score = silhouette_score(cluster_input_data[valid_mask], valid_labels)

    st.markdown("## Clustering Metrics")

    if sil_score is not None:
        st.metric("Silhouette Score", f"{sil_score:.4f}")
    else:
        st.warning("Silhouette Score cannot be computed properly for this clustering result.")

    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()

    cluster_df = cluster_counts.reset_index()
    cluster_df.columns = ["Cluster", "Count"]

    st.markdown("## Cluster Distribution")
    st.dataframe(cluster_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(cluster_df["Cluster"].astype(str), cluster_df["Count"])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.set_title("Cluster Distribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    if -1 in cluster_df["Cluster"].values:
        st.info("Cluster -1 in DBSCAN means noise / outlier points.")

# ============================================================
# FALLBACK
# ============================================================
else:
    st.error("Invalid problem type selected.")
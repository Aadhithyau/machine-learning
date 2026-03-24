import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)

from utils.helpers import get_from_session

st.title("8. Final Summary")

# ============================================================
# LOAD SESSION DATA
# ============================================================
raw_data = get_from_session("data")
processed_data = get_from_session("processed_data")

problem_type = get_from_session("saved_problem_type", "Not Selected")
target_col = get_from_session("saved_target_column", "Not Selected")
model_name = get_from_session("model_name", "Not Trained")

dropped_feature_columns = get_from_session("dropped_feature_columns", [])

y_test = get_from_session("y_test")
y_pred = get_from_session("y_pred")
class_names = get_from_session("class_names", None)

cluster_labels = get_from_session("cluster_labels")
cluster_input_data = get_from_session("cluster_input_data")

# ============================================================
# BASIC DATASET SUMMARY
# ============================================================
st.subheader("Project Overview")

c1, c2, c3 = st.columns(3)

raw_rows = raw_data.shape[0] if raw_data is not None else 0
raw_cols = raw_data.shape[1] if raw_data is not None else 0

processed_rows = processed_data.shape[0] if processed_data is not None else 0
processed_cols = processed_data.shape[1] if processed_data is not None else 0

c1.metric("Problem Type", str(problem_type))
c2.metric("Target Column", str(target_col) if target_col else "Not Required")
c3.metric("Model Used", str(model_name))

c4, c5 = st.columns(2)
c4.metric("Raw Dataset Shape", f"{raw_rows} × {raw_cols}")
c5.metric("Processed Dataset Shape", f"{processed_rows} × {processed_cols}")

# ============================================================
# DATASET DETAILS
# ============================================================
st.markdown("## Dataset Details")

if processed_data is not None:
    summary_df = pd.DataFrame({
        "Column Name": processed_data.columns,
        "Data Type": [str(dtype) for dtype in processed_data.dtypes],
        "Missing Values": processed_data.isnull().sum().values,
        "Unique Values": processed_data.nunique(dropna=False).values
    })

    st.dataframe(summary_df, width="stretch")

    st.markdown("### Processed Dataset Preview")
    st.dataframe(processed_data.head(10), width="stretch")
else:
    st.warning("No processed dataset found.")

# ============================================================
# FEATURE SUMMARY
# ============================================================
st.markdown("## Feature Summary")

if processed_data is not None:
    usable_features = list(processed_data.columns)

    if problem_type in ["Regression", "Classification"] and target_col in usable_features:
        usable_features.remove(target_col)

    st.write(f"**Number of usable feature columns:** {len(usable_features)}")
    st.write("**Usable feature columns:**")
    st.code(", ".join(usable_features) if usable_features else "No usable features found.")

    st.write(f"**Dropped feature columns count:** {len(dropped_feature_columns)}")
    st.write("**Dropped feature columns:**")
    st.code(", ".join(dropped_feature_columns) if dropped_feature_columns else "None")
else:
    st.info("Feature summary cannot be shown because processed dataset is missing.")

# ============================================================
# MODEL PERFORMANCE SUMMARY
# ============================================================
st.markdown("## Model Performance Summary")

# ---------------- REGRESSION ----------------
if problem_type == "Regression":
    if y_test is not None and y_pred is not None:
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        c1, c2 = st.columns(2)
        c1.metric("MAE", f"{mae:.4f}")
        c2.metric("MSE", f"{mse:.4f}")

        c3, c4 = st.columns(2)
        c3.metric("RMSE", f"{rmse:.4f}")
        c4.metric("R² Score", f"{r2:.4f}")

        st.markdown("### Performance Interpretation")
        if r2 >= 0.90:
            st.success("Excellent regression performance. The model explains most of the target variation.")
        elif r2 >= 0.75:
            st.info("Good regression performance. The model is reasonably reliable.")
        elif r2 >= 0.50:
            st.warning("Moderate regression performance. Further feature engineering may improve results.")
        else:
            st.error("Weak regression performance. Model or features may need improvement.")
    else:
        st.warning("Regression model results not found.")

# ---------------- CLASSIFICATION ----------------
elif problem_type == "Classification":
    if y_test is not None and y_pred is not None:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{accuracy:.4f}")
        c2.metric("Precision", f"{precision:.4f}")

        c3, c4 = st.columns(2)
        c3.metric("Recall", f"{recall:.4f}")
        c4.metric("F1 Score", f"{f1:.4f}")

        st.markdown("### Performance Interpretation")
        if accuracy >= 0.90:
            st.success("Excellent classification performance. The model predicts classes very accurately.")
        elif accuracy >= 0.75:
            st.info("Good classification performance. The model is performing well.")
        elif accuracy >= 0.60:
            st.warning("Moderate classification performance. More tuning may improve the results.")
        else:
            st.error("Weak classification performance. Consider changing the target or improving features.")

        if class_names is not None:
            st.markdown("### Target Classes")
            st.write(", ".join(map(str, class_names)))
    else:
        st.warning("Classification model results not found.")

# ---------------- CLUSTERING ----------------
elif problem_type == "Clustering":
    if cluster_labels is not None and cluster_input_data is not None:
        cluster_labels = np.array(cluster_labels)
        unique_labels = np.unique(cluster_labels)

        valid_mask = cluster_labels != -1
        valid_labels = cluster_labels[valid_mask]

        sil_score = None
        if valid_mask.sum() >= 2 and len(np.unique(valid_labels)) >= 2:
            sil_score = silhouette_score(cluster_input_data[valid_mask], valid_labels)

        total_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        noise_points = int(np.sum(cluster_labels == -1)) if -1 in cluster_labels else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Clusters", total_clusters)
        c2.metric("Noise Points", noise_points)
        c3.metric("Silhouette Score", f"{sil_score:.4f}" if sil_score is not None else "N/A")

        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        cluster_df = cluster_counts.reset_index()
        cluster_df.columns = ["Cluster", "Count"]

        st.markdown("### Cluster Distribution")
        st.dataframe(cluster_df, width="stretch")

        st.markdown("### Performance Interpretation")
        if sil_score is None:
            st.warning("Silhouette score could not be computed properly for this clustering result.")
        elif sil_score >= 0.50:
            st.success("Strong cluster separation. Clustering quality is very good.")
        elif sil_score >= 0.25:
            st.info("Moderate cluster structure. Clustering is acceptable.")
        else:
            st.warning("Weak cluster separation. Feature engineering or tuning may improve clustering.")
    else:
        st.warning("Clustering results not found.")

else:
    st.info("Problem type not selected yet.")

# ============================================================
# FINAL CONCLUSION
# ============================================================
st.markdown("## Final Conclusion")

if problem_type == "Regression" and y_test is not None and y_pred is not None:
    r2 = r2_score(y_test, y_pred)
    st.write(
        f"The selected regression model **{model_name}** was trained using the target column "
        f"**{target_col}**. The model achieved an **R² score of {r2:.4f}**, indicating its predictive performance "
        f"on the processed dataset."
    )

elif problem_type == "Classification" and y_test is not None and y_pred is not None:
    accuracy = accuracy_score(y_test, y_pred)
    st.write(
        f"The selected classification model **{model_name}** was trained using the target column "
        f"**{target_col}**. The model achieved an **accuracy of {accuracy:.4f}**, showing its ability "
        f"to classify the processed dataset effectively."
    )

elif problem_type == "Clustering" and cluster_labels is not None and cluster_input_data is not None:
    cluster_labels = np.array(cluster_labels)
    valid_mask = cluster_labels != -1
    valid_labels = cluster_labels[valid_mask]

    sil_score = None
    if valid_mask.sum() >= 2 and len(np.unique(valid_labels)) >= 2:
        sil_score = silhouette_score(cluster_input_data[valid_mask], valid_labels)

    st.write(
        f"The selected clustering model **{model_name}** was applied to the processed dataset. "
        f"The clustering output produced meaningful groupings with a "
        f"**silhouette score of {sil_score:.4f}**." if sil_score is not None else
        f"The selected clustering model **{model_name}** was applied to the processed dataset. "
        f"However, silhouette score could not be computed properly for this clustering result."
    )

else:
    st.write("Train a model first to generate the final summary.")

# ============================================================
# DOWNLOADABLE REPORT TEXT
# ============================================================
st.markdown("## Ready-to-Use Summary")

summary_text = f"""
Problem Type: {problem_type}
Target Column: {target_col}
Model Used: {model_name}
Raw Dataset Shape: {raw_rows} x {raw_cols}
Processed Dataset Shape: {processed_rows} x {processed_cols}
Dropped Feature Columns: {', '.join(dropped_feature_columns) if dropped_feature_columns else 'None'}
"""

st.text_area("Copy Summary", summary_text.strip(), height=200)
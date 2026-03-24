import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.cluster import KMeans, DBSCAN, MeanShift
from minisom import MiniSom

from utils.helpers import get_from_session, save_to_session

st.title("6. Modeling")

# ============================================================
# Load dataset and saved configuration
# ============================================================
df = get_from_session("processed_data")
if df is None:
    df = get_from_session("data")

if df is None:
    st.warning("Please upload and preprocess a dataset first.")
    st.stop()

problem_type = get_from_session("saved_problem_type", "Regression")
target_col = get_from_session("saved_target_column", None)

st.subheader("Current Configuration")
st.write(f"**Problem Type:** {problem_type}")
st.write(f"**Target Column:** {target_col if target_col else 'Not Required for Clustering'}")

# ============================================================
# Feature preparation helpers
# ============================================================
HIGH_CARDINALITY_THRESHOLD = 50
TEXT_LENGTH_THRESHOLD = 30

def build_safe_features(dataframe, target_column=None):
    work_df = dataframe.copy()

    if target_column is not None and target_column in work_df.columns:
        work_df = work_df.dropna(subset=[target_column])

    drop_cols = []
    encode_cols = []

    id_keywords = ["id", "invoice", "stockcode", "stock_code", "customer"]

    for col in work_df.columns:
        if col == target_column:
            continue

        col_lower = col.lower().replace(" ", "").replace("_", "")

        if any(keyword in col_lower for keyword in id_keywords):
            drop_cols.append(col)
            continue

        series = work_df[col]

        if pd.api.types.is_numeric_dtype(series):
            continue

        nunique = series.nunique(dropna=True)
        avg_len = series.astype(str).str.len().mean()

        if nunique > HIGH_CARDINALITY_THRESHOLD or avg_len > TEXT_LENGTH_THRESHOLD:
            drop_cols.append(col)
        else:
            encode_cols.append(col)

    cols_to_drop = list(set(drop_cols + ([target_column] if target_column in work_df.columns else [])))
    X = work_df.drop(columns=cols_to_drop, errors="ignore").copy()

    encoders = {}
    for col in encode_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).fillna("Missing"))
            encoders[col] = le

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())

    remaining_object_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in remaining_object_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str).fillna("Missing"))
        encoders[col] = le

    if target_column is not None:
        y = work_df[target_column]
        return X, y, drop_cols, encoders

    return X, drop_cols, encoders


# ============================================================
# REGRESSION
# ============================================================
if problem_type == "Regression":
    if not target_col or target_col not in df.columns:
        st.warning("Please select a valid regression target in Target Configuration.")
        st.stop()

    X, y, dropped_cols, encoders = build_safe_features(df, target_col)

    if X.empty:
        st.error("No usable feature columns remain after filtering high-cardinality/text columns.")
        st.stop()

    if not pd.api.types.is_numeric_dtype(y):
        st.error("Regression target must be numeric.")
        st.stop()

    st.write(f"**Usable feature columns:** {len(X.columns)}")
    st.write(f"**Dropped high-cardinality/text columns:** {dropped_cols if dropped_cols else 'None'}")

    test_size = st.slider("Test Size (%)", 10, 40, 20, 5, key="reg_test_size")

    if st.button("Train Regression Model"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        save_to_session("trained_model", model)
        save_to_session("model_name", "Linear Regression")
        save_to_session("X_test", X_test)
        save_to_session("y_test", y_test)
        save_to_session("y_pred", y_pred)
        save_to_session("dropped_feature_columns", dropped_cols)

        st.success("Regression model trained successfully.")


# ============================================================
# CLASSIFICATION
# ============================================================
elif problem_type == "Classification":
    if not target_col or target_col not in df.columns:
        st.warning("Please select a valid classification target in Target Configuration.")
        st.stop()

    X, y, dropped_cols, encoders = build_safe_features(df, target_col)

    if X.empty:
        st.error("No usable feature columns remain after filtering high-cardinality/text columns.")
        st.stop()

    target_unique_count = y.nunique(dropna=True)
    target_sample_count = len(y)

    if target_unique_count > min(50, target_sample_count * 0.3):
        st.error(
            "The selected target has too many unique classes for classification. "
            "Please go back and choose a target with fewer repeated categories."
        )
        st.stop()

    y_encoder = LabelEncoder()
    y_encoded = y_encoder.fit_transform(y.astype(str).fillna("Missing"))

    class_counts = pd.Series(y_encoded).value_counts()
    rare_classes = class_counts[class_counts < 2]

    st.write(f"**Usable feature columns:** {len(X.columns)}")
    st.write(f"**Dropped high-cardinality/text columns:** {dropped_cols if dropped_cols else 'None'}")
    st.write(f"**Number of classes:** {len(y_encoder.classes_)}")
    st.write(f"**Classes with fewer than 2 samples:** {len(rare_classes)}")

    if len(rare_classes) > 0:
        st.warning(
            "The selected target has some rare classes. "
            "Stratified splitting will be disabled."
        )

    test_size = st.slider("Test Size (%)", 10, 40, 20, 5, key="cls_test_size")

    model_name = st.selectbox(
        "Select Classification Model",
        [
            "Decision Tree",
            "Bayes",
            "KNN",
            "SVM",
            "ANN"
        ],
        key="classification_model_choice"
    )

    if st.button("Train Classification Model"):
        use_stratify = None
        if class_counts.min() >= 2 and len(class_counts) > 1:
            use_stratify = y_encoded

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=test_size / 100,
            random_state=42,
            stratify=use_stratify
        )

        scaler = None
        if model_name in ["Bayes", "KNN", "SVM", "ANN"]:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if model_name == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
            saved_model_name = "Decision Tree Classifier"

        elif model_name == "Bayes":
            model = GaussianNB()
            saved_model_name = "Gaussian Naive Bayes"

        elif model_name == "KNN":
            model = KNeighborsClassifier()
            saved_model_name = "K-Nearest Neighbors Classifier"

        elif model_name == "SVM":
            model = SVC()
            saved_model_name = "Support Vector Machine Classifier"

        elif model_name == "ANN":
            model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
            saved_model_name = "Artificial Neural Network Classifier"

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        save_to_session("trained_model", model)
        save_to_session("model_name", saved_model_name)
        save_to_session("X_test", X_test)
        save_to_session("y_test", y_test)
        save_to_session("y_pred", y_pred)
        save_to_session("class_names", list(y_encoder.classes_))
        save_to_session("dropped_feature_columns", dropped_cols)

        st.success(f"{saved_model_name} trained successfully.")


# ============================================================
# CLUSTERING
# ============================================================
elif problem_type == "Clustering":
    X, dropped_cols, encoders = build_safe_features(df, None)

    if X.empty:
        st.error("No usable feature columns remain after filtering high-cardinality/text columns.")
        st.stop()

    st.write(f"**Usable feature columns:** {len(X.columns)}")
    st.write(f"**Dropped high-cardinality/text columns:** {dropped_cols if dropped_cols else 'None'}")

    cluster_model_name = st.selectbox(
        "Select Clustering Model",
        ["KMeans", "DBSCAN", "MeanShift", "MiniSom"],
        key="clustering_model_choice"
    )

    n_clusters = None
    eps = None
    min_samples = None
    som_x = None
    som_y = None

    if cluster_model_name == "KMeans":
        n_clusters = st.slider("Number of Clusters", 2, 10, 3, 1)

    elif cluster_model_name == "DBSCAN":
        eps = st.slider("DBSCAN eps", 0.1, 5.0, 0.5, 0.1)
        min_samples = st.slider("DBSCAN min_samples", 2, 20, 5, 1)

    elif cluster_model_name == "MiniSom":
        som_x = st.slider("SOM Grid Rows", 2, 10, 3, 1)
        som_y = st.slider("SOM Grid Columns", 2, 10, 3, 1)

    if st.button("Train Clustering Model"):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if cluster_model_name == "KMeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
            saved_model_name = "KMeans Clustering"

        elif cluster_model_name == "DBSCAN":
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_scaled)
            saved_model_name = "DBSCAN Clustering"

        elif cluster_model_name == "MeanShift":
            model = MeanShift()
            labels = model.fit_predict(X_scaled)
            saved_model_name = "MeanShift Clustering"

        elif cluster_model_name == "MiniSom":
            model = MiniSom(x=som_x, y=som_y, input_len=X_scaled.shape[1], sigma=1.0, learning_rate=0.5)
            model.random_weights_init(X_scaled)
            model.train_random(X_scaled, 100)

            labels = []
            for row in X_scaled:
                winner = model.winner(row)
                label = winner[0] * som_y + winner[1]
                labels.append(label)

            labels = np.array(labels)
            saved_model_name = "MiniSom Clustering"

        save_to_session("trained_model", model)
        save_to_session("model_name", saved_model_name)
        save_to_session("cluster_labels", labels)
        save_to_session("cluster_input_data", X_scaled)
        save_to_session("dropped_feature_columns", dropped_cols)

        st.success(f"{saved_model_name} trained successfully.")

else:
    st.error("Invalid problem type selected.")


if 'X' in locals():
    st.write("Feature columns used:", list(X.columns))
def recommend_imputation(df, numeric_cols, categorical_cols):
    total_rows = len(df)
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) if df.shape[0] * df.shape[1] > 0 else 0

    if missing_ratio == 0:
        return {"recommended": "No imputation needed", "reason": "The dataset currently has no missing values."}

    if total_rows < 1000:
        return {"recommended": "Simple Imputer", "reason": "The dataset is relatively small, so Simple Imputer is fast, stable, and easier to interpret."}

    if total_rows >= 1000 and len(numeric_cols) > 3:
        return {"recommended": "KNN Imputer", "reason": "The dataset has enough rows and multiple numeric features, so KNN can use neighboring patterns more effectively."}

    return {"recommended": "Iterative Imputer", "reason": "Iterative Imputer is suitable when feature relationships matter and more advanced estimation is preferred."}


def recommend_scaling(df, numeric_cols):
    if not numeric_cols:
        return {"recommended": "No scaling needed", "reason": "No numeric columns are available for scaling."}

    outlier_detected = False
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        if ((series < lower) | (series > upper)).sum() > 0:
            outlier_detected = True
            break

    if outlier_detected:
        return {"recommended": "RobustScaler", "reason": "Outliers are present in numeric features, so RobustScaler is more stable than StandardScaler or MinMaxScaler."}

    return {"recommended": "StandardScaler", "reason": "Numeric features appear reasonably well-behaved, so StandardScaler is a strong general-purpose choice."}


def recommend_problem_types(column_info):
    numeric_cols = column_info["numeric"]
    categorical_cols = column_info["categorical"]
    suggestions = []

    if len(numeric_cols) >= 2:
        suggestions.append("Regression is supported because numeric target candidates exist.")
    if len(categorical_cols) >= 1:
        suggestions.append("Classification is supported because categorical target candidates exist.")
    if len(numeric_cols) >= 2:
        suggestions.append("Clustering is supported because multiple numeric features are available.")
    if not suggestions:
        suggestions.append("This dataset may need more preprocessing before modeling.")

    return suggestions


def recommend_model(problem_type, feature_count, target_unique_count=None):
    if problem_type == "Regression":
        return {"recommended": "Multivariate Linear Regression", "reason": "This is the required regression model for the project and works well as a baseline for numeric targets."}

    if problem_type == "Classification":
        if target_unique_count is not None and target_unique_count <= 2:
            return {"recommended": "Decision Tree Classifier", "reason": "A Decision Tree is interpretable, supports rule extraction, and is suitable for binary classification."}
        return {"recommended": "Naive Bayes or Decision Tree", "reason": "Naive Bayes is efficient, while Decision Tree offers interpretability and rule generation."}

    if problem_type == "Clustering":
        return {"recommended": "K-Means", "reason": "K-Means is a strong first clustering method when numeric features are available and cluster count can be tuned."}

    return {"recommended": "No recommendation", "reason": "Problem type is not selected yet."}

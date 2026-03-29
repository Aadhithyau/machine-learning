import pandas as pd


def classify_columns(df):
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    categorical_cols = []
    text_like_cols = []

    for col in df.columns:
        if col in numeric_cols or col in datetime_cols:
            continue

        unique_count = df[col].nunique(dropna=True)
        total_count = len(df[col])
        avg_len = df[col].astype(str).str.len().mean()

        if unique_count <= max(20, total_count * 0.2):
            categorical_cols.append(col)
        elif avg_len > 30:
            text_like_cols.append(col)
        else:
            categorical_cols.append(col)

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "text_like": text_like_cols,
        "datetime": datetime_cols
    }


def missing_value_summary(df):
    summary = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isnull().sum().values,
        "Missing %": (df.isnull().sum().values / len(df)) * 100
    })
    return summary.sort_values(by="Missing Count", ascending=False).reset_index(drop=True)

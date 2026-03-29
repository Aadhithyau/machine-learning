from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import io
import base64
import json
import os

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   MaxAbsScaler, PowerTransformer, LabelEncoder)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, mean_absolute_error,
                              mean_squared_error, r2_score, silhouette_score)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from utils.helpers import classify_columns, missing_value_summary
from utils.recommendations import (recommend_imputation, recommend_scaling,
                                   recommend_problem_types, recommend_model)

app = Flask(__name__)
app.secret_key = 'ecommerce_ml_secret_2024'

# In-memory session store (since session can't hold DataFrames)
_store = {}

def store_set(sid, key, value):
    if sid not in _store:
        _store[sid] = {}
    _store[sid][key] = value

def store_get(sid, key, default=None):
    return _store.get(sid, {}).get(key, default)

def get_sid():
    if 'sid' not in session:
        session['sid'] = os.urandom(16).hex()
    return session['sid']

def df_to_json(df):
    return df.to_json(orient='split')

def json_to_df(j):
    return pd.read_json(io.StringIO(j), orient='split')

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#0f1117')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_b64

def setup_plot_style():
    plt.rcParams.update({
        'figure.facecolor': '#0f1117',
        'axes.facecolor': '#1a1f2e',
        'axes.edgecolor': '#2d3561',
        'text.color': '#e0e6f0',
        'axes.labelcolor': '#e0e6f0',
        'xtick.color': '#a0aec0',
        'ytick.color': '#a0aec0',
        'grid.color': '#2d3561',
        'grid.alpha': 0.5,
    })

HIGH_CARDINALITY_THRESHOLD = 50
TEXT_LENGTH_THRESHOLD = 30

def build_safe_features(df, target_column=None):
    work_df = df.copy()
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

    for col in encode_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).fillna("Missing"))

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna(X[col].median())

    for col in X.select_dtypes(include=["object", "category"]).columns.tolist():
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str).fillna("Missing"))

    if target_column is not None:
        y = work_df[target_column]
        return X, y, drop_cols
    return X, drop_cols


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload():
    sid = get_sid()
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    fname = file.filename.lower()
    try:
        if fname.endswith('.csv'):
            df = pd.read_csv(file)
        elif fname.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    store_set(sid, 'data', df_to_json(df))
    store_set(sid, 'uploaded_filename', file.filename)
    store_set(sid, 'processed_data', None)
    store_set(sid, 'preprocessing_summary', [])

    missing = missing_value_summary(df)
    dtypes = df.dtypes.reset_index()
    dtypes.columns = ['Column', 'Data Type']

    return jsonify({
        'success': True,
        'filename': file.filename,
        'rows': int(df.shape[0]),
        'cols': int(df.shape[1]),
        'missing_total': int(df.isnull().sum().sum()),
        'columns': df.columns.tolist(),
        'preview': df.head(10).fillna('').astype(str).values.tolist(),
        'preview_cols': df.columns.tolist(),
        'dtypes': dtypes.astype(str).values.tolist(),
        'missing_summary': missing.values.tolist(),
        'missing_cols': missing.columns.tolist()
    })


@app.route('/api/profiling', methods=['GET'])
def profiling():
    sid = get_sid()
    raw = store_get(sid, 'data')
    if raw is None:
        return jsonify({'error': 'No data uploaded'}), 400

    df = json_to_df(raw)
    col_info = classify_columns(df)
    numeric_cols = col_info['numeric']
    categorical_cols = col_info['categorical']
    text_like_cols = col_info['text_like']
    datetime_cols = col_info['datetime']

    stats = None
    if numeric_cols:
        stats_df = df[numeric_cols].describe().T.reset_index()
        stats_df.columns = ['Feature'] + list(stats_df.columns[1:])
        stats = {'cols': stats_df.columns.tolist(), 'data': stats_df.round(4).fillna('').astype(str).values.tolist()}

    imp_rec = recommend_imputation(df, numeric_cols, categorical_cols)
    scale_rec = recommend_scaling(df, numeric_cols)
    prob_recs = recommend_problem_types(col_info)

    return jsonify({
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'text_like': text_like_cols,
        'datetime': datetime_cols,
        'stats': stats,
        'regression_candidates': numeric_cols,
        'classification_candidates': categorical_cols,
        'imputation_rec': imp_rec,
        'scaling_rec': scale_rec,
        'problem_recs': prob_recs
    })


@app.route('/api/preprocessing/status', methods=['GET'])
def preprocessing_status():
    sid = get_sid()
    raw = store_get(sid, 'processed_data') or store_get(sid, 'data')
    if raw is None:
        return jsonify({'error': 'No data'}), 400

    df = json_to_df(raw)
    col_info = classify_columns(df)
    numeric_cols = col_info['numeric']
    categorical_cols = col_info['categorical']

    # skewness
    skew_rows = []
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(series) == 0:
            skew_rows.append({'Feature': col, 'Mean': 'N/A', 'Median': 'N/A', 'Skewness': 'N/A', 'Classification': 'N/A'})
        else:
            skew_val = series.skew()
            if -0.5 <= skew_val <= 0.5:
                cls = 'Symmetric'
            elif skew_val > 0.5:
                cls = 'Right-Skewed'
            else:
                cls = 'Left-Skewed'
            skew_rows.append({'Feature': col, 'Mean': round(series.mean(), 4),
                               'Median': round(series.median(), 4),
                               'Skewness': round(skew_val, 4), 'Classification': cls})

    summary = store_get(sid, 'preprocessing_summary', [])
    missing = missing_value_summary(df)
    imp_rec = recommend_imputation(df, numeric_cols, categorical_cols)
    scale_rec = recommend_scaling(df, numeric_cols)

    return jsonify({
        'rows': int(df.shape[0]),
        'cols': int(df.shape[1]),
        'missing': int(df.isnull().sum().sum()),
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'preview': df.head(5).fillna('').astype(str).values.tolist(),
        'preview_cols': df.columns.tolist(),
        'skewness': skew_rows,
        'missing_summary': missing.values.tolist(),
        'missing_cols': missing.columns.tolist(),
        'summary': summary,
        'imputation_rec': imp_rec,
        'scaling_rec': scale_rec,
    })


@app.route('/api/preprocessing/apply', methods=['POST'])
def preprocessing_apply():
    sid = get_sid()
    data = request.json
    action = data.get('action')

    raw = store_get(sid, 'processed_data') or store_get(sid, 'data')
    if raw is None:
        return jsonify({'error': 'No data'}), 400

    df = json_to_df(raw)
    col_info = classify_columns(df)
    numeric_cols = col_info['numeric']
    categorical_cols = col_info['categorical']
    summary = store_get(sid, 'preprocessing_summary', [])

    if action == 'impute':
        technique = data.get('technique', 'No Imputation')
        num_cols = data.get('numeric_cols', [])
        cat_cols = data.get('categorical_cols', [])

        if technique == 'No Imputation':
            return jsonify({'success': True, 'message': 'No imputation applied'})

        elif technique == 'Simple Imputer':
            num_strategy = data.get('numeric_strategy', 'mean')
            cat_strategy = data.get('categorical_strategy', 'most_frequent')
            fill_value = data.get('fill_value', 'Missing')

            if num_cols:
                imputer = SimpleImputer(strategy=num_strategy)
                df[num_cols] = imputer.fit_transform(df[num_cols])
            if cat_cols:
                if cat_strategy == 'constant':
                    imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
                else:
                    imputer = SimpleImputer(strategy=cat_strategy)
                df[cat_cols] = imputer.fit_transform(df[cat_cols])

            msg = f"Applied Simple Imputer | Numeric: {num_strategy} on {', '.join(num_cols) or 'None'} | Categorical: {cat_strategy} on {', '.join(cat_cols) or 'None'}"

        elif technique == 'KNN Imputer':
            k = data.get('knn_neighbors', 5)
            if not num_cols:
                return jsonify({'error': 'Select numeric columns for KNN'}), 400
            if cat_cols:
                imputer = SimpleImputer(strategy='most_frequent')
                df[cat_cols] = imputer.fit_transform(df[cat_cols])
            knn = KNNImputer(n_neighbors=k)
            df[num_cols] = knn.fit_transform(df[num_cols])
            msg = f"Applied KNN Imputer | k={k} | Numeric: {', '.join(num_cols)}"

        elif technique == 'Iterative Imputer':
            max_iter = data.get('max_iter', 10)
            random_state = data.get('random_state', 42)
            if not num_cols:
                return jsonify({'error': 'Select numeric columns for Iterative Imputer'}), 400
            if cat_cols:
                imputer = SimpleImputer(strategy='most_frequent')
                df[cat_cols] = imputer.fit_transform(df[cat_cols])
            iter_imp = IterativeImputer(max_iter=max_iter, random_state=int(random_state))
            df[num_cols] = iter_imp.fit_transform(df[num_cols])
            msg = f"Applied Iterative Imputer | max_iter={max_iter} | random_state={random_state} | Numeric: {', '.join(num_cols)}"

        if msg not in summary:
            summary.append(msg)

    elif action == 'scale':
        technique = data.get('technique', 'No Scaling')
        cols = data.get('columns', [])

        if technique == 'No Scaling':
            return jsonify({'success': True, 'message': 'No scaling applied'})
        if not cols:
            return jsonify({'error': 'Select columns to scale'}), 400
        if df[cols].isnull().sum().sum() > 0:
            return jsonify({'error': 'Apply imputation first before scaling'}), 400

        scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'MaxAbsScaler': MaxAbsScaler()
        }
        scaler = scalers.get(technique)
        if scaler:
            df[cols] = scaler.fit_transform(df[cols])
            msg = f"Applied {technique} on: {', '.join(cols)}"
            if msg not in summary:
                summary.append(msg)

    elif action == 'transform':
        technique = data.get('technique', 'No Transformation')
        cols = data.get('columns', [])

        if technique == 'No Transformation':
            return jsonify({'success': True, 'message': 'No transformation applied'})
        if not cols:
            return jsonify({'error': 'Select columns to transform'}), 400
        if df[cols].isnull().sum().sum() > 0:
            return jsonify({'error': 'Apply imputation first before transforming'}), 400

        skipped = []
        if technique == 'Log':
            for col in cols:
                if (df[col] < 0).any():
                    skipped.append(col)
                else:
                    df[col] = np.log1p(df[col])
        elif technique == 'Sqrt':
            for col in cols:
                if (df[col] < 0).any():
                    skipped.append(col)
                else:
                    df[col] = np.sqrt(df[col])
        elif technique == 'Box-Cox':
            valid = [c for c in cols if not (df[c] <= 0).any()]
            skipped = [c for c in cols if (df[c] <= 0).any()]
            if valid:
                t = PowerTransformer(method='box-cox')
                df[valid] = t.fit_transform(df[valid])
        elif technique == 'Yeo-Johnson':
            t = PowerTransformer(method='yeo-johnson')
            df[cols] = t.fit_transform(df[cols])

        msg = f"Applied {technique} transformation on: {', '.join(cols)}"
        if msg not in summary:
            summary.append(msg)

        if skipped:
            store_set(sid, 'processed_data', df_to_json(df))
            store_set(sid, 'preprocessing_summary', summary)
            return jsonify({'success': True, 'warning': f"Skipped for columns with invalid values: {', '.join(skipped)}"})

    store_set(sid, 'processed_data', df_to_json(df))
    store_set(sid, 'preprocessing_summary', summary)
    return jsonify({'success': True, 'message': 'Applied successfully'})


@app.route('/api/target/status', methods=['GET'])
def target_status():
    sid = get_sid()
    raw = store_get(sid, 'processed_data') or store_get(sid, 'data')
    if raw is None:
        return jsonify({'error': 'No data'}), 400

    df = json_to_df(raw)
    col_info = classify_columns(df)
    numeric_cols = col_info['numeric']
    categorical_cols = col_info['categorical']

    classification_candidates = [col for col in categorical_cols if 2 <= df[col].nunique(dropna=True) <= 15]

    saved_problem = store_get(sid, 'saved_problem_type', 'Regression')
    saved_target = store_get(sid, 'saved_target_column', None)

    return jsonify({
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'classification_candidates': classification_candidates,
        'saved_problem_type': saved_problem,
        'saved_target_column': saved_target
    })


@app.route('/api/target/save', methods=['POST'])
def target_save():
    sid = get_sid()
    data = request.json
    problem_type = data.get('problem_type', 'Regression')
    target_col = data.get('target_col', None)

    raw = store_get(sid, 'processed_data') or store_get(sid, 'data')
    if raw is None:
        return jsonify({'error': 'No data'}), 400
    df = json_to_df(raw)

    store_set(sid, 'saved_problem_type', problem_type)
    store_set(sid, 'saved_target_column', target_col)

    rec = recommend_model(problem_type, len(df.columns),
                          df[target_col].nunique() if target_col and target_col in df.columns else None)
    return jsonify({'success': True, 'recommendation': rec})


@app.route('/api/visualization', methods=['GET'])
def visualization():
    sid = get_sid()
    raw = store_get(sid, 'processed_data') or store_get(sid, 'data')
    if raw is None:
        return jsonify({'error': 'No data'}), 400

    df = json_to_df(raw)
    MAX_ROWS = 10000
    sampled = False
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=42)
        sampled = True

    id_keywords = ["id", "invoice", "stock", "code", "number", "no"]
    categorical_cols, numeric_cols = [], []
    for col in df.columns:
        col_lower = col.lower()
        if df[col].dtype == "object":
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            if any(kw in col_lower for kw in id_keywords):
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)

    setup_plot_style()
    MAX_CAT = 20
    charts = {'bar': [], 'histogram': [], 'boxplot': []}

    palette = ['#4f9cf9', '#7c6af5', '#36d399', '#f96a4f', '#f9c74f', '#f3722c']

    for col in categorical_cols:
        vc = df[col].astype(str).value_counts().head(MAX_CAT)
        if vc.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 4))
        bars = ax.bar(range(len(vc)), vc.values, color=palette[:len(vc)] if len(vc) <= len(palette) else palette * (len(vc)//len(palette)+1))
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels(vc.index, rotation=45, ha='right', fontsize=8)
        ax.set_title(f"Top Categories in {col}", color='#e0e6f0', pad=10, fontsize=11)
        ax.set_xlabel(col, color='#a0aec0')
        ax.set_ylabel("Count", color='#a0aec0')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        charts['bar'].append({'col': col, 'img': fig_to_base64(fig), 'unique': int(df[col].astype(str).nunique())})

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(series, bins=20, color='#4f9cf9', alpha=0.8, edgecolor='#0f1117')
        ax.set_title(f"Histogram of {col}", color='#e0e6f0', pad=10, fontsize=11)
        ax.set_xlabel(col, color='#a0aec0')
        ax.set_ylabel("Frequency", color='#a0aec0')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        charts['histogram'].append({'col': col, 'img': fig_to_base64(fig)})

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        outliers = int(((series < lower) | (series > upper)).sum())
        fig, ax = plt.subplots(figsize=(10, 3))
        bp = ax.boxplot(series, vert=False, patch_artist=True,
                        boxprops=dict(facecolor='#4f9cf9', color='#7c6af5'),
                        whiskerprops=dict(color='#a0aec0'),
                        capprops=dict(color='#a0aec0'),
                        medianprops=dict(color='#36d399', linewidth=2),
                        flierprops=dict(marker='o', color='#f96a4f', alpha=0.5))
        ax.set_title(f"Box Plot of {col}", color='#e0e6f0', pad=10, fontsize=11)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        charts['boxplot'].append({
            'col': col, 'img': fig_to_base64(fig),
            'median': round(float(series.median()), 4),
            'iqr': round(float(iqr), 4),
            'outliers': outliers
        })

    return jsonify({'charts': charts, 'sampled': sampled, 'sample_size': MAX_ROWS if sampled else len(df)})


@app.route('/api/modeling/train', methods=['POST'])
def modeling_train():
    sid = get_sid()
    data = request.json
    raw = store_get(sid, 'processed_data') or store_get(sid, 'data')
    if raw is None:
        return jsonify({'error': 'No data'}), 400

    df = json_to_df(raw)
    problem_type = store_get(sid, 'saved_problem_type', 'Regression')
    target_col = store_get(sid, 'saved_target_column', None)
    test_size = data.get('test_size', 20) / 100
    model_choice = data.get('model', 'Linear Regression')

    if problem_type == 'Regression':
        if not target_col or target_col not in df.columns:
            return jsonify({'error': 'Invalid target column'}), 400

        X, y, dropped = build_safe_features(df, target_col)
        if X.empty:
            return jsonify({'error': 'No usable features'}), 400
        if not pd.api.types.is_numeric_dtype(y):
            return jsonify({'error': 'Regression target must be numeric'}), 400

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        store_set(sid, 'y_test', y_test.tolist())
        store_set(sid, 'y_pred', y_pred.tolist())
        store_set(sid, 'model_name', 'Linear Regression')
        store_set(sid, 'dropped_feature_columns', dropped)

        return jsonify({'success': True, 'message': 'Linear Regression trained successfully',
                        'features': list(X.columns), 'dropped': dropped})

    elif problem_type == 'Classification':
        if not target_col or target_col not in df.columns:
            return jsonify({'error': 'Invalid target column'}), 400

        X, y, dropped = build_safe_features(df, target_col)
        if X.empty:
            return jsonify({'error': 'No usable features'}), 400

        target_unique_count = y.nunique()
        if target_unique_count > min(50, len(y) * 0.3):
            return jsonify({'error': 'Target has too many unique classes'}), 400

        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str).fillna("Missing"))
        class_counts = pd.Series(y_encoded).value_counts()
        use_stratify = y_encoded if class_counts.min() >= 2 and len(class_counts) > 1 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=use_stratify)

        scaler = None
        if model_choice in ['Bayes', 'KNN', 'SVM', 'ANN']:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        models = {
            'Decision Tree': (DecisionTreeClassifier(random_state=42), 'Decision Tree Classifier'),
            'Bayes': (GaussianNB(), 'Gaussian Naive Bayes'),
            'KNN': (KNeighborsClassifier(), 'K-Nearest Neighbors Classifier'),
            'SVM': (SVC(), 'Support Vector Machine Classifier'),
            'ANN': (MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42), 'Artificial Neural Network Classifier')
        }
        clf, model_name = models.get(model_choice, models['Decision Tree'])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        store_set(sid, 'y_test', y_test.tolist())
        store_set(sid, 'y_pred', y_pred.tolist())
        store_set(sid, 'model_name', model_name)
        store_set(sid, 'class_names', list(le.classes_))
        store_set(sid, 'dropped_feature_columns', dropped)

        return jsonify({'success': True, 'message': f'{model_name} trained successfully',
                        'features': list(X.columns), 'dropped': dropped})

    elif problem_type == 'Clustering':
        X, dropped = build_safe_features(df, None)
        if X.empty:
            return jsonify({'error': 'No usable features'}), 400

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        cluster_model = data.get('model', 'KMeans')
        n_clusters = data.get('n_clusters', 3)
        eps = data.get('eps', 0.5)
        min_samples = data.get('min_samples', 5)

        if cluster_model == 'KMeans':
            m = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = m.fit_predict(X_scaled)
            mname = 'KMeans Clustering'
        elif cluster_model == 'DBSCAN':
            m = DBSCAN(eps=eps, min_samples=min_samples)
            labels = m.fit_predict(X_scaled)
            mname = 'DBSCAN Clustering'
        elif cluster_model == 'MeanShift':
            m = MeanShift()
            labels = m.fit_predict(X_scaled)
            mname = 'MeanShift Clustering'
        else:
            return jsonify({'error': 'Unknown clustering model'}), 400

        store_set(sid, 'cluster_labels', labels.tolist())
        store_set(sid, 'cluster_input_data', X_scaled.tolist())
        store_set(sid, 'model_name', mname)
        store_set(sid, 'dropped_feature_columns', dropped)

        return jsonify({'success': True, 'message': f'{mname} trained successfully',
                        'features': list(X.columns), 'dropped': dropped})

    return jsonify({'error': 'Unknown problem type'}), 400


@app.route('/api/metrics', methods=['GET'])
def metrics():
    sid = get_sid()
    problem_type = store_get(sid, 'saved_problem_type', 'Regression')
    model_name = store_get(sid, 'model_name', 'Model')

    setup_plot_style()

    if problem_type == 'Regression':
        y_test = store_get(sid, 'y_test')
        y_pred = store_get(sid, 'y_pred')
        if y_test is None or y_pred is None:
            return jsonify({'error': 'No regression results found'}), 400

        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(mse ** 0.5)
        r2 = float(r2_score(y_test, y_pred))

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.scatter(y_test, y_pred, alpha=0.6, color='#4f9cf9', s=20)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='#36d399')
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Actual vs Predicted', color='#e0e6f0')
        plt.tight_layout()
        scatter_img = fig_to_base64(fig1)

        residuals = y_test - y_pred
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.scatter(y_pred, residuals, alpha=0.6, color='#7c6af5', s=20)
        ax2.axhline(y=0, linestyle='--', color='#36d399')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot', color='#e0e6f0')
        plt.tight_layout()
        residual_img = fig_to_base64(fig2)

        return jsonify({
            'problem_type': problem_type,
            'model_name': model_name,
            'mae': round(mae, 4), 'mse': round(mse, 4),
            'rmse': round(rmse, 4), 'r2': round(r2, 4),
            'scatter_img': scatter_img,
            'residual_img': residual_img
        })

    elif problem_type == 'Classification':
        y_test = store_get(sid, 'y_test')
        y_pred = store_get(sid, 'y_pred')
        class_names = store_get(sid, 'class_names')
        if y_test is None or y_pred is None:
            return jsonify({'error': 'No classification results found'}), 400

        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        recall = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(max(8, len(np.unique(y_test))), max(6, len(np.unique(y_test)))))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title('Confusion Matrix', color='#e0e6f0')
        plt.colorbar(im, ax=ax)
        tick_marks = np.arange(len(class_names) if class_names else len(np.unique(y_test)))
        if class_names:
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(class_names, rotation=60, fontsize=7)
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(class_names, fontsize=7)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else '#e0e6f0', fontsize=8)
        plt.tight_layout()
        cm_img = fig_to_base64(fig)

        result_df = pd.DataFrame({'Actual': y_test[:20], 'Predicted': y_pred[:20]})

        return jsonify({
            'problem_type': problem_type,
            'model_name': model_name,
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'cm_img': cm_img,
            'preview': result_df.astype(str).values.tolist(),
            'preview_cols': ['Actual', 'Predicted']
        })

    elif problem_type == 'Clustering':
        cluster_labels = store_get(sid, 'cluster_labels')
        cluster_input_data = store_get(sid, 'cluster_input_data')
        if cluster_labels is None or cluster_input_data is None:
            return jsonify({'error': 'No clustering results found'}), 400

        labels = np.array(cluster_labels)
        X_scaled = np.array(cluster_input_data)
        valid_mask = labels != -1
        valid_labels = labels[valid_mask]

        sil = None
        if len(np.unique(valid_labels)) >= 2 and valid_mask.sum() >= 2:
            sil = float(silhouette_score(X_scaled[valid_mask], valid_labels))

        counts = pd.Series(labels).value_counts().sort_index()
        cluster_df = [{'Cluster': str(k), 'Count': int(v)} for k, v in counts.items()]

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar([str(k) for k in counts.index], counts.values, color='#4f9cf9')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Count')
        ax.set_title('Cluster Distribution', color='#e0e6f0')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        bar_img = fig_to_base64(fig)

        return jsonify({
            'problem_type': problem_type,
            'model_name': model_name,
            'sil_score': round(sil, 4) if sil is not None else None,
            'cluster_data': cluster_df,
            'bar_img': bar_img,
            'has_noise': bool(-1 in labels)
        })

    return jsonify({'error': 'Unknown problem type'}), 400


@app.route('/api/summary', methods=['GET'])
def summary():
    sid = get_sid()
    raw_data = store_get(sid, 'data')
    processed_data = store_get(sid, 'processed_data')
    problem_type = store_get(sid, 'saved_problem_type', 'Not Selected')
    target_col = store_get(sid, 'saved_target_column', 'Not Selected')
    model_name = store_get(sid, 'model_name', 'Not Trained')
    dropped = store_get(sid, 'dropped_feature_columns', [])
    y_test = store_get(sid, 'y_test')
    y_pred = store_get(sid, 'y_pred')
    class_names = store_get(sid, 'class_names')
    cluster_labels = store_get(sid, 'cluster_labels')
    cluster_input = store_get(sid, 'cluster_input_data')

    raw_shape = (0, 0)
    proc_shape = (0, 0)
    summary_table = []
    proc_preview = []
    proc_cols = []
    usable_features = []

    if raw_data:
        df_raw = json_to_df(raw_data)
        raw_shape = df_raw.shape

    if processed_data:
        df_proc = json_to_df(processed_data)
        proc_shape = df_proc.shape
        summary_table = [
            {'Column': c, 'DType': str(df_proc[c].dtype),
             'Missing': int(df_proc[c].isnull().sum()),
             'Unique': int(df_proc[c].nunique(dropna=False))}
            for c in df_proc.columns
        ]
        proc_preview = df_proc.head(10).fillna('').astype(str).values.tolist()
        proc_cols = df_proc.columns.tolist()
        usable_features = [c for c in df_proc.columns if c != target_col]

    metrics_data = {}
    interpretation = ''

    if problem_type == 'Regression' and y_test and y_pred:
        yt, yp = np.array(y_test), np.array(y_pred)
        r2 = float(r2_score(yt, yp))
        metrics_data = {
            'MAE': round(float(mean_absolute_error(yt, yp)), 4),
            'MSE': round(float(mean_squared_error(yt, yp)), 4),
            'RMSE': round(float(mean_squared_error(yt, yp)**0.5), 4),
            'R²': round(r2, 4)
        }
        if r2 >= 0.90:
            interpretation = ('excellent', 'Excellent regression performance. The model explains most of the target variation.')
        elif r2 >= 0.75:
            interpretation = ('good', 'Good regression performance. The model is reasonably reliable.')
        elif r2 >= 0.50:
            interpretation = ('moderate', 'Moderate regression performance. Further feature engineering may improve results.')
        else:
            interpretation = ('weak', 'Weak regression performance. Model or features may need improvement.')

        conclusion = f"The regression model {model_name} was trained on target {target_col}. R² score: {r2:.4f}."

    elif problem_type == 'Classification' and y_test and y_pred:
        yt, yp = np.array(y_test), np.array(y_pred)
        acc = float(accuracy_score(yt, yp))
        metrics_data = {
            'Accuracy': round(acc, 4),
            'Precision': round(float(precision_score(yt, yp, average='weighted', zero_division=0)), 4),
            'Recall': round(float(recall_score(yt, yp, average='weighted', zero_division=0)), 4),
            'F1 Score': round(float(f1_score(yt, yp, average='weighted', zero_division=0)), 4)
        }
        if acc >= 0.90:
            interpretation = ('excellent', 'Excellent classification performance.')
        elif acc >= 0.75:
            interpretation = ('good', 'Good classification performance.')
        elif acc >= 0.60:
            interpretation = ('moderate', 'Moderate classification performance.')
        else:
            interpretation = ('weak', 'Weak classification performance.')
        conclusion = f"The model {model_name} achieved accuracy of {acc:.4f} on target {target_col}."

    elif problem_type == 'Clustering' and cluster_labels and cluster_input:
        labels = np.array(cluster_labels)
        X_s = np.array(cluster_input)
        valid_mask = labels != -1
        valid_labels = labels[valid_mask]
        sil = None
        if len(np.unique(valid_labels)) >= 2 and valid_mask.sum() >= 2:
            sil = float(silhouette_score(X_s[valid_mask], valid_labels))
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = int(np.sum(labels == -1)) if -1 in labels else 0
        metrics_data = {
            'Total Clusters': n_clusters,
            'Noise Points': noise,
            'Silhouette Score': round(sil, 4) if sil else 'N/A'
        }
        if sil and sil >= 0.50:
            interpretation = ('excellent', 'Strong cluster separation.')
        elif sil and sil >= 0.25:
            interpretation = ('good', 'Moderate cluster structure.')
        else:
            interpretation = ('moderate', 'Weak cluster separation. Tuning may improve quality.')
        conclusion = f"The model {model_name} produced {n_clusters} clusters."
        counts = pd.Series(labels).value_counts().sort_index()
        cluster_dist = [{'Cluster': str(k), 'Count': int(v)} for k, v in counts.items()]
        metrics_data['cluster_dist'] = cluster_dist

    else:
        conclusion = "Train a model first to generate the final summary."
        interpretation = ('info', 'No model trained yet.')

    summary_text = f"""Problem Type: {problem_type}
Target Column: {target_col}
Model Used: {model_name}
Raw Dataset Shape: {raw_shape[0]} x {raw_shape[1]}
Processed Dataset Shape: {proc_shape[0]} x {proc_shape[1]}
Dropped Feature Columns: {', '.join(dropped) if dropped else 'None'}"""

    return jsonify({
        'problem_type': problem_type,
        'target_col': str(target_col),
        'model_name': model_name,
        'raw_shape': list(raw_shape),
        'proc_shape': list(proc_shape),
        'summary_table': summary_table,
        'proc_preview': proc_preview,
        'proc_cols': proc_cols,
        'usable_features': usable_features,
        'dropped': dropped,
        'metrics': metrics_data,
        'interpretation': interpretation,
        'conclusion': conclusion,
        'summary_text': summary_text
    })


if __name__ == '__main__':
    app.run(debug=True, port=5050)

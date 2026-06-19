"""
Portable Lead Detection - ML Model Training Script
===================================================
Downloads a water quality dataset from Kaggle, auto-detects relevant columns,
trains a RandomForest model, and saves the model + metadata to backend/models/.

Usage:
  cd backend
  python train_model.py

Requirements in .env:
  KAGGLE_USERNAME=your_username
  KAGGLE_KEY=your_api_key
  KAGGLE_DATASET_SLUG=adityakadiwal/water-potability
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score, classification_report
import joblib

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")  # always load from backend/.env regardless of cwd
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Download dataset from Kaggle
# ---------------------------------------------------------------------------

def download_dataset():
    kaggle_username = os.getenv("KAGGLE_USERNAME", "").strip()
    kaggle_key = os.getenv("KAGGLE_KEY", "").strip()
    dataset_slug = os.getenv("KAGGLE_DATASET_SLUG", "adityakadiwal/water-potability").strip()

    if not kaggle_username or not kaggle_key:
        raise ValueError(
            "KAGGLE_USERNAME and KAGGLE_KEY are not set.\n"
            "Create backend/.env from backend/.env.example and add your Kaggle credentials.\n"
            "Get API key from: https://www.kaggle.com/settings -> API -> Create New Token"
        )

    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key

    print(f"[Kaggle] Downloading dataset: {dataset_slug}")

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_slug, path=str(DATA_DIR), unzip=True)

    print(f"[Kaggle] Dataset downloaded to: {DATA_DIR}")
    return DATA_DIR, dataset_slug


# ---------------------------------------------------------------------------
# Step 2: Load and select best CSV file
# ---------------------------------------------------------------------------

def load_dataset(data_dir):
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"\n[Data] Found CSV files: {[f.name for f in csv_files]}")

    water_keywords = ['tds', 'turbidity', 'potability', 'quality', 'lead', 'ph',
                      'water', 'safe', 'solids', 'hardness', 'conductivity']

    best_df = None
    best_name = None
    best_score = -1

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            cols_lower = [c.lower().replace(' ', '_') for c in df.columns]
            score = sum(1 for kw in water_keywords if any(kw in c for c in cols_lower))
            print(f"  {csv_file.name}: {df.shape[0]} rows x {df.shape[1]} cols | relevance={score}")
            print(f"    Columns: {list(df.columns)}")
            if score > best_score:
                best_score = score
                best_df = df
                best_name = csv_file.name
        except Exception as e:
            print(f"  SKIP {csv_file.name}: {e}")

    if best_df is None:
        raise ValueError("Could not load any CSV file from the downloaded dataset.")

    print(f"\n[Data] Using: {best_name}")
    return best_df, best_name


# ---------------------------------------------------------------------------
# Step 3: Auto-detect relevant columns
# ---------------------------------------------------------------------------

def detect_columns(df):
    """Map generic column types to actual dataset column names."""
    cols_lower_map = {c.lower().strip().replace(' ', '_'): c for c in df.columns}

    column_keywords = {
        'tds':              ['tds', 'solids', 'dissolved_solids', 'total_dissolved', 'conductance'],
        'turbidity':        ['turbidity', 'turb', 'ntu', 'clarity'],
        'ph':               ['ph', 'ph_value', 'acidity'],
        'hardness':         ['hardness', 'hard'],
        'chloramines':      ['chloramines', 'chloramine', 'chlorine'],
        'sulfate':          ['sulfate', 'sulphate'],
        'conductivity':     ['conductivity', 'electrical_conductivity'],
        'organic_carbon':   ['organic_carbon', 'toc', 'doc'],
        'trihalomethanes':  ['trihalomethanes', 'thm'],
        'temperature':      ['temperature', 'temp'],
        'heavy_metals':     ['heavy_metal', 'metals'],
        # Regression target
        'lead':             ['lead', 'lead_ppm', 'lead_concentration', 'lead_level', 'pb', 'lead_mg'],
        # Classification target
        'target':           ['potability', 'potable', 'safe', 'drinkable', 'quality',
                             'safety', 'label', 'class', 'is_safe', 'water_quality'],
    }

    detected = {}
    for feature_type, keywords in column_keywords.items():
        for kw in keywords:
            for col_lower, col_orig in cols_lower_map.items():
                if kw in col_lower:
                    detected[feature_type] = col_orig
                    break
            if feature_type in detected:
                break

    print(f"\n[Detect] Detected columns: {detected}")
    return detected


# ---------------------------------------------------------------------------
# Step 4: Prepare features and target
# ---------------------------------------------------------------------------

def prepare_features_and_target(df, detected_cols):
    # Determine model task
    if 'lead' in detected_cols:
        model_type = 'regression'
        target_col = detected_cols['lead']
        print(f"[Prepare] Task: REGRESSION — Target: '{target_col}' (lead concentration)")
    elif 'target' in detected_cols:
        model_type = 'classification'
        target_col = detected_cols['target']
        print(f"[Prepare] Task: CLASSIFICATION — Target: '{target_col}'")
    else:
        raise ValueError(
            "\nERROR: Dataset does not contain a valid target column.\n"
            "Expected one of: lead, lead_ppm, potability, safe, drinkable, quality, label\n"
            "Please choose a dataset with water quality or lead concentration labels.\n"
            "Do NOT use this script to create fake/random labels."
        )

    # Ordered list of feature types to use (priority order)
    feature_priority = [
        'tds', 'turbidity', 'ph', 'hardness', 'chloramines',
        'sulfate', 'conductivity', 'organic_carbon', 'trihalomethanes',
        'temperature', 'heavy_metals'
    ]

    feature_cols = []
    for fc in feature_priority:
        if fc in detected_cols and detected_cols[fc] != target_col:
            feature_cols.append(detected_cols[fc])

    # Fallback: use all numeric columns except target
    if not feature_cols:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_col]
        print(f"[Prepare] Fallback: using all numeric features: {feature_cols}")

    print(f"[Prepare] Feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"[Prepare] Target column: {target_col}")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Drop rows where target is NaN
    valid = y.notna()
    X, y = X[valid], y[valid]
    print(f"[Prepare] Rows after dropping NaN targets: {len(X)}")

    if model_type == 'classification':
        unique_vals = y.unique()
        print(f"[Prepare] Unique target values: {sorted(unique_vals)[:15]}")

        # Normalize string labels to 0 (safe) / 1 (unsafe)
        safe_labels = {'yes', 'safe', 'good', '1', 'potable', 'drinkable', 'clean'}
        if y.dtype == object or str(y.iloc[0]).lower() in safe_labels:
            y = y.map(lambda v: 0 if str(v).lower() in safe_labels else 1)

        y = y.astype(int)

        # Validate binary classification
        unique_int = y.unique()
        if len(unique_int) < 2:
            raise ValueError(f"Target column has only one class: {unique_int}. Cannot train classifier.")

        print(f"[Prepare] Class distribution:\n{y.value_counts().to_string()}")

    return X, y, feature_cols, model_type, target_col


# ---------------------------------------------------------------------------
# Step 5: Train model
# ---------------------------------------------------------------------------

def train_model(X, y, model_type):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if model_type == 'classification' else None
    )

    estimator = (
        RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        if model_type == 'regression'
        else RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, class_weight='balanced')
    )

    # Pipeline: median imputation for missing features + standard scaling + model
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', estimator),
    ])

    print(f"\n[Train] Training {estimator.__class__.__name__} on {len(X_train)} samples...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    if model_type == 'regression':
        score = r2_score(y_test, y_pred)
        metric_name = 'R2 Score'
        print(f"[Train] R2 Score on test set: {score:.4f}")
    else:
        score = accuracy_score(y_test, y_pred)
        metric_name = 'Accuracy'
        print(f"[Train] Accuracy on test set: {score:.4f}")
        print("[Train] Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Potable (0)', 'Non-Potable (1)']))

    return pipeline, score, metric_name


# ---------------------------------------------------------------------------
# Step 6: Save model artifacts
# ---------------------------------------------------------------------------

def save_model(pipeline, feature_cols, model_type, score, metric_name, dataset_name, detected_cols, dataset_slug):
    # Save pipeline (includes imputer + scaler + model)
    joblib.dump(pipeline, MODELS_DIR / "lead_model.pkl")

    # Build live-feature → training-feature mapping
    live_feature_mapping = {}
    for i, col in enumerate(feature_cols):
        col_lower = col.lower()
        if any(kw in col_lower for kw in ['tds', 'solids', 'dissolved']):
            live_feature_mapping['TDS'] = col
        elif any(kw in col_lower for kw in ['turbidity', 'turb']):
            live_feature_mapping['TURB'] = col

    missing_live = [c for c in feature_cols if c not in live_feature_mapping.values()]

    feature_info = {
        "feature_columns": feature_cols,
        "model_type": model_type,
        "live_feature_mapping": live_feature_mapping,
        "missing_from_live_hardware": missing_live,
    }
    with open(MODELS_DIR / "feature_columns.json", 'w') as f:
        json.dump(feature_info, f, indent=2)

    model_info = {
        "model_type": model_type,
        "estimator": "RandomForestRegressor" if model_type == 'regression' else "RandomForestClassifier",
        "dataset": dataset_name,
        "kaggle_slug": dataset_slug,
        "features_used": feature_cols,
        "live_features": list(live_feature_mapping.keys()),
        "missing_live_features": missing_live,
        "metric_name": metric_name,
        "score": round(float(score), 4),
        "trained_at": datetime.now().isoformat(),
        "n_features": len(feature_cols),
        "n_estimators": 150,
    }
    with open(MODELS_DIR / "model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"\n[Save] Model saved: {MODELS_DIR / 'lead_model.pkl'}")
    print(f"[Save] Feature info: {MODELS_DIR / 'feature_columns.json'}")
    print(f"[Save] Model info:   {MODELS_DIR / 'model_info.json'}")
    return model_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  Portable Lead Detection — ML Model Training")
    print("=" * 65)

    try:
        data_dir, dataset_slug = download_dataset()
        df, dataset_name = load_dataset(data_dir)
        detected_cols = detect_columns(df)
        X, y, feature_cols, model_type, target_col = prepare_features_and_target(df, detected_cols)
        pipeline, score, metric_name = train_model(X, y, model_type)
        model_info = save_model(pipeline, feature_cols, model_type, score, metric_name,
                                dataset_name, detected_cols, dataset_slug)

        print("\n" + "=" * 65)
        print("  Training Complete!")
        print(f"  Model   : {model_info['estimator']}")
        print(f"  Dataset : {model_info['dataset']}")
        print(f"  Features: {model_info['features_used']}")
        print(f"  {metric_name}: {score:.4f} ({score*100:.1f}%)")
        print(f"  Live hw : {model_info['live_features']}")
        if model_info['missing_live_features']:
            print(f"  Imputed : {model_info['missing_live_features']} (not from hardware)")
        print("=" * 65)

    except (ValueError, FileNotFoundError) as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

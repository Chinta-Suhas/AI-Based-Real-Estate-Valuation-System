# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from pathlib import Path

# ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(layout="centered", page_title="Real Estate Validator")
st.title("AI-based Real Estate Validation")

ROOT = Path(__file__).parent
MODEL_PATHS = [ROOT / "real_estate_model.pkl", ROOT / "model.pkl"]
DATASET_NAME = "india_housing_prices.csv"

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model():
    """Load a saved model. Supports legacy raw model files and new metadata dicts.

    Returns (model, feature_names, target_name) where feature_names and
    target_name may be None if not present in the saved file.
    """
    for p in MODEL_PATHS:
        if p.exists():
            try:
                data = joblib.load(p)
                # new format: metadata dict with 'model'
                if isinstance(data, dict) and 'model' in data:
                    st.info(f"Loaded model + metadata from {p.name}")
                    return data['model'], data.get('feature_names'), data.get('target_name')
                else:
                    # legacy plain model file
                    st.info(f"Loaded model from {p.name}")
                    return data, None, None
            except Exception as e:
                st.warning(f"Found {p.name} but failed to load: {e}")
    return None, None, None

def quick_train(df, target_name=None, n_estimators=100):
    # Choose target column heuristically (case-insensitive)
    if not target_name:
        cols_lower = [c.lower() for c in df.columns]
        if 'price' in cols_lower:
            target_name = df.columns[cols_lower.index('price')]
        else:
            target_name = df.columns[-1]

    # select numeric features only
    numeric = df.select_dtypes(include=[np.number]).copy()
    if target_name not in numeric.columns:
        # if target not numeric or missing, pick last numeric as target
        if numeric.shape[1] < 2:
            raise ValueError("Not enough numeric columns to train a model.")
        target_name = numeric.columns[-1]

    X = numeric.drop(columns=[target_name])
    y = numeric[target_name]

    # simple fillna
    X = X.fillna(X.median())
    y = y.fillna(y.median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    # save model + metadata for reliable future predictions
    metadata = {'model': model, 'feature_names': X.columns.tolist(), 'target_name': target_name}
    joblib.dump(metadata, ROOT / 'real_estate_model.pkl')
    # keep legacy plain model file for compatibility
    joblib.dump(model, ROOT / 'model.pkl')
    return model, X.columns.tolist(), target_name, mae

# ---- UI ----
st.sidebar.header("Project files & actions")
uploaded = st.sidebar.file_uploader("Upload CSV dataset (optional)", type=['csv'])
use_repo_dataset = (ROOT / DATASET_NAME).exists()

if use_repo_dataset and not uploaded:
    st.sidebar.write(f"Found `{DATASET_NAME}` in repo — will use it if needed.")

model = load_model()

if uploaded:
    st.write("### Using uploaded dataset")
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())
else:
    if use_repo_dataset:
        try:
            df = load_csv(ROOT / DATASET_NAME)
            st.write(f"### Loaded dataset `{DATASET_NAME}` from repo")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to load {DATASET_NAME}: {e}")
            df = None
    else:
        df = None

# If no model, try to train from dataset
if model is None and df is not None:
    st.info("No pretrained model found — training a quick model from the CSV. This runs once on deploy.")
    try:
        model, feature_names, target_name, mae = quick_train(df)
        st.success(f"Trained model and saved to `model.pkl`. Validation MAE: {mae:.2f}")
    except Exception as e:
        st.error(f"Auto-train failed: {e}")
        model = None

if model is None:
    st.warning("No model available. Upload a trained `real_estate_model.pkl` or include a dataset named 'india_housing_prices.csv' with numeric columns.")

# Make prediction UI if we have a model
if model is not None:
    st.header("Make a prediction")

    # If we trained, we have feature names from last training; try to load them
    # Try to infer numeric features from df if available
    if 'feature_names' in locals():
        inputs = feature_names
    else:
        if df is not None:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            # assume last numeric column is target and remove it
            if len(numeric) >= 2:
                inputs = numeric[:-1]
            else:
                inputs = numeric
        else:
            inputs = []

    if not inputs:
        st.error("Couldn't infer input features. Provide a dataset in repo or upload one.")
    else:
        st.write("Enter values for the features below and click Predict")
        user_vals = {}
        for f in inputs:
            col = df[f] if df is not None else None
            min_v = float(np.nanmin(col)) if col is not None else 0.0
            max_v = float(np.nanmax(col)) if col is not None else min_v + 1000.0
            mean_v = float(np.nanmean(col)) if col is not None else (min_v + max_v) / 2
            # create numeric input with sensible defaults
            user_vals[f] = st.number_input(f, value=mean_v, format="%.3f")

        if st.button("Predict"):
            X_pred = pd.DataFrame([user_vals])
            # ensure columns order
            X_pred = X_pred[inputs]
            try:
                pred = model.predict(X_pred)[0]
                st.metric(label="Predicted target", value=float(pred))
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.markdown("---")
st.write("If you want a custom UI (maps, images, or validation rules), reply here and I will update the app.")

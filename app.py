# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(layout="centered", page_title="AI Real Estate Valuation")

ROOT = Path(__file__).parent
MODEL_FILE = ROOT / "real_estate_model.pkl"   # metadata dict with 'model','feature_names','target_name'
DATA_FILE = ROOT / "india_housing_prices.csv"  # optional dataset for visuals

def load_model_metadata(path=MODEL_FILE):
    if path.exists():
        obj = joblib.load(path)
        if isinstance(obj, dict) and 'model' in obj:
            return obj
        # fallback to raw model (no metadata)
        return {'model': obj, 'feature_names': None, 'target_name': None}
    return None

meta = load_model_metadata()
model = meta['model'] if meta else None
feature_names = meta.get('feature_names') if meta else None
target_name = meta.get('target_name') if meta else None

st.title("🏠 AI-Based Real Estate Valuation System")
st.write("Enter the property details and click Predict.")

# Try to infer dataset for useful visualizations/options
df = None
if DATA_FILE.exists():
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception:
        df = None

# Build the input form
with st.form("predict_form"):
    st.subheader("Property details")
    # If we have feature_names from training, show inputs in that order (numeric only)
    inputs = {}
    if feature_names:
        for f in feature_names:
            # choose input widget type heuristically
            if df is not None and f in df.columns and pd.api.types.is_numeric_dtype(df[f]):
                min_v = float(df[f].min())
                max_v = float(df[f].max())
                mean_v = float(df[f].median())
                inputs[f] = st.number_input(f, value=mean_v, min_value=min_v, max_value=max_v)
            else:
                # default numeric input
                inputs[f] = st.number_input(f, value=0.0)
    else:
        # fallback sample inputs (customize to your dataset)
        inputs['area_sqft'] = st.number_input("Area (sq ft)", value=1200)
        inputs['bedrooms'] = st.slider("Bedrooms", 1, 10, 3)
        inputs['bathrooms'] = st.slider("Bathrooms", 1, 5, 2)
        inputs['age'] = st.number_input("Property Age", value=10)

    predict_btn = st.form_submit_button("Predict")

if predict_btn:
    if model is None:
        st.error("No model found. Export `real_estate_model.pkl` from the notebook first.")
    else:
        Xp = pd.DataFrame([inputs])
        # If model expects specific columns, reorder and fill missing
        if feature_names:
            Xp = Xp.reindex(columns=feature_names).fillna(0)
        try:
            pred = model.predict(Xp)[0]
            st.success(f"💰 Estimated price: ₹{pred:,.2f}")
        except Exception as e:
            st.error("Prediction failed: " + str(e))

# Visualizations
st.markdown("---")
st.subheader("Visualizations")

col1, col2 = st.columns(2)

# a) Feature importance
with col1:
    st.markdown("#### 📊 Feature importance")
    if model is None:
        st.info("Export a trained model to show feature importances.")
    else:
        fi_fig = None
        # try sklearn-style
        if hasattr(model, "feature_importances_"):
            try:
                fi = np.array(model.feature_importances_)
                names = feature_names if feature_names else [f"f{i}" for i in range(len(fi))]
                idx = np.argsort(fi)
                fig, ax = plt.subplots(figsize=(6, max(3, len(names)*0.25)))
                ax.barh(np.array(names)[idx], fi[idx], color='tab:blue')
                ax.set_xlabel("Importance")
                st.pyplot(fig)
            except Exception:
                st.info("Feature importance not available for this model.")
        else:
            st.info("Feature importance not available for this model.")

# b) Price distribution
with col2:
    st.markdown("#### 🏘️ Price distribution")
    if df is None or target_name is None:
        st.info("Provide dataset with price column (or export target_name in model metadata).")
    else:
        if target_name in df.columns:
            fig, ax = plt.subplots(figsize=(6,3))
            ax.hist(df[target_name].dropna(), bins=30, color='tab:green')
            ax.set_xlabel(target_name); ax.set_ylabel("count")
            st.pyplot(fig)
        else:
            st.info(f"Target `{target_name}` not found in dataset for distribution.")

# c) Market trends
st.markdown("#### 📈 Market trends")
if df is None:
    st.info("Provide dataset (e.g., india_housing_prices.csv) to show trends.")
else:
    # try year column then date-like then city/state
    if 'year' in df.columns and target_name in df.columns:
        trend = df.groupby('year')[target_name].mean().reset_index()
        st.line_chart(trend.set_index('year'))
    else:
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
        if date_col and target_name in df.columns:
            df['_dt'] = pd.to_datetime(df[date_col], errors='coerce').dt.to_period('M')
            trend = df.groupby('_dt')[target_name].mean()
            st.line_chart(trend)
        else:
            city_col = next((c for c in df.columns if c.lower() in ('city','town','location','state')), None)
            if city_col and target_name in df.columns:
                top = df.groupby(city_col)[target_name].mean().sort_values(ascending=False).head(8)
                st.bar_chart(top)
            else:
                st.info("No date/year/city column found to create market trends.")
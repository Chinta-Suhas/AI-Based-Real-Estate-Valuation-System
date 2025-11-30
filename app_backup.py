





# Enhanced Streamlit app for AI-Based Real Estate Valuation System
# Features:
#  - Predict tab with rich inputs, presets, amenities, and validation hints
#  - History tab with download / clear history
#  - Market Insights tab with interactive Plotly charts (feature importance, price distribution, trends)
#  - Robust model metadata loading (real_estate_model.pkl) with optional encoders/scalers support

import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

st.set_page_config(layout="wide", page_title="AI Real Estate Valuation", page_icon="🏠")

# ---------- Custom Theme: Classic Corporate (Blue & Neutrals) ----------
# Primary (Navy Blue): #003366 | Background: #FFFFFF / #F0F2F6 | Secondary BG: #CCCCCC
# Accent (Bright Orange): #FF6600 | Text: #111827
# Professional palette for trust, authority and reliability in real estate/finance
st.markdown("""
<style>
    /* Main background and text */
    .stApp {
        background: linear-gradient(135deg, #F0F2F6 0%, #FFFFFF 100%);
        color: #111827;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #003366;
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }
    
    /* Main headers and titles */
    h1 {
        color: #003366;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    h2 {
        color: #003366;
        font-weight: 600;
        font-size: 1.8rem;
    }
    h3 {
        color: #003366;
        font-weight: 600;
        font-size: 1.4rem;
    }
    
    /* Primary buttons (Predict, Get Price Estimate) */
    .stButton > button {
        background: linear-gradient(135deg, #FF6600 0%, #FF8533 100%);
        color: #FFFFFF !important;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(255, 102, 0, 0.2);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #E55A00 0%, #FF6600 100%);
        color: #FFFFFF !important;
        box-shadow: 0 6px 12px rgba(255, 102, 0, 0.4);
        transform: translateY(-2px);
    }
    .stButton > button p {
        color: #FFFFFF !important;
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background-color: #003366;
        color: #FFFFFF;
        border: 2px solid #003366;
        border-radius: 6px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stDownloadButton > button:hover {
        background-color: #004488;
        border-color: #004488;
        box-shadow: 0 4px 8px rgba(0, 51, 102, 0.3);
    }
    
    /* Form containers and cards */
    [data-testid="stForm"] {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 12px;
        border: 2px solid #CCCCCC;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border: 2px solid #CCCCCC;
        border-radius: 6px;
        background-color: #FFFFFF;
        padding: 0.6rem;
        transition: border-color 0.3s ease;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #003366;
        box-shadow: 0 0 0 3px rgba(0, 51, 102, 0.1);
    }
    
    /* Success boxes */
    .stSuccess {
        background: linear-gradient(90deg, #E6F7E6 0%, #F0FFF4 100%);
        border-left: 5px solid #48BB78;
        border-radius: 6px;
        padding: 1rem;
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(90deg, #E6F0FF 0%, #EBF8FF 100%);
        border-left: 5px solid #003366;
        border-radius: 6px;
        padding: 1rem;
    }
    
    /* Warning boxes */
    .stWarning {
        background: linear-gradient(90deg, #FFF5E6 0%, #FFFBF0 100%);
        border-left: 5px solid #FF6600;
        border-radius: 6px;
        padding: 1rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #F0F2F6;
        padding: 0.75rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border-radius: 8px;
        color: #003366;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #CCCCCC;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF6600 0%, #FF8533 100%);
        color: #FFFFFF;
        border-color: #FF6600;
        box-shadow: 0 4px 8px rgba(255, 102, 0, 0.3);
    }
    
    /* Metric styling */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #CCCCCC;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    [data-testid="stMetricLabel"] {
        color: #003366;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #FF6600;
        font-weight: 700;
    }
    
    /* Dataframe styling */
    .dataframe {
        border: 2px solid #CCCCCC !important;
        border-radius: 8px;
    }
    .dataframe thead th {
        background-color: #003366 !important;
        color: #FFFFFF !important;
        font-weight: 600;
    }
    
    /* Caption text */
    .stCaption {
        color: #6B7280;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).parent
MODEL_FILE = ROOT / "real_estate_model.pkl"   # metadata dict with 'model','feature_names','target_name','encoders','scalers' (optional)
DATA_FILE = ROOT / "india_housing_prices.csv"  # optional dataset for visuals / defaults

# ---------- Utilities ----------
def load_model_metadata(path=MODEL_FILE):
    if path.exists():
        try:
            meta = joblib.load(path)
            if isinstance(meta, dict) and 'model' in meta:
                return meta
            else:
                return {'model': meta, 'feature_names': None, 'target_name': None}
        except Exception as e:
            st.warning(f"Failed to load model metadata: {e}")
            return None
    return None

def fmt_currency(x):
    try:
        return f"₹{float(x):,.2f}"
    except Exception:
        return str(x)

def df_median_or_default(df, col, default=0):
    try:
        if col in df.columns:
            return float(df[col].median())
    except Exception:
        pass
    return default

def safe_reindex_fill(X, feature_names):
    if feature_names is None:
        return X.fillna(0)
    return X.reindex(columns=feature_names, fill_value=0).fillna(0)

def encode_categoricals(X, encoders):
    # encoders: dict mapping column -> fitted LabelEncoder (or similar with transform)
    if not encoders:
        return X
    X = X.copy()
    for col, enc in encoders.items():
        if col in X.columns:
            try:
                X[col] = enc.transform(X[col].astype(str))
            except Exception:
                # fallback: try to map known classes to codes; unknown -> -1
                try:
                    mapping = {c: i for i, c in enumerate(enc.classes_)}
                    X[col] = X[col].map(mapping).fillna(-1)
                except Exception:
                    pass
    return X

# ---------- Load model & dataset ----------
meta = load_model_metadata()
model = meta['model'] if meta else None
feature_names = meta.get('feature_names') if meta else None
target_name = meta.get('target_name') if meta else 'Price_in_Lakhs'
encoders = meta.get('encoders') if meta else None
scalers = meta.get('scalers') if meta else None

df = None
if DATA_FILE.exists():
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        st.warning(f"Failed to load dataset for visuals/defaults: {e}")
        df = None

# ---------- Session state ----------
if 'pred_history' not in st.session_state:
    st.session_state['pred_history'] = []  # list of dicts

# ---------- Top header ----------
st.markdown("""
# 🏠 AI Real Estate Valuation System
_Data-driven property price estimates with interactive visual insights_
""")

tabs = st.tabs(["Predict", "History", "Market Insights"])

# ---------- PREDICT TAB ----------
with tabs[0]:
    col_main, col_side = st.columns([3, 1])
    with col_main:
        st.markdown("### Property Details")
        with st.form("predict_form", clear_on_submit=False):
            # If we have training feature names, try to present the most common ones
            # Map friendly labels to internal features where possible
            # Common expected features used in notebook
            # Provide min/max/default hints from dataset if available
            def hint_range(col):
                if df is not None and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    mn = float(df[col].min())
                    mx = float(df[col].max())
                    return mn, mx
                return None, None

            # Location: State & City
            if df is not None and 'State' in df.columns:
                states = sorted(df['State'].dropna().unique().tolist())
                state = st.selectbox("State", states, index=states.index(states[0]) if states else 0, help="Select the state where the property is located")
            else:
                state = st.text_input("State", "", help="Enter state name (e.g., Maharashtra)")

            if df is not None and 'City' in df.columns:
                cities = sorted(df['City'].dropna().unique().tolist())
                city = st.selectbox("City", cities, index=0, help="Select the city where the property is located")
            else:
                city = st.text_input("City", "", help="Enter city / locality")

            # Size (sq ft)
            mn, mx = hint_range('Size_in_SqFt')
            default_size = int(df_median_or_default(df, 'Size_in_SqFt', 1000))
            if mn is not None:
                size = st.number_input("Size (Sq.Ft)", min_value=int(max(10, mn)), max_value=int(max(1000, mx)), value=default_size, step=10, help="Total built-up area in square feet")
                st.caption(f"Typical range: {int(mn)} - {int(mx)} sq.ft")
            else:
                size = st.number_input("Size (Sq.Ft)", min_value=100, max_value=10000, value=default_size, step=10, help="Total built-up area in square feet")

            # BHK
            bhk_default = int(df_median_or_default(df, 'BHK', 2))
            bhk = st.slider("BHK (Bedrooms)", 1, 10, bhk_default, help="Number of bedrooms (e.g., 1,2,3...)")

            # Property type
            prop_types = df['Property_Type'].dropna().unique().tolist() if (df is not None and 'Property_Type' in df.columns) else ['Apartment', 'Independent House', 'Villa']
            prop_type = st.selectbox("Property Type", prop_types, index=0, help="Select property type")

            # Furnished status
            furn_options = df['Furnished_Status'].dropna().unique().tolist() if (df is not None and 'Furnished_Status' in df.columns) else ['Unfurnished', 'Semi-Furnished', 'Fully-Furnished']
            furnished = st.selectbox("Furnished Status", furn_options, help="Furnishing level of the property")

            # Floor info
            floor_default = int(df_median_or_default(df, 'Floor_No', 1))
            total_floors_default = int(df_median_or_default(df, 'Total_Floors', 5))
            floor_no = st.number_input("Floor Number", min_value=0, max_value=200, value=floor_default, help="Floor number where the unit is located (1 = ground/first floor)")
            total_floors = st.number_input("Total Floors in Building", min_value=1, max_value=200, value=total_floors_default, help="Total floors in the building")

            # Year built / Age
            year_default = int(df_median_or_default(df, 'Year_Built', 2015))
            year_built = st.number_input("Year Built", min_value=1900, max_value=datetime.now().year, value=year_default, help="Year the property was constructed")
            age = datetime.now().year - year_built

            # Parking, Security
            parking_opts = ['No', 'Yes'] if df is None or 'Parking_Space' not in df.columns else sorted(df['Parking_Space'].dropna().unique().tolist())
            parking = st.selectbox("Parking Space", parking_opts, index=0, help="Is parking available?")
            security_opts = ['No', 'Yes'] if df is None or 'Security' not in df.columns else sorted(df['Security'].dropna().unique().tolist())
            security = st.selectbox("Security", security_opts, index=0, help="Security available in the building/complex?")

            # Nearby facilities
            nearby_schools_default = int(df_median_or_default(df, 'Nearby_Schools', 0))
            nearby_hospitals_default = int(df_median_or_default(df, 'Nearby_Hospitals', 0))
            nearby_schools = st.number_input("Nearby Schools (count)", min_value=0, max_value=100, value=nearby_schools_default, help="Number of schools near the property")
            nearby_hospitals = st.number_input("Nearby Hospitals (count)", min_value=0, max_value=100, value=nearby_hospitals_default, help="Number of hospitals/clinics nearby")

            # Amenities multi-select (if dataset contains amenities, else show common ones)
            amenities_list = []
            if df is not None and 'Amenities' in df.columns:
                # extract distinct amenity tokens
                try:
                    tokens = df['Amenities'].dropna().astype(str).str.split(',').explode().str.strip()
                    amenities_list = sorted(tokens.unique().tolist())
                except Exception:
                    amenities_list = ['Swimming Pool', 'Gym', 'Park', 'Security', 'Power Backup', 'Lift', 'Club House', 'Intercom', 'Play Area', 'Visitor Parking']
            else:
                amenities_list = ['Swimming Pool', 'Gym', 'Park', 'Security', 'Power Backup', 'Lift', 'Club House', 'Intercom', 'Play Area', 'Visitor Parking', 'Shopping Center']

            chosen_amenities = st.multiselect("Amenities (select all that apply)", amenities_list, default=[], help="Select amenities available with the property")

            # Hidden / derived features (Price_per_SqFt, Price_per_BHK, Area_per_BHK) left for model to compute or user can fill if desired
            st.caption("Tip: If your model expects derived features (Price_per_SqFt, Area_per_BHK), the app will compute or attempt to align features automatically.")

            predict_button = st.form_submit_button("Get Price Estimate")

    # Right column: presets and quick examples
    with col_side:
        st.markdown("### ⚡ Quick Presets")
        st.info("Try these sample presets to see typical estimates.")
        presets = {
            "Luxury Apartment - Mumbai": {
                "State": "Maharashtra", "City": "Mumbai", "Size_in_SqFt": 1500, "BHK": 3, "Property_Type": "Apartment", "Furnished_Status": "Fully-Furnished", "Floor_No": 12, "Total_Floors": 20, "Year_Built": 2018, "Parking_Space": "Yes", "Nearby_Schools": 5, "Nearby_Hospitals": 3, "Amenities": ["Swimming Pool", "Gym", "Security"]
            },
            "Budget Studio - Pune": {
                "State": "Maharashtra", "City": "Pune", "Size_in_SqFt": 420, "BHK": 1, "Property_Type": "Apartment", "Furnished_Status": "Semi-Furnished", "Floor_No": 2, "Total_Floors": 6, "Year_Built": 2012, "Parking_Space": "No", "Nearby_Schools": 2, "Nearby_Hospitals": 1, "Amenities": ["Lift", "Power Backup"]
            },
            "Spacious Villa - Bangalore": {
                "State": "Karnataka", "City": "Bengaluru", "Size_in_SqFt": 3200, "BHK": 4, "Property_Type": "Villa", "Furnished_Status": "Semi-Furnished", "Floor_No": 1, "Total_Floors": 2, "Year_Built": 2010, "Parking_Space": "Yes", "Nearby_Schools": 4, "Nearby_Hospitals": 2, "Amenities": ["Garden", "Security", "Parking"]
            }
        }
        for name, p in presets.items():
            if st.button(name):
                # apply preset values to the form by setting variables in session_state and reloading
                st.session_state['preset_values'] = p
                st.experimental_rerun()

        st.markdown("---")
        st.markdown("### ℹ️ Notes")
        st.write("- Use presets if unsure about numeric encodings.")
        st.write("- If the app warns about missing encoders, export encoders from training notebook and include them in model metadata.")

    # Apply preset values if available (populate form defaults)
    if 'preset_values' in st.session_state and st.session_state['preset_values']:
        pv = st.session_state['preset_values']
        # Note: Because Streamlit forms don't support programmatic set of widget values easily without session-state bindings,
        # we keep a simple UX: after pressing preset button, the app reloads and users can press Predict (values will be visible if session-state bound).
        # For full programmatic population, each widget must use st.session_state keys when created.

    # Handle prediction
    if predict_button:
        # Build input dict matching model features where possible
        input_row = {
            'State': state,
            'City': city,
            'Size_in_SqFt': float(size),
            'BHK': int(bhk),
            'Property_Type': prop_type,
            'Furnished_Status': furnished,
            'Floor_No': int(floor_no),
            'Total_Floors': int(total_floors),
            'Year_Built': int(year_built),
            'Age_of_Property': int(age),
            'Parking_Space': parking,
            'Security': security,
            'Nearby_Schools': int(nearby_schools),
            'Nearby_Hospitals': int(nearby_hospitals),
            'Amenities': ",".join(chosen_amenities) if chosen_amenities else ""
        }

        X_pred = pd.DataFrame([input_row])

        # If model expects derived columns, compute common ones
        if 'Price_per_SqFt' in (feature_names or []) and 'Price_in_Lakhs' not in X_pred.columns:
            # can't compute without price; leave missing
            pass
        if 'Area_per_BHK' in (feature_names or []):
            try:
                X_pred['Area_per_BHK'] = X_pred['Size_in_SqFt'] / X_pred['BHK']
            except Exception:
                X_pred['Area_per_BHK'] = 0

        if 'Amenity_Count' in (feature_names or []):
            X_pred['Amenity_Count'] = X_pred['Amenities'].apply(lambda x: len([t for t in x.split(',') if t.strip()]))


        # Apply label encoders if present
        if encoders:
            try:
                X_pred = encode_categoricals(X_pred, encoders)
            except Exception as e:
                st.warning(f"Failed to apply encoders: {e}")

        # Ensure columns order and fill missing
        X_eval = safe_reindex_fill(X_pred, feature_names)

        if model is None:
            st.error("Model not found. Export 'real_estate_model.pkl' from the training notebook to the app folder.")
        else:
            try:
                pred = model.predict(X_eval)[0]
                # display value with currency and explicit 'Lakhs' unit
                st.success(f"💰 Estimated Price: {fmt_currency(pred)} Lakhs")
                # Optional: show modest summary
                st.markdown("#### Prediction Summary")
                st.write(f"- Location: {city}, {state}")
                st.write(f"- Size: {size} sq.ft | {bhk} BHK | Floor {floor_no} of {total_floors}")
                st.write(f"- Amenities: {', '.join(chosen_amenities) if chosen_amenities else 'None'}")

                # Save into history
                hist_entry = input_row.copy()
                hist_entry.update({
                    'Predicted_Price': float(pred),
                    'Predicted_Price_Lakhs': f"{fmt_currency(pred)} Lakhs",
                    'Predicted_At': datetime.now().isoformat()
                })
                st.session_state['pred_history'].insert(0, hist_entry)

                # Show feature importance snippet (if available)
                if hasattr(model, "feature_importances_") and feature_names:
                    fi = np.array(model.feature_importances_)
                    fig = px.bar(x=feature_names, y=fi, labels={'x': 'Feature', 'y': 'Importance'},
                                 title="Model Feature Importances",
                                 color_discrete_sequence=['#FF6600'])
                    fig.update_layout(
                        plot_bgcolor='#FFFFFF',
                        paper_bgcolor='#F0F2F6',
                        title_font=dict(size=20, color='#003366', family='Arial Black'),
                        font=dict(color='#111827')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model type.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------- HISTORY TAB ----------
with tabs[1]:
    st.markdown("### 🔁 Prediction History")
    hist = pd.DataFrame(st.session_state['pred_history'])
    if hist.empty:
        st.info("No predictions made yet in this session. Use the Predict tab to estimate property prices.")
    else:
        st.dataframe(hist, use_container_width=True)
        # Download CSV
        csv = hist.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download history (CSV)", data=csv, file_name="prediction_history.csv", mime="text/csv")
        if st.button("Clear history"):
            st.session_state['pred_history'] = []
            st.success("Prediction history cleared.")
            st.experimental_rerun()

# ---------- MARKET INSIGHTS TAB ----------
with tabs[2]:
    st.markdown("### 📊 Market Insights")
    if df is None:
        st.info("No dataset available for market insights. Place 'india_housing_prices.csv' in the app folder to enable charts.")
    else:
        # Price distribution
        if target_name in df.columns:
            fig1 = px.histogram(df, x=target_name, nbins=50, title="Price Distribution", 
                               labels={target_name: "Price (Lakhs)"},
                               color_discrete_sequence=['#003366'])
            fig1.update_layout(
                bargap=0.05,
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#F0F2F6',
                title_font=dict(size=20, color='#003366', family='Arial Black'),
                font=dict(color='#111827')
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info(f"Target column '{target_name}' not found in dataset; price distribution unavailable.")

        # Market trends by year if possible
        if 'Year_Built' in df.columns and target_name in df.columns:
            # average price by Year_Built as proxy trend (not ideal but useful)
            trend = df.groupby('Year_Built')[target_name].mean().reset_index().sort_values('Year_Built')
            fig2 = px.line(trend, x='Year_Built', y=target_name, title='Average Price by Year Built', 
                          labels={'Year_Built': 'Year Built', target_name: 'Avg Price (Lakhs)'},
                          color_discrete_sequence=['#FF6600'])
            fig2.update_layout(
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#F0F2F6',
                title_font=dict(size=20, color='#003366', family='Arial Black'),
                font=dict(color='#111827')
            )
            fig2.update_traces(line=dict(width=3))
            st.plotly_chart(fig2, use_container_width=True)
        elif 'Listing_Date' in df.columns and target_name in df.columns:
            try:
                df['_dt'] = pd.to_datetime(df['Listing_Date'], errors='coerce').dt.to_period('M').dt.to_timestamp()
                trend = df.groupby('_dt')[target_name].mean().reset_index()
                fig2 = px.line(trend, x='_dt', y=target_name, title='Average Price Over Time', 
                              labels={'_dt': 'Date', target_name: 'Avg Price (Lakhs)'},
                              color_discrete_sequence=['#FF6600'])
                fig2.update_layout(
                    plot_bgcolor='#FFFFFF',
                    paper_bgcolor='#F0F2F6',
                    title_font=dict(size=20, color='#003366', family='Arial Black'),
                    font=dict(color='#111827')
                )
                fig2.update_traces(line=dict(width=3))
                st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                st.info("No usable date/time column for trend analysis.")
        else:
            # fallback: average price by city or state
            if 'City' in df.columns and target_name in df.columns:
                city_avg = df.groupby('City')[target_name].mean().sort_values(ascending=False).head(12).reset_index()
                fig3 = px.bar(city_avg, x='City', y=target_name, title='Top Cities by Average Price', 
                             labels={target_name: 'Avg Price (Lakhs)'},
                             color_discrete_sequence=['#FF6600'])
                fig3.update_layout(
                    plot_bgcolor='#FFFFFF',
                    paper_bgcolor='#F0F2F6',
                    title_font=dict(size=20, color='#003366', family='Arial Black'),
                    font=dict(color='#111827')
                )
                st.plotly_chart(fig3, use_container_width=True)

        # Feature importance (model-level)
        st.markdown("#### Model Feature Importance (if available)")
        if model is not None and hasattr(model, "feature_importances_") and feature_names:
            fi = np.array(model.feature_importances_)
            fi_df = pd.DataFrame({'feature': feature_names, 'importance': fi}).sort_values('importance', ascending=True)
            fig4 = go.Figure(go.Bar(
                x=fi_df['importance'], 
                y=fi_df['feature'], 
                orientation='h',
                marker=dict(
                    color='#FF6600',
                    line=dict(color='#003366', width=1)
                )
            ))
            fig4.update_layout(
                title="Feature Importances",
                xaxis_title="Importance",
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#F0F2F6',
                title_font=dict(size=20, color='#003366', family='Arial Black'),
                font=dict(color='#111827')
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Feature importance not available for the current model. Consider using tree-based models (RandomForest/XGBoost) or export feature_importances_ in metadata.")

# ---------- Footer / Help ----------
st.markdown("---")
st.markdown("Developed for the AI-Based Real Estate Valuation System • Ensure your trained model metadata file `real_estate_model.pkl` (joblib) is present in the app folder with keys: `model`, `feature_names`, `target_name`. Export encoders/scalers under `encoders`/`scalers` to preserve preprocessing at inference.")
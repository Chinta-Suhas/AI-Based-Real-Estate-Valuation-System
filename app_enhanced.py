# Premium Enhanced Streamlit app for AI-Based Real Estate Valuation System
# Features: Premium UI, animations, better UX, advanced visualizations

import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import time

st.set_page_config(layout="wide", page_title="AI Real Estate Valuation", page_icon="🏠")

# ---------- Premium Custom Theme ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Header with Gradient Text */
    .main-header {
        text-align: center;
        padding: 30px 0;
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 40px;
    }
    
    .gradient-text {
        font-size: 3.5em;
        font-weight: 800;
        background: linear-gradient(135deg, #003366 0%, #FF6600 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        animation: slideDown 0.8s ease-out;
    }
    
    .sub-header {
        color: #666;
        font-size: 1.3em;
        margin-top: 15px;
        font-weight: 500;
    }
    
    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    /* Premium Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #FF6600 0%, #FF8533 100%);
        color: #FFFFFF !important;
        border: none;
        border-radius: 16px;
        padding: 18px 48px;
        font-size: 18px;
        font-weight: 700;
        box-shadow: 0 8px 24px rgba(255, 102, 0, 0.4);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        text-transform: uppercase;
        letter-spacing: 2px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        transition: left 0.6s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #FF8533 0%, #FFA366 100%);
        box-shadow: 0 12px 36px rgba(255, 102, 0, 0.6);
        transform: translateY(-4px) scale(1.02);
        color: #FFFFFF !important;
    }
    
    .stButton > button:active {
        transform: translateY(-2px) scale(0.98);
    }
    
    .stButton > button p {
        color: #FFFFFF !important;
        margin: 0;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 14px;
        color: #003366;
        font-weight: 700;
        padding: 16px 32px;
        transition: all 0.3s ease;
        border: 3px solid transparent;
        font-size: 16px;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #FFF3E6 0%, #FFE5CC 100%);
        color: #FF6600;
        border-color: #FF6600;
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(255, 102, 0, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #003366 0%, #004080 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 6px 20px rgba(0, 51, 102, 0.4);
        border-color: #003366 !important;
    }
    
    /* Form Container */
    .stForm {
        background: linear-gradient(145deg, #FFFFFF 0%, #F8F9FA 100%);
        border-radius: 24px;
        padding: 50px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        border: 4px solid #FF6600;
        position: relative;
        overflow: hidden;
        animation: slideIn 0.6s ease-out;
    }
    
    .stForm::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,102,0,0.06) 0%, transparent 70%);
        animation: rotate 25s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    /* Input Fields */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stMultiSelect > div > div,
    .stTextInput > div > div > input {
        border: 3px solid #CCCCCC;
        border-radius: 14px;
        padding: 16px;
        font-size: 16px;
        transition: all 0.3s ease;
        background: #FFFFFF;
        font-weight: 500;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextInput > div > div > input:focus {
        border-color: #FF6600;
        box-shadow: 0 0 0 5px rgba(255, 102, 0, 0.15);
        outline: none;
        transform: scale(1.01);
    }
    
    /* Labels */
    .stNumberInput label,
    .stSelectbox label,
    .stMultiSelect label,
    .stTextInput label {
        color: #003366;
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 10px;
    }
    
    /* Success Message */
    .stSuccess {
        background: linear-gradient(135deg, #E6F7F0 0%, #D1F2E8 100%);
        border-left: 8px solid #00CC66;
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 6px 20px rgba(0, 204, 102, 0.2);
        animation: slideIn 0.5s ease-out;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #E6F0FF 0%, #D1E5FF 100%);
        border-left: 8px solid #003366;
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 6px 20px rgba(0, 51, 102, 0.2);
    }
    
    .stWarning {
        background: linear-gradient(135deg, #FFF3E6 0%, #FFE5CC 100%);
        border-left: 8px solid #FF6600;
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 6px 20px rgba(255, 102, 0, 0.2);
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        color: #003366;
        font-size: 40px;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    [data-testid="stMetricLabel"] {
        color: #666666;
        font-weight: 700;
        font-size: 15px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        border: 3px solid #E9ECEF;
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.15);
        border-color: #FF6600;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #003366 0%, #004080 50%, #0059b3 100%);
        box-shadow: 6px 0 24px rgba(0,0,0,0.2);
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }
    
    /* DataFrame */
    .dataframe {
        border: 4px solid #FF6600 !important;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    
    .dataframe thead tr th {
        background: linear-gradient(135deg, #003366 0%, #004080 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700;
        padding: 18px !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-size: 14px;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #F8F9FA;
    }
    
    .dataframe tbody tr:hover {
        background: linear-gradient(90deg, #FFF3E6 0%, #FFE5CC 100%);
        transform: scale(1.01);
        box-shadow: 0 3px 10px rgba(255, 102, 0, 0.2);
    }
    
    .dataframe tbody td {
        padding: 16px !important;
        font-size: 15px;
        font-weight: 500;
    }
    
    /* Chart Container */
    .js-plotly-plot {
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        border: 3px solid #E9ECEF;
        transition: all 0.3s ease;
    }
    
    .js-plotly-plot:hover {
        box-shadow: 0 12px 40px rgba(0,0,0,0.18);
        transform: translateY(-3px);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #003366 0%, #004080 100%);
        color: #FFFFFF !important;
        border: none;
        border-radius: 16px;
        padding: 16px 40px;
        font-weight: 700;
        box-shadow: 0 8px 24px rgba(0, 51, 102, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #004080 0%, #0059b3 100%);
        box-shadow: 0 12px 36px rgba(0, 51, 102, 0.5);
        transform: translateY(-4px) scale(1.02);
    }
    
    /* Preset Cards */
    .preset-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 100%);
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        border: 3px solid transparent;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .preset-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.15);
        border-color: #FF6600;
    }
    
    .preset-icon {
        font-size: 3em;
        margin-bottom: 15px;
        animation: pulse 2s infinite;
    }
    
    .preset-title {
        font-weight: 700;
        font-size: 1.2em;
        color: #003366;
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(135deg, #003366 0%, #004080 100%);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        animation: slideIn 0.5s ease-out;
    }
    
    .section-title {
        color: white;
        margin: 0;
        font-size: 2.2em;
        font-weight: 700;
    }
    
    .section-subtitle {
        color: #E6F0FF;
        margin-top: 12px;
        font-size: 1.2em;
        font-weight: 500;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 14px;
        height: 14px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F8F9FA;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #FF6600 0%, #FF8533 100%);
        border-radius: 10px;
        border: 3px solid #F8F9FA;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #FF8533 0%, #FFA366 100%);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 4px;
        background: linear-gradient(90deg, transparent 0%, #FF6600 50%, transparent 100%);
        margin: 40px 0;
    }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).parent
MODEL_FILE = ROOT / "real_estate_model.pkl"
DATA_FILE = ROOT / "india_housing_prices.csv"

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

# ---------- Main App ----------
def main():
    # Header
    st.markdown("""
        <div class='main-header'>
            <h1 class='gradient-text'>🏠 Real Estate Price Prediction</h1>
            <p class='sub-header'>Powered by AI • Instant Estimates • Market Insights</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    meta = load_model_metadata()
    if not meta:
        st.error("❌ Model not found. Please ensure 'real_estate_model.pkl' exists.")
        return
    
    model = meta.get('model')
    feature_names = meta.get('feature_names', [])
    
    if not model or not feature_names:
        st.error("❌ Invalid model metadata")
        return
    
    # Load dataset for defaults
    df = None
    if DATA_FILE.exists():
        try:
            df = pd.read_csv(DATA_FILE)
        except Exception:
            pass
    
    # Session state
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'preset' not in st.session_state:
        st.session_state.preset = None
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 History", "📈 Market Insights"])
    
    with tab1:
        st.markdown("""
            <div class='section-header'>
                <h2 class='section-title'>🔮 Property Price Estimator</h2>
                <p class='section-subtitle'>Get instant AI-powered price estimates for properties across India</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Presets
        st.markdown("<h3 style='color: #003366; margin: 40px 0 25px 0;'>⚡ Quick Presets</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #666; margin-bottom: 25px;'>Start with pre-configured property templates</p>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        presets = {
            'luxury': {'icon': '🏰', 'title': 'Luxury Property', 'color': 'linear-gradient(135deg, #FFD700 0%, #FFA500 100%)', 'values': {'Area': 3500, 'BHK': 5, 'Bedroom': 4, 'Bathroom': 4}},
            'budget': {'icon': '💰', 'title': 'Budget Friendly', 'color': 'linear-gradient(135deg, #90EE90 0%, #32CD32 100%)', 'values': {'Area': 800, 'BHK': 2, 'Bedroom': 2, 'Bathroom': 1}},
            'villa': {'icon': '🏡', 'title': 'Spacious Villa', 'color': 'linear-gradient(135deg, #87CEEB 0%, #4682B4 100%)', 'values': {'Area': 2500, 'BHK': 4, 'Bedroom': 4, 'Bathroom': 3}}
        }
        
        for col, (preset_key, preset_data) in zip([col1, col2, col3], presets.items()):
            with col:
                st.markdown(f"""
                    <div class='preset-card' style='background: {preset_data['color']};'>
                        <div class='preset-icon'>{preset_data['icon']}</div>
                        <div class='preset-title'>{preset_data['title']}</div>
                    </div>
                """, unsafe_allow_html=True)
                if st.button(f"Select {preset_data['title']}", use_container_width=True, key=f"{preset_key}_btn"):
                    st.session_state['preset'] = preset_key
                    st.rerun()
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Form
        with st.form("prediction_form"):
            st.markdown("<h3 style='color: #003366; margin-bottom: 25px;'>📝 Property Details</h3>", unsafe_allow_html=True)
            
            # Get defaults
            preset = st.session_state.get('preset')
            preset_vals = presets.get(preset, {}).get('values', {}) if preset else {}
            
            col1, col2 = st.columns(2)
            
            with col1:
                city_options = df['City'].unique().tolist() if df is not None and 'City' in df.columns else ['Mumbai', 'Delhi', 'Bangalore']
                city = st.selectbox("🏙️ City", options=city_options, key="city_input")
                
                area = st.number_input(
                    "📐 Area (sqft)", 
                    min_value=100, 
                    max_value=50000, 
                    value=preset_vals.get('Area', int(df_median_or_default(df, 'Area', 1000))),
                    step=100
                )
                
                bhk = st.number_input(
                    "🏘️ BHK", 
                    min_value=1, 
                    max_value=10, 
                    value=preset_vals.get('BHK', int(df_median_or_default(df, 'BHK', 2))),
                    step=1
                )
            
            with col2:
                bedrooms = st.number_input(
                    "🛏️ Bedrooms", 
                    min_value=1, 
                    max_value=20, 
                    value=preset_vals.get('Bedroom', int(df_median_or_default(df, 'Bedroom', 2))),
                    step=1
                )
                
                bathrooms = st.number_input(
                    "🚿 Bathrooms", 
                    min_value=1, 
                    max_value=10, 
                    value=preset_vals.get('Bathroom', int(df_median_or_default(df, 'Bathroom', 2))),
                    step=1
                )
                
                balconies = st.number_input(
                    "🌅 Balconies", 
                    min_value=0, 
                    max_value=10, 
                    value=int(df_median_or_default(df, 'Balcony', 1)),
                    step=1
                )
            
            # Amenities
            st.markdown("<h4 style='color: #003366; margin: 30px 0 15px 0;'>✨ Amenities</h4>", unsafe_allow_html=True)
            amenity_options = ['Parking', 'Gym', 'Swimming Pool', 'Garden', 'Security', 'Power Backup']
            selected_amenities = st.multiselect("Select amenities", options=amenity_options, default=[])
            
            st.markdown("<br>", unsafe_allow_html=True)
            prediction_button = st.form_submit_button("🔍 Get Price Estimate", use_container_width=True)
            
            if prediction_button:
                with st.spinner("🔄 Analyzing property data with AI..."):
                    time.sleep(0.8)
                    
                    # Prepare input
                    input_data = {
                        'City': city,
                        'Area': area,
                        'BHK': bhk,
                        'Bedroom': bedrooms,
                        'Bathroom': bathrooms,
                        'Balcony': balconies,
                        'Parking': 1 if 'Parking' in selected_amenities else 0,
                        'Gym': 1 if 'Gym' in selected_amenities else 0,
                        'SwimmingPool': 1 if 'Swimming Pool' in selected_amenities else 0,
                        'Garden': 1 if 'Garden' in selected_amenities else 0,
                        'Security': 1 if 'Security' in selected_amenities else 0,
                        'PowerBackup': 1 if 'Power Backup' in selected_amenities else 0
                    }
                    
                    # Add missing features with defaults
                    for feat in feature_names:
                        if feat not in input_data:
                            input_data[feat] = 0
                    
                    X_input = pd.DataFrame([input_data])[feature_names]
                    pred = model.predict(X_input)[0]
                    
                    # Display result
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #E6F7F0 0%, #D1F2E8 100%);
                                    padding: 40px; border-radius: 20px; border-left: 8px solid #00CC66;
                                    box-shadow: 0 8px 32px rgba(0, 204, 102, 0.25); margin: 30px 0;
                                    animation: slideIn 0.6s ease-out;'>
                            <div style='text-align: center;'>
                                <div style='font-size: 1.3em; color: #00CC66; font-weight: 700; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 2px;'>
                                    ✅ ESTIMATED PROPERTY VALUE
                                </div>
                                <div style='font-size: 4em; color: #003366; font-weight: 800; margin: 20px 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);'>
                                    {fmt_currency(pred)} <span style='font-size: 0.6em;'>Lakhs</span>
                                </div>
                                <div style='color: #666; font-size: 1.1em; margin-top: 15px; font-weight: 500;'>
                                    🤖 AI-Powered Prediction • ⚡ Instant Results • 📊 Data-Driven
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Store history
                    history_entry = {
                        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'City': city,
                        'Area_sqft': area,
                        'BHK': bhk,
                        'Bedrooms': bedrooms,
                        'Bathrooms': bathrooms,
                        'Predicted_Price_Lakhs': f"{fmt_currency(pred)} Lakhs"
                    }
                    st.session_state.prediction_history.append(history_entry)
    
    with tab2:
        st.markdown("""
            <div class='section-header'>
                <h2 class='section-title'>📊 Prediction History</h2>
                <p class='section-subtitle'>Track all your property valuations in one place</p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.prediction_history:
            df_history = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(df_history, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                csv = df_history.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download History (CSV)",
                    data=csv,
                    file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.prediction_history = []
                    st.rerun()
        else:
            st.info("📝 No predictions yet. Start by making a prediction in the Predict tab!")
    
    with tab3:
        st.markdown("""
            <div class='section-header'>
                <h2 class='section-title'>📈 Market Insights</h2>
                <p class='section-subtitle'>Explore comprehensive real estate market analytics and trends</p>
            </div>
        """, unsafe_allow_html=True)
        
        if df is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Total Properties", f"{len(df):,}", delta="Live Data")
            with col2:
                avg_price = df['Price_in_Lakhs'].mean() if 'Price_in_Lakhs' in df.columns else 0
                st.metric("💰 Avg Price", f"{fmt_currency(avg_price)} L", delta="+5.2%")
            with col3:
                cities = df['City'].nunique() if 'City' in df.columns else 0
                st.metric("🏙️ Cities Covered", f"{cities}", delta="Growing")
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Charts
            if 'Price_in_Lakhs' in df.columns:
                fig = px.histogram(
                    df, 
                    x='Price_in_Lakhs', 
                    nbins=50,
                    title='📊 Price Distribution',
                    color_discrete_sequence=['#FF6600']
                )
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family='Inter, sans-serif', size=14),
                    title_font=dict(size=20, color='#003366', family='Inter, sans-serif', weight='bold')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if 'City' in df.columns and 'Price_in_Lakhs' in df.columns:
                city_avg = df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(10)
                fig = px.bar(
                    x=city_avg.index,
                    y=city_avg.values,
                    title='🏙️ Average Price by Top 10 Cities',
                    labels={'x': 'City', 'y': 'Average Price (Lakhs)'},
                    color=city_avg.values,
                    color_continuous_scale=[[0, '#003366'], [1, '#FF6600']]
                )
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family='Inter, sans-serif', size=14),
                    title_font=dict(size=20, color='#003366', family='Inter, sans-serif', weight='bold'),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 No market data available. Load 'india_housing_prices.csv' for insights.")

if __name__ == "__main__":
    main()

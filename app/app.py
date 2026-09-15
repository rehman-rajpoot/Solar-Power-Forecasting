import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

# Set page configuration with clean layout
st.set_page_config(
    page_title="Solar Power Predictor | FAST-NUCES",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling: FAST University Classic Navy Blue & White Theme
st.markdown("""
    <style>
    /* Main container background */
    .stApp {
        background-color: #F8F9FA;
        color: #212529;
    }
    
    /* FAST Navy Blue Header Card */
    .header-card {
        background: linear-gradient(135deg, #002D62 0%, #004080 100%);
        color: #FFFFFF;
        padding: 24px 28px;
        border-radius: 8px;
        margin-bottom: 24px;
        border-left: 6px solid #FFC107;
        box-shadow: 0 2px 6px rgba(0, 45, 98, 0.12);
    }
    .header-card h1 {
        color: #FFFFFF !important;
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .header-card p {
        color: #E0E8F5;
        font-size: 14px;
        margin-bottom: 0;
    }

    /* Section styling */
    .content-box {
        background-color: #FFFFFF;
        border: 1px solid #DEE2E6;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .section-title {
        color: #002D62;
        font-size: 17px;
        font-weight: 600;
        border-bottom: 2px solid #002D62;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }

    /* Prediction Result Card */
    .result-card {
        background-color: #FFFFFF;
        border: 2px solid #002D62;
        border-radius: 8px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 45, 98, 0.1);
    }
    .result-label {
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #495057;
        font-weight: 600;
    }
    .result-value {
        font-size: 38px;
        font-weight: 800;
        color: #002D62;
        margin: 8px 0;
    }
    .result-subtext {
        font-size: 14px;
        color: #6C757D;
    }

    /* Primary FAST Button */
    div.stButton > button:first-child {
        background-color: #002D62;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        transition: background-color 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #004080;
        color: white;
    }

    /* Status badge */
    .badge-active {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        background-color: #D4EDDA;
        color: #155724;
    }
    .badge-night {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        background-color: #E2E3E5;
        color: #383D41;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Saved Weights and Standardization Parameters Safely
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
if not RESULTS_DIR.exists():
    RESULTS_DIR = Path("results")

@st.cache_data
def load_model_artifacts():
    theta_path = RESULTS_DIR / "theta_set_b.npy"
    means_path = RESULTS_DIR / "means_set_b.csv"
    stds_path = RESULTS_DIR / "stds_set_b.csv"
    
    if not (theta_path.exists() and means_path.exists() and stds_path.exists()):
        st.error("Model artifacts not found in results/ folder. Please run src/regression.py first.")
        st.stop()
        
    theta_arr = np.load(theta_path)
    means_series = pd.read_csv(means_path, index_col=0).squeeze()
    stds_series = pd.read_csv(stds_path, index_col=0).squeeze()
    return theta_arr, means_series, stds_series

theta, means, stds = load_model_artifacts()

# ---------------------------------------------------------
# Academic Header
# ---------------------------------------------------------
st.markdown("""
    <div class="header-card">
        <h1>FAST-NUCES | Solar Power Forecasting System</h1>
        <p>Applied Machine Learning — Linear Regression with Open-Meteo Public Weather Features (Set B Model)</p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Presets & Input Section
# ---------------------------------------------------------
st.markdown('<div class="content-box">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Scenario Preset & Input Parameters</div>', unsafe_allow_html=True)

preset = st.selectbox(
    "Select a Scenario Preset (or customize parameters below):",
    [
        "Custom Input",
        "Clear Summer Noon (12:00 PM)",
        "Morning Sun (08:00 AM)",
        "Overcast / Cloudy Afternoon (02:00 PM)",
        "Night / Pre-Dawn (02:00 AM)"
    ]
)

# Preset initial values
default_hour = 12
default_rad = 750.0
default_temp = 34.0
default_cloud = 15.0

if preset == "Clear Summer Noon (12:00 PM)":
    default_hour = 12
    default_rad = 880.0
    default_temp = 36.5
    default_cloud = 10.0
elif preset == "Morning Sun (08:00 AM)":
    default_hour = 8
    default_rad = 380.0
    default_temp = 28.0
    default_cloud = 20.0
elif preset == "Overcast / Cloudy Afternoon (02:00 PM)":
    default_hour = 14
    default_rad = 310.0
    default_temp = 31.0
    default_cloud = 85.0
elif preset == "Night / Pre-Dawn (02:00 AM)":
    default_hour = 2
    default_rad = 0.0
    default_temp = 25.0
    default_cloud = 30.0

col1, col2 = st.columns(2)

with col1:
    hour = st.slider("Hour of Day (24-Hour)", min_value=0, max_value=23, value=default_hour, format="%d:00")
    formatted_time = f"{hour:02d}:00 (" + (f"{hour} AM" if hour < 12 else (f"12 PM" if hour == 12 else f"{hour-12} PM")) + ")"
    st.caption(f"Selected Local Time: **{formatted_time}**")
    
    sw_radiation = st.number_input(
        "Shortwave Solar Radiation (W/m²)",
        min_value=0.0,
        max_value=1200.0,
        value=float(default_rad),
        step=25.0,
        help="Open-Meteo hourly shortwave radiation at ground level."
    )

with col2:
    temp_2m = st.number_input(
        "Ambient Temperature at 2m (°C)",
        min_value=-10.0,
        max_value=55.0,
        value=float(default_temp),
        step=0.5,
        help="Air temperature measured 2 meters above ground."
    )
    
    cloud_cover = st.slider(
        "Total Cloud Cover (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(default_cloud),
        step=5.0,
        help="Percentage of sky covered by clouds."
    )

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Prediction Computation
# ---------------------------------------------------------
predict_btn = st.button("Predict Solar AC Power Generation")

# Always calculate using current inputs
sin_hour = np.sin(2 * np.pi * hour / 24)
cos_hour = np.cos(2 * np.pi * hour / 24)

raw_features = pd.Series({
    "sw_radiation": sw_radiation,
    "temp_2m": temp_2m,
    "cloud_cover": cloud_cover,
    "sin_hour": sin_hour,
    "cos_hour": cos_hour,
})

# Z-score standardization using training statistics
scaled_features = (raw_features - means) / stds

# Construct design vector with intercept x0 = 1.0
X_input = np.hstack([1.0, scaled_features.to_numpy()])
raw_pred = float(X_input @ theta)
final_prediction = max(0.0, raw_pred)

# Peak capacity benchmark for Plant 1 is ~22,000 kW
peak_capacity_kw = 22000.0
capacity_factor = (final_prediction / peak_capacity_kw) * 100

st.markdown('<div class="content-box">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)

res_col1, res_col2, res_col3 = st.columns([1.5, 1, 1])

with res_col1:
    st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Predicted Plant Output (AC Power)</div>
            <div class="result-value">{final_prediction:,.2f} <span style="font-size:20px;">kW</span></div>
            <div class="result-subtext">Equivalent to <strong>{(final_prediction / 1000.0):.3f} MW</strong></div>
        </div>
    """, unsafe_allow_html=True)

with res_col2:
    st.markdown("**Operational Status**")
    if sw_radiation <= 5.0 or hour < 5 or hour > 19:
        st.markdown('<span class="badge-night">🌙 Night / Off-Grid</span>', unsafe_allow_html=True)
        st.write("Solar irradiation is negligible. Inverters remain in standby mode.")
    else:
        st.markdown('<span class="badge-active">☀️ Active Daylight Generation</span>', unsafe_allow_html=True)
        st.write("Plant actively feeding converted solar energy to the grid.")

with res_col3:
    st.markdown("**Estimated Plant Load**")
    st.metric(label="Capacity Utilization", value=f"{min(100.0, capacity_factor):.1f}%")
    st.caption("Benchmark: Plant 1 cumulative peak output (~22 MW).")

st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------

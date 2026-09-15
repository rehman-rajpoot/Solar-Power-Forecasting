# Solar Power Generation Forecasting (FAST-NUCES)

### Applied Machine Learning (CS) — Assignment 1
**Domain:** Solar Photovoltaic (PV) Energy Forecasting  
**Dataset:** Kaggle Solar Power Generation (Plant 1 & Plant 2) + Open-Meteo Historical Archive API  

---

## 📌 Project Overview
This project implements an end-to-end Machine Learning pipeline to predict plant-level **AC Power output (kW)** for a solar power plant. 

The models are implemented **from scratch using NumPy** (Ordinary Least Squares via Normal Equation, Batch Gradient Descent, and Stochastic Gradient Descent). The project compares:
- **Set A Features (Local Sensor Telemetry):** In-situ Pyranometer Irradiation, Module Temperature, Ambient Temperature, and Cyclical Hour ($\sin/\cos$).
- **Set B Features (Public Meteorological Data):** Open-Meteo Global Shortwave Radiation ($W/m^2$), 2m Ambient Temperature ($^\circ C$), Total Cloud Cover ($\%$), and Cyclical Hour ($\sin/\cos$).

A clean, single-page **Streamlit** dashboard is provided for interactive inference based on public weather inputs.

---

## 🏗️ Repository Structure

```
├── app/
│   └── app.py                          # Streamlit web application (FAST theme)
├── data/
│   ├── Plant_1_Generation_Data.csv      # Raw generation records (22 inverters)
│   ├── Plant_1_Weather_Sensor_Data.csv  # Raw on-site weather sensor records
│   ├── Plant_2_Generation_Data.csv      # Plant 2 raw generation data
│   ├── Plant_2_Weather_Sensor_Data.csv  # Plant 2 raw weather sensor data
│   ├── plant1_hourly.csv               # Cleaned & hourly-aggregated plant dataset
│   ├── plant1_openmeteo.csv            # Extracted Open-Meteo weather data
│   └── plant1_merged.csv               # Merged dataset (sensors + Open-Meteo)
├── results/
│   ├── theta_set_b.npy                 # Serialized Set B model weights (Normal Eq)
│   ├── means_set_b.csv                 # Feature means for standardization
│   ├── stds_set_b.csv                  # Feature standard deviations
│   └── plot*.png                       # EDA, convergence, and residual plots
├── src/
│   ├── load_data.py                    # Ingestion & raw file verification
│   ├── prepare.py                      # DateTime parsing, inverter aggregation, hourly resampling
│   ├── fetch_weather.py                # Open-Meteo API query & peak irradiation analysis
│   ├── eda.py                          # Diagnostic EDA & generation profile plots
│   └── regression.py                   # From-scratch regression (Normal Eq, Batch GD, SGD)
├── .env.example                        # Environment variable configuration template
├── .gitignore                          # Ignored caches, credentials, and virtual environments
├── requirements.txt                    # Project Python dependencies
└── README.md                           # Documentation & execution guide
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone <YOUR_REPOSITORY_URL>
cd colab_workspace_backup
```

### 2. Install Dependencies
Run the following command in your terminal/CLI:
```bash
pip install -r requirements.txt
```

*(Optional)* Configure local environment parameters:
```bash
copy .env.example .env
```

---

## 🚀 How to Run

### 1. Launch the Interactive Web Dashboard (GUI)
To run the Streamlit prediction application:
```bash
streamlit run app/app.py
```
This opens a clean local web page (`http://localhost:8501`) styled in FAST University Navy Blue & White.

### 2. (Optional) Re-run the Pipeline from Scratch
If you wish to re-execute data processing, API fetching, or model retraining:
```bash
# 1. Verify raw data loading
python src/load_data.py

# 2. Resample and aggregate 15-minute data to hourly
python src/prepare.py

# 3. Fetch historical weather from Open-Meteo API
python src/fetch_weather.py

# 4. Generate EDA visualizations
python src/eda.py

# 5. Train regression models and export weights
python src/regression.py
```

---

## 📊 Methodology & Technical Details
- **Optimization Algorithms (from scratch):**
  - **Normal Equation:** Closed-form OLS solution $\theta = (X^T X)^{-1} X^T y$
  - **Batch Gradient Descent:** Iterative update across full batch ($\alpha = 10^{-4}$, 50,000 iterations)
  - **Stochastic Gradient Descent:** Per-sample updates ($\alpha = 0.01$, 50 epochs)
- **Feature Scaling:** Z-Score normalization computed strictly on the training partition (May 15 – June 10, 2020) and applied to test (June 11 – June 17, 2020) to prevent data leakage.
- **Physical Non-negativity:** Predictions are bounded by $\max(0, \hat{y})$ to enforce physical realism for nocturnal hours.

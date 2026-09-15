import requests
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import os
from dotenv import load_dotenv

# Load environment variables if .env exists
load_dotenv()

BASE_URL = os.getenv("OPEN_METEO_API_URL", "https://archive-api.open-meteo.com/v1/archive")
LATITUDE = os.getenv("PLANT_LATITUDE", "14.82")
LONGITUDE = os.getenv("PLANT_LONGITUDE", "78.28")
API_KEY = os.getenv("OPEN_METEO_API_KEY", "")

URL = (
    f"{BASE_URL}"
    f"?latitude={LATITUDE}&longitude={LONGITUDE}"
    "&start_date=2020-05-15&end_date=2020-06-17"
    "&hourly=temperature_2m,cloud_cover,shortwave_radiation"
    "&timezone=Asia%2FKolkata"
)
if API_KEY:
    URL += f"&apikey={API_KEY}"

def fetch_weather():
    response = requests.get(URL)
    response.raise_for_status()
    data = response.json()
    hourly = data["hourly"]
    df = pd.DataFrame({
        "datetime": hourly["time"],
        "sw_radiation": hourly["shortwave_radiation"],
        "temp_2m": hourly["temperature_2m"],
        "cloud_cover": hourly["cloud_cover"],
    })
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

if __name__ == "__main__":
    weather = fetch_weather()
    print("Rows downloaded:", weather.shape[0])

    Path("data").mkdir(exist_ok=True)
    weather.to_csv("data/plant1_openmeteo.csv", index=False)

    hourly_df = pd.read_csv("data/plant1_hourly.csv")
    hourly_df["datetime"] = pd.to_datetime(hourly_df["datetime"])

    merged = pd.merge(hourly_df, weather, on="datetime", how="inner")
    merged["irradiation_wm2"] = merged["irradiation"] * 1000
    merged.to_csv("data/plant1_merged.csv", index=False)
    print("Merged rows:", merged.shape[0])

    # Task 3.4: teen din chunein aur peaks compare karein
    merged["date"] = merged["datetime"].dt.date
    merged["hour"] = merged["datetime"].dt.hour

    unique_dates = sorted(merged["date"].unique())
    chosen_days = unique_dates[:3]  # pehle 3 din

    Path("results").mkdir(exist_ok=True)

    for day in chosen_days:
        day_data = merged[merged["date"] == day]

        plt.figure()
        plt.plot(day_data["hour"], day_data["irradiation_wm2"], marker="o", label="Sensor Irradiation (W/m2)")
        plt.plot(day_data["hour"], day_data["sw_radiation"], marker="x", label="Open-Meteo SW Radiation (W/m2)")
        plt.xlabel("Hour of Day")
        plt.ylabel("Radiation (W/m2)")
        plt.title(f"Sensor vs Open-Meteo Radiation - {day}")
        plt.legend()
        plt.savefig(f"results/plot5_peak_check_{day}.png")
        plt.show()

        sensor_peak_hour = day_data.loc[day_data["irradiation_wm2"].idxmax(), "hour"]
        openmeteo_peak_hour = day_data.loc[day_data["sw_radiation"].idxmax(), "hour"]

        print(f"Date: {day} | Sensor peak hour: {sensor_peak_hour} | Open-Meteo peak hour: {openmeteo_peak_hour}")

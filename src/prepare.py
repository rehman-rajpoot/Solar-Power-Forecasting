from pathlib import Path
import pandas as pd
from load_data import load_raw

# Step 1: raw data load karo
raw = load_raw()
gen1 = raw["gen1"]
sensor1 = raw["sensor1"]

# Step 2: DATE_TIME ko real datetime mein convert karo
gen1["DATE_TIME"] = pd.to_datetime(gen1["DATE_TIME"], format="%d-%m-%Y %H:%M")
sensor1["DATE_TIME"] = pd.to_datetime(sensor1["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")

print("gen1 first timestamp:", gen1["DATE_TIME"].min())
print("gen1 last timestamp:", gen1["DATE_TIME"].max())
print("sensor1 first timestamp:", sensor1["DATE_TIME"].min())
print("sensor1 last timestamp:", sensor1["DATE_TIME"].max())

# Step 3: har inverter ka AC_POWER aur DC_POWER sum karo, per timestamp
gen1_plant = gen1.groupby("DATE_TIME")[["AC_POWER", "DC_POWER"]].sum().reset_index()

# Step 4: dono ko DATE_TIME pe merge karo
merged = pd.merge(gen1_plant, sensor1, on="DATE_TIME", how="outer")
print("\nmerged rows:", merged.shape[0])

# Step 5: hourly resample karo
merged = merged.set_index("DATE_TIME")
hourly = merged.resample("1h").mean(numeric_only=True)
hourly = hourly.reset_index()

hourly = hourly.rename(columns={
    "DATE_TIME": "datetime",
    "AC_POWER": "ac_power",
    "DC_POWER": "dc_power",
    "AMBIENT_TEMPERATURE": "ambient_temp",
    "MODULE_TEMPERATURE": "module_temp",
    "IRRADIATION": "irradiation",
})

print("\nhourly rows (before dropna):", hourly.shape[0])

# Step 6: missing values check aur handle karo
missing_count = hourly.isna().any(axis=1).sum()
print("rows with missing values:", missing_count)

hourly_clean = hourly.dropna()
print("hourly rows (after dropna):", hourly_clean.shape[0])

# Step 7: CSV mein save karo
Path("data").mkdir(exist_ok=True)
hourly_clean.to_csv("data/plant1_hourly.csv", index=False)
print("\nSaved to data/plant1_hourly.csv")

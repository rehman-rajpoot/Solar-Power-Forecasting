import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Preprocessed hourly dataset load kia
df = pd.read_csv("data/plant1_hourly.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

# Plot 1: ac_power vs irradiation
plt.figure()
plt.scatter(df["irradiation"], df["ac_power"])
plt.xlabel("Irradiation (kW/m2)")
plt.ylabel("AC Power (kW)")
plt.title("AC Power vs Irradiation")
plt.savefig("results/plot1_ac_vs_irradiation.png")
plt.show()

# Plot 2: module_temp vs ambient_temp, coloured by irradiation
plt.figure()
sc = plt.scatter(df["ambient_temp"], df["module_temp"], c=df["irradiation"], cmap="viridis")
plt.xlabel("Ambient Temperature (C)")
plt.ylabel("Module Temperature (C)")
plt.title("Module Temp vs Ambient Temp (coloured by Irradiation)")
plt.colorbar(sc, label="Irradiation (kW/m2)")
plt.savefig("results/plot2_module_vs_ambient.png")
plt.show()

# Plot 3: ac_power vs dc_power
plt.figure()
plt.scatter(df["dc_power"], df["ac_power"])
plt.xlabel("DC Power (kW)")
plt.ylabel("AC Power (kW)")
plt.title("AC Power vs DC Power")
plt.savefig("results/plot3_ac_vs_dc.png")
plt.show()

# Inverter conversion efficiency (AC/DC ratio) calculate kia
ratio = (df["ac_power"] / df["dc_power"]).mean()
print("Average AC/DC ratio:", ratio)

# Plot 4: average ac_power for each hour of day
df["hour"] = df["datetime"].dt.hour
hourly_avg = df.groupby("hour")["ac_power"].mean()

plt.figure()
plt.plot(hourly_avg.index, hourly_avg.values, marker="o")
plt.xlabel("Hour of Day")
plt.ylabel("Average AC Power (kW)")
plt.title("Average AC Power by Hour of Day")
plt.xticks(range(0, 24))
plt.savefig("results/plot4_hourly_avg_power.png")
plt.show()

print("All plots saved.")

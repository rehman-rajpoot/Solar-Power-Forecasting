import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/plant1_merged.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

df["hour"] = df["datetime"].dt.hour
df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)

df["date"] = df["datetime"].dt.date

train_mask = (df["date"] >= pd.to_datetime("2020-05-15").date()) & (df["date"] <= pd.to_datetime("2020-06-10").date())
test_mask = (df["date"] >= pd.to_datetime("2020-06-11").date()) & (df["date"] <= pd.to_datetime("2020-06-17").date())

train_df = df[train_mask].copy()
test_df = df[test_mask].copy()

SET_A_FEATURES = ["irradiation", "module_temp", "ambient_temp", "sin_hour", "cos_hour"]
SET_B_FEATURES = ["sw_radiation", "temp_2m", "cloud_cover", "sin_hour", "cos_hour"]
TARGET = "ac_power"

def scale_features(train_df, test_df, feature_cols):
    means = train_df[feature_cols].mean()
    stds = train_df[feature_cols].std()
    train_scaled = (train_df[feature_cols] - means) / stds
    test_scaled = (test_df[feature_cols] - means) / stds
    return train_scaled, test_scaled, means, stds

def build_X(scaled_features_df):
    n = scaled_features_df.shape[0]
    intercept = np.ones((n, 1))
    features_array = scaled_features_df.to_numpy()
    return np.hstack([intercept, features_array])

def hypothesis(X, theta):
    return X @ theta

def cost(X, y, theta):
    predictions = hypothesis(X, theta)
    errors = predictions - y
    return 0.5 * np.sum(errors ** 2)

def fit_normal(X, y):
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    Xty = X.T @ y
    return XtX_inv @ Xty

def fit_batch_gd(X, y, alpha, n_iters):
    n, d = X.shape
    theta = np.zeros(d)
    cost_history = []
    for i in range(n_iters):
        predictions = hypothesis(X, theta)
        errors = y - predictions
        gradient = X.T @ errors
        theta = theta + alpha * gradient
        cost_history.append(cost(X, y, theta))
    return theta, cost_history

def fit_sgd(X, y, alpha, n_epochs):
    n, d = X.shape
    theta = np.zeros(d)
    cost_history = []
    for epoch in range(n_epochs):
        for i in range(n):
            xi = X[i]
            yi = y[i]
            prediction = hypothesis(xi, theta)
            error = yi - prediction
            gradient = error * xi
            theta = theta + alpha * gradient
        cost_history.append(cost(X, y, theta))
    return theta, cost_history

def rmse(X, y, theta):
    """RMSE = sqrt(2*J(theta)/m)"""
    m = X.shape[0]
    return np.sqrt(2 * cost(X, y, theta) / m)

def predict_clipped(X, theta):
    """Predictions calculate ki aur physical constraint apply karne ke liye negative values ko 0 pe clip kia."""
    preds = hypothesis(X, theta)
    return np.clip(preds, a_min=0, a_max=None)

if __name__ == "__main__":
    y_train = train_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()

    results = []

    for set_name, feature_cols in [("Set A", SET_A_FEATURES), ("Set B", SET_B_FEATURES)]:
        train_scaled, test_scaled, means, stds = scale_features(train_df, test_df, feature_cols)
        X_train = build_X(train_scaled)
        X_test = build_X(test_scaled)

        # Daytime mask for test set (irradiation > 0)
        daytime_mask = test_df["irradiation"].to_numpy() > 0

        methods = {
            "Normal Equation": fit_normal(X_train, y_train),
            "Batch GD": fit_batch_gd(X_train, y_train, alpha=0.0001, n_iters=50000)[0],
            "SGD": fit_sgd(X_train, y_train, alpha=0.01, n_epochs=50)[0],
        }

        for method_name, theta in methods.items():
            preds_test = predict_clipped(X_test, theta)

            # All-hours RMSE
            errors_all = preds_test - y_test
            rmse_all = np.sqrt(np.mean(errors_all ** 2))

            # Daytime-only RMSE
            errors_day = preds_test[daytime_mask] - y_test[daytime_mask]
            rmse_day = np.sqrt(np.mean(errors_day ** 2))

            print(f"{set_name} | {method_name} | RMSE (all)={rmse_all:.2f} | RMSE (daytime)={rmse_day:.2f}")
            results.append((set_name, method_name, rmse_all, rmse_day))
        
    # Task 5.5: Residuals vs hour of day (Set A, Normal Equation)
    train_scaled_A, test_scaled_A, means_A, stds_A = scale_features(train_df, test_df, SET_A_FEATURES)
    X_train_A = build_X(train_scaled_A)
    X_test_A = build_X(test_scaled_A)

    theta_normal_A = fit_normal(X_train_A, y_train)
    preds_test_A = predict_clipped(X_test_A, theta_normal_A)

    residuals = y_test - preds_test_A
    test_hours = test_df["hour"].to_numpy()

    plt.figure()
    plt.scatter(test_hours, residuals)
    plt.axhline(y=0, color="red", linestyle="--")  # Zero error reference line draw kia
    plt.xlabel("Hour of Day")
    plt.ylabel("Residual (actual - predicted) kW")
    plt.title("Residuals vs Hour of Day (Set A, Normal Equation)")
    plt.savefig("results/plot8_residuals_vs_hour.png")
    plt.show()

    # Diurnal pattern check karne ke liye hour-level average residual compute kia
    residual_df = pd.DataFrame({"hour": test_hours, "residual": residuals})
    avg_residual_by_hour = residual_df.groupby("hour")["residual"].mean()
    print("\nAverage residual by hour:\n", avg_residual_by_hour)
    
    # Task 6 deployment ke liye: Set B Normal Equation weights aur scaling parameters save kia
    train_scaled_B, test_scaled_B, means_B, stds_B = scale_features(train_df, test_df, SET_B_FEATURES)
    X_train_B = build_X(train_scaled_B)
    theta_normal_B = fit_normal(X_train_B, y_train)

    Path("results").mkdir(exist_ok=True)
    np.save("results/theta_set_b.npy", theta_normal_B)
    means_B.to_csv("results/means_set_b.csv")
    stds_B.to_csv("results/stds_set_b.csv")

    print("\nSet B theta saved:", theta_normal_B)
    print("Saved to results/theta_set_b.npy, means_set_b.csv, stds_set_b.csv")

    # Figure 5: Test week ke liye actual vs predicted AC power plot kia (Normal equation: Set A & Set B)
    preds_test_B = predict_clipped(X_test_B, theta_normal_B)

    plt.figure(figsize=(12, 5))
    plt.plot(test_df["datetime"], y_test, label="Actual AC Power", color="black", linewidth=1.5)
    plt.plot(test_df["datetime"], preds_test_A, label="Predicted (Set A: Sensors)", color="#1f77b4", linestyle="--", linewidth=1.2)
    plt.plot(test_df["datetime"], preds_test_B, label="Predicted (Set B: Open-Meteo)", color="#d62728", linestyle=":", linewidth=1.2)
    plt.xlabel("Date & Time")
    plt.ylabel("AC Power (kW)")
    plt.title("Actual vs Predicted AC Power on Test Week (June 11 - June 17, 2020)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig("results/plot9_actual_vs_predicted_test_week.png")
    plt.savefig("results/fig5_actual_vs_predicted_test_week.png")
    plt.show()
    print("Figure 5 saved to results/plot9_actual_vs_predicted_test_week.png")

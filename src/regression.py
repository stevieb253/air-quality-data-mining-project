import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# Load cleaned data
cleaned_file = "outputs/cleaned_air_quality.csv"

print("Loading cleaned dataset...")
df = pd.read_csv(cleaned_file)

# Columns
target_col = "EARLWOOD PM2.5 1h average [µg/m³]"

feature_cols = [
    "EARLWOOD PM10 1h average [µg/m³]",
    "EARLWOOD NO 1h average [pphm]",
    "EARLWOOD NO2 1h average [pphm]",
    "EARLWOOD OZONE 1h average [pphm]",
    "EARLWOOD TEMP 1h average [°C]",
    "EARLWOOD HUMID 1h average [%]",
    "EARLWOOD WSP 1h average [m/s]"
]

# Keep only needed columns and remove rows with missing values
model_df = df[feature_cols + [target_col]].dropna()

print("\nRows used for regression:")
print(model_df.shape[0])

# Split input features and target
X = model_df[feature_cols]
y = model_df[target_col]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))

print("\nRegression Results")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Save output folders
os.makedirs("outputs/graphs", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

# Save metrics
results_df = pd.DataFrame({
    "Metric": ["R2 Score", "MAE", "RMSE"],
    "Value": [r2, mae, rmse]
})

results_df.to_csv("outputs/tables/regression_results.csv", index=False)

# Save model coefficients
coefficients_df = pd.DataFrame({
    "Feature": feature_cols,
    "Coefficient": model.coef_
})

coefficients_df.to_csv("outputs/tables/regression_coefficients.csv", index=False)

print("\nRegression metrics saved to outputs/tables/regression_results.csv")
print("Regression coefficients saved to outputs/tables/regression_coefficients.csv")

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Actual vs Predicted PM2.5 Concentration")

# Add reference line
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())
plt.plot([min_value, max_value], [min_value, max_value])

plt.tight_layout()
plt.savefig("outputs/graphs/regression_actual_vs_predicted.png", dpi=300)
plt.close()

print("Regression graph saved to outputs/graphs/regression_actual_vs_predicted.png")
print("\nRegression analysis complete.")
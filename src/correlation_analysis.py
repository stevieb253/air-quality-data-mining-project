import os
import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset created by preprocessing.py
cleaned_file = "outputs/cleaned_air_quality.csv"

print("Loading cleaned dataset...")
df = pd.read_csv(cleaned_file)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

# Select numeric columns only
numeric_df = df.select_dtypes(include=["number"])

print("\nNumeric columns used for correlation:")
print(numeric_df.columns)

# Create correlation matrix
correlation_matrix = numeric_df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Create outputs folder for graphs
os.makedirs("outputs/graphs", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

# Save correlation matrix as CSV
correlation_matrix.to_csv("outputs/tables/correlation_matrix.csv")

# Plot heatmap using matplotlib
plt.figure(figsize=(12, 8))
plt.imshow(correlation_matrix, cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="Correlation Coefficient")

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title("Correlation Heatmap of Earlwood Air Quality Variables")
plt.tight_layout()

# Save figure
heatmap_file = "outputs/graphs/correlation_heatmap.png"
plt.savefig(heatmap_file, dpi=300)
plt.close()

print(f"\nCorrelation heatmap saved to: {heatmap_file}")
print("Correlation matrix saved to: outputs/tables/correlation_matrix.csv")

# Find strongest correlations with PM2.5
pm25_col = None

for col in correlation_matrix.columns:
    if "PM2.5" in col:
        pm25_col = col
        break

if pm25_col:
    print(f"\nStrongest correlations with {pm25_col}:")
    pm25_correlations = correlation_matrix[pm25_col].sort_values(ascending=False)
    print(pm25_correlations)

    pm25_correlations.to_csv("outputs/tables/pm25_correlations.csv")
    print("\nPM2.5 correlations saved to: outputs/tables/pm25_correlations.csv")
else:
    print("\nPM2.5 column not found.")
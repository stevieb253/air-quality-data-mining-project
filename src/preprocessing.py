import pandas as pd
import os

# File paths
main_file = "data/Earlwood_Air_Data_17_18.csv"
site_file = "data/air-quality-monitoring-sites-summary.csv"

# Load datasets
print("Loading datasets...\n")

df = pd.read_csv(main_file)
site_df = pd.read_csv(site_file)

# Basic info
print("Main Dataset Shape:")
print(df.shape)
print("\nMain Dataset Columns:")
print(df.columns)

print("\nSite Summary Dataset Shape:")
print(site_df.shape)
print("\nSite Summary Columns:")
print(site_df.columns)

# Missing values check
print("\nMissing Values in Main Dataset:")
print(df.isnull().sum())

# Drop CO columns if they are fully empty
co_columns = [col for col in df.columns if "CO" in col]

for col in co_columns:
    if df[col].isnull().all():
        print(f"\nDropping empty column: {col}")
        df.drop(columns=[col], inplace=True)

# Combine Date + Time into one datetime column
if "Date" in df.columns and "Time" in df.columns:
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        errors="coerce"
    )
    print("\nCreated Datetime column successfully.")

# Convert numeric columns safely
for col in df.columns:
    if col not in ["Date", "Time", "Datetime"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Create outputs folder if needed
os.makedirs("outputs", exist_ok=True)

# Save cleaned dataset
output_file = "outputs/cleaned_air_quality.csv"
df.to_csv(output_file, index=False)

print(f"\nCleaned dataset saved to: {output_file}")
print("\nPreprocessing complete.")
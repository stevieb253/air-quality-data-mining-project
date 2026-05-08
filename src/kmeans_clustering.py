import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


def safe_file_name(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def find_column(columns, required_parts, forbidden_parts=None):
    forbidden_parts = forbidden_parts or []

    for column in columns:
        upper_column = column.upper()
        has_required_parts = all(
            part.upper() in upper_column
            for part in required_parts
        )
        has_forbidden_parts = any(
            part.upper() in upper_column
            for part in forbidden_parts
        )

        if has_required_parts and not has_forbidden_parts:
            return column

    return None


# Load data
df = pd.read_csv(
    r"C:\Users\Lance Tieng\Documents\GitHub\air-quality-data-mining-project\outputs\cleaned_air_quality.csv"
)

# Features used for clustering. Columns are matched by measurement names so
# unit-symbol encoding differences do not break the script.
feature_lookup = {
    "PM2.5": find_column(df.columns, ["PM2.5", "1h average"]),
    "PM10": find_column(df.columns, ["PM10", "1h average"]),
    "O3": find_column(df.columns, ["OZONE", "1h average"], ["4h"]),
    "NO": find_column(df.columns, ["NO", "1h average"], ["NO2"]),
    "NO2": find_column(df.columns, ["NO2", "1h average"]),
    "Temperature": find_column(df.columns, ["TEMP", "1h average"]),
    "Wind Speed": find_column(df.columns, ["WSP", "1h average"]),
    "Humidity": find_column(df.columns, ["HUMID", "1h average"]),
}

missing_features = [
    label for label, column in feature_lookup.items()
    if column is None
]

if missing_features:
    print("\nMissing features skipped:")
    print(", ".join(missing_features))

feature_lookup = {
    label: column for label, column in feature_lookup.items()
    if column is not None
}

features = list(feature_lookup.values())
feature_labels = {
    column: label for label, column in feature_lookup.items()
}

pm25_col = feature_lookup["PM2.5"]

# Keep only needed columns and drop missing values
df = df[features].dropna()

# Normalize data before K-Means because features use different units
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Print cluster averages for every feature
cluster_averages = df.groupby("cluster").mean()
print("\nCluster averages for all selected features:")
print(cluster_averages.to_string())

# PCA graph for all features
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_components = pd.DataFrame(
    pca.components_,
    columns=features,
    index=["PCA 1", "PCA 2"],
)

print("\nPCA component weights for all selected features:")
print(pca_components.to_string())

explained_variance = pd.DataFrame({
    "Component": ["PCA 1", "PCA 2"],
    "Explained Variance Ratio": pca.explained_variance_ratio_,
})

print("\nHow much variation each PCA component explains:")
print(explained_variance.to_string(index=False))

graphs_dir = "outputs/graphs"
pm25_maps_dir = "outputs/graphs/pm25_cluster_maps"
os.makedirs(graphs_dir, exist_ok=True)
os.makedirs(pm25_maps_dir, exist_ok=True)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df["cluster"],
    cmap="viridis",
    alpha=0.65,
)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("K-Means Clusters (All Features)")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig("outputs/graphs/kmeans_selected_features.png", dpi=300)
plt.close()

print("\nSaved PCA cluster graph to outputs/graphs/kmeans_selected_features.png")

# Create one PM2.5 comparison cluster map for each other feature
for feature in features:
    if feature == pm25_col:
        continue

    label = feature_labels[feature]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df[feature],
        df[pm25_col],
        c=df["cluster"],
        cmap="viridis",
        alpha=0.6,
    )
    plt.xlabel(label)
    plt.ylabel("PM2.5")
    plt.title(f"PM2.5 vs {label} by K-Means Cluster")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()

    output_file = f"{pm25_maps_dir}/pm25_vs_{safe_file_name(label)}.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved PM2.5 cluster map to {output_file}")

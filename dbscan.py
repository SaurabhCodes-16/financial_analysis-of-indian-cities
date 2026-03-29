import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# ---------------------------
# Setup
# ---------------------------
os.makedirs("plots", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ---------------------------
# Load Data
# ---------------------------
df = pd.read_csv("outputs/phase1_processed.csv")

# ---------------------------
# Feature Selection
# ---------------------------
features = [
    'Income', 'Expense_Ratio', 'Savings_Gap',
    'Recovery_Rate', 'Dependents', 'Age'
]

X = df[features]

# ---------------------------
# Scaling (VERY IMPORTANT)
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# K-Distance Plot (to find eps)
# ---------------------------
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

distances = np.sort(distances[:, 4])

plt.figure()
plt.plot(distances)
plt.title("K-Distance Graph (Choose eps at elbow)")
plt.savefig("plots/k_distance.png")
plt.close()

# ---------------------------
# Apply DBSCAN
# ---------------------------
dbscan = DBSCAN(eps=0.75, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

df['DBSCAN_Cluster'] = labels

# ---------------------------
# Number of clusters
# ---------------------------
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print("Number of clusters:", n_clusters)
print("Number of noise points:", n_noise)

# ---------------------------
# Cluster distribution
# ---------------------------
print("\nCluster Counts:")
print(df['DBSCAN_Cluster'].value_counts())

# ---------------------------
# Save results
# ---------------------------
df.to_csv("outputs/phase_dbscan_clusters.csv", index=False)

print("\nDBSCAN Clustering Completed ✅")
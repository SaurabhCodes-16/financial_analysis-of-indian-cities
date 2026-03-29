import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("outputs/phase1_processed.csv")

features = [
    'Income', 'Expense_Ratio', 'Savings_Gap',
    'Recovery_Rate', 'Dependents', 'Age'
]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []

for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure()
plt.plot(range(2,8), inertia, marker='o')
plt.title("Elbow Plot")
plt.savefig("plots/elbow_plot.png")
plt.close()

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

cluster_map = {
    0: "Stable",
    1: "At Risk",
    2: "Vulnerable"
}

df['Vulnerability_Label'] = df['Cluster'].map(cluster_map)

df.to_csv("outputs/phase2_clustered.csv", index=False)

print("Phase 2 Completed")

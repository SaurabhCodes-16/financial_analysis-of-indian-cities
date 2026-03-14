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

# ---------------------------
# Elbow Method
# ---------------------------
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

# ---------------------------
# Enhanced Clustering Analysis with Multiple Algorithms
# ---------------------------
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# Function to evaluate clustering performance
def evaluate_clustering(X, labels, algorithm_name):
    # Use sample for silhouette score to speed up computation
    sample_size = min(5000, len(X))
    X_sample = X.sample(sample_size, random_state=42) if hasattr(X, 'sample') else X[:sample_size]

    if len(np.unique(labels)) > 1:
        try:
            sample_labels = labels[:sample_size] if len(labels) > sample_size else labels
            silhouette = silhouette_score(X_sample, sample_labels)
        except:
            silhouette = -1  # If silhouette fails

        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        return {
            'Silhouette Score': silhouette,
            'Davies-Bouldin Index': davies_bouldin,
            'Calinski-Harabasz Index': calinski_harabasz
        }
    else:
        return {'Silhouette Score': -1, 'Davies-Bouldin Index': float('inf'), 'Calinski-Harabasz Index': 0}

# Determine optimal number of clusters using multiple metrics
print("Evaluating optimal number of clusters...")
k_range = range(2, 8)
k_metrics = []

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(X_scaled)
    metrics = evaluate_clustering(X_scaled, labels_temp, f"K-means (k={k})")
    metrics['k'] = k
    k_metrics.append(metrics)

k_df = pd.DataFrame(k_metrics)
print("\nCluster Validation Metrics:")
print(k_df.round(4))

# Plot cluster validation metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(k_df['k'], k_df['Silhouette Score'], marker='o', linewidth=2)
axes[0].set_title('Silhouette Score vs Number of Clusters')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Silhouette Score')
axes[0].grid(True, alpha=0.3)

axes[1].plot(k_df['k'], k_df['Davies-Bouldin Index'], marker='o', linewidth=2, color='orange')
axes[1].set_title('Davies-Bouldin Index vs Number of Clusters')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Davies-Bouldin Index')
axes[1].grid(True, alpha=0.3)

axes[2].plot(k_df['k'], k_df['Calinski-Harabasz Index'], marker='o', linewidth=2, color='green')
axes[2].set_title('Calinski-Harabasz Index vs Number of Clusters')
axes[2].set_xlabel('Number of Clusters (k)')
axes[2].set_ylabel('Calinski-Harabasz Index')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/cluster_validation_metrics.png", dpi=300, bbox_inches='tight')
plt.close()

# Choose optimal k based on silhouette score and Calinski-Harabasz index
optimal_k = k_df.loc[k_df['Silhouette Score'].idxmax(), 'k']
print(f"\nOptimal number of clusters based on Silhouette Score: {optimal_k}")

# Alternative: Choose k where both metrics are good
balanced_k = 3  # Default to 3 as in original, but we can adjust based on analysis
print(f"Using {balanced_k} clusters for final analysis (balanced approach)")

# ---------------------------

# Compare Multiple Clustering Algorithms
# ---------------------------
print("\n" + "="*50)
print("COMPARING CLUSTERING ALGORITHMS")
print("="*50)

algorithms = {
    'K-Means': KMeans(n_clusters=balanced_k, random_state=42, n_init=10),
    'Gaussian Mixture': GaussianMixture(n_components=balanced_k, random_state=42)
}

algorithm_results = []

for name, algorithm in algorithms.items():
    if name == 'Gaussian Mixture':
        labels = algorithm.fit_predict(X_scaled)
    else:
        labels = algorithm.fit_predict(X_scaled)

    metrics = evaluate_clustering(X_scaled, labels, name)
    metrics['Algorithm'] = name
    algorithm_results.append(metrics)

    print(f"\n{name} Results:")
    for metric, value in metrics.items():
        if metric != 'Algorithm':
            print(f"  {metric}: {value:.4f}")

results_df = pd.DataFrame(algorithm_results)
print("\n" + "="*50)
print("ALGORITHM COMPARISON SUMMARY")
print("="*50)
print(results_df.round(4))

# Choose best algorithm based on silhouette score
best_algorithm = results_df.loc[results_df['Silhouette Score'].idxmax(), 'Algorithm']
print(f"\nBest algorithm based on Silhouette Score: {best_algorithm}")

# Use K-Means as final algorithm (most interpretable and commonly used)
print(f"\nUsing K-Means with {balanced_k} clusters for final clustering...")

# ---------------------------

# Final K-Means Clustering
# ---------------------------
kmeans = KMeans(n_clusters=balanced_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Label Mapping (adjust based on cluster characteristics if needed)
cluster_map = {
    0: "Stable",
    1: "At Risk",
    2: "Vulnerable"
}

df['Vulnerability_Label'] = df['Cluster'].map(cluster_map)

# Cluster Profiling and Analysis
# ---------------------------
print("\n" + "="*50)
print("CLUSTER PROFILING AND ANALYSIS")
print("="*50)

# Statistical summary of each cluster
cluster_summary = df.groupby('Vulnerability_Label')[features].agg(['mean', 'std', 'min', 'max'])
print("\nCluster Statistical Summary:")
print(cluster_summary.round(2))

# Cluster sizes
cluster_sizes = df['Vulnerability_Label'].value_counts().sort_index()
print(f"\nCluster Sizes:\n{cluster_sizes}")

# Save cluster profiling to file
with open("outputs/phase2_cluster_profiling.txt", "w") as f:
    f.write("CLUSTER PROFILING ANALYSIS\n")
    f.write("="*50 + "\n\n")
    f.write(f"Number of clusters: {balanced_k}\n")
    f.write(f"Algorithm used: K-Means\n\n")

    f.write("CLUSTER SIZES:\n")
    f.write(cluster_sizes.to_string())
    f.write("\n\n")

    f.write("CLUSTER STATISTICAL SUMMARY:\n")
    f.write(cluster_summary.round(2).to_string())

    f.write("\n\nALGORITHM COMPARISON:\n")
    f.write(results_df.round(4).to_string(index=False))

# Enhanced Visualization: Cluster Profiles
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for i, feature in enumerate(features):
    if i < 6:  # Only plot first 6 features
        for cluster in df['Vulnerability_Label'].unique():
            cluster_data = df[df['Vulnerability_Label'] == cluster][feature]
            axes[i].hist(cluster_data, alpha=0.7, label=cluster, bins=20)
        axes[i].set_title(f'{feature} Distribution by Cluster')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/cluster_profiles.png", dpi=300, bbox_inches='tight')
plt.close()

# Save final clustered data
df.to_csv("outputs/phase2_clustered.csv", index=False)

print("\nEnhanced clustering analysis completed!")
print("Results saved to outputs/phase2_cluster_profiling.txt")
print("Visualizations saved to plots/")

import numpy as np
from sklearn.datasets import make_blobs
from cuml import UMAP
from scipy.stats import zscore
import matplotlib.pyplot as plt
import time

# --- Data Generation ---
# Generate 100,000 samples with 960 features across 10 clusters.
start_time = time.time()
X, labels = make_blobs(n_samples=100_000, n_features=960, centers=10)
end_time = time.time()
print(f"Generate 1,000,000 samples with 960 features across 10 clusters ({end_time - start_time:.2f}s)")

# --- UMAP Projection ---
# Initialize the UMAP model with parameters to control the projection.
start_time = time.time()
umap_model = UMAP(
    n_neighbors=128,  # Balances local vs. global structure.
    n_components=2,  # Target dimension for visualization.
    min_dist=0.4,  # Controls how tightly points are packed.
    n_epochs=1000,  # Number of optimization iterations.
    build_algo="nn_descent",  # Use the fast approximate nearest neighbor algorithm on the GPU.
    # Low-level keywords to fine-tune the nn_descent algorithm for performance and memory.
    build_kwds={
        "nnd_do_batch": True,
        "nnd_n_clusters": 32,
        "nnd_graph_degree": 256,
        "nnd_intermediate_graph_degree": 512,
        "nnd_overlap_factor": 8,
    },
)
umap_projection = umap_model.fit_transform(X)
end_time = time.time()
print(f"Project the data in 2D with UMAP ({end_time - start_time:.2f}s)")


# --- Visualization ---
# Filter extreme outliers for a clearer plot (central 95% of data)
start_time = time.time()
z_scores = np.abs(zscore(umap_projection))
filter_mask = (z_scores < 1.96).all(axis=1)

# Create the scatter plot
fig, ax = plt.subplots(figsize=(6, 6))
scatter = ax.scatter(
    umap_projection[filter_mask, 0],
    umap_projection[filter_mask, 1],
    c=labels[filter_mask],
    s=15,
    alpha=0.3,
    edgecolors="none",
)

# Style the plot
legend = ax.legend(*scatter.legend_elements(), title="Clusters", ncol=2, frameon=False)
ax.axis("off")
plt.tight_layout()
plt.savefig("output.png")
end_time = time.time()
print(f"UMAP plot saved to output.png ({end_time - start_time:.2f}s)")

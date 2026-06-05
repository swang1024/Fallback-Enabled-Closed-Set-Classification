import numpy as np

def kmeans(X, k, max_iters=100):
    N, D = X.shape
    # Random initialization
    idx = np.random.choice(N, k, replace=False)
    centroids = X[idx].copy()                          # [k, D]

    for _ in range(max_iters):
        # Assign: compute distances [N, k]
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x·c
        x_sq = np.sum(X ** 2, axis=1, keepdims=True)      # [N, 1]
        c_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T  # [1, k]
        dist = x_sq + c_sq - 2 * (X @ centroids.T)        # [N, k]
        labels = np.argmin(dist, axis=1)                   # [N]

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                new_centroids[j] = members.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]  # keep old if empty

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels, centroids

# Example usage
if __name__ == "__main__":
    # Create synthetic data
    np.random.seed(0)
    X = np.vstack([
        np.random.randn(100, 2) + np.array([5, 5]),
        np.random.randn(100, 2) + np.array([-5, -5]),
        np.random.randn(100, 2) + np.array([5, -5])
    ])

    labels, centroids = kmeans(X, k=3)
    print("Centroids:\n", centroids)
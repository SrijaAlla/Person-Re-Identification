import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

# Load the detections.json file
with open("detections.json", "r") as f:
    detections = json.load(f)

# Extract features and detection_ids
features = []
detection_ids = []

for detection in detections:
    features.append(detection["feature"])
    detection_ids.append(detection["detection_id"])



# Normalize the feature vectors
features = np.array(features)
normalized_features = normalize(features, norm="l2")


# Number of iterations to perform
num_iterations = 10
best_silhouette = -1  # Initialize to lowest possible score
best_labels = None

# Iterate spectral clustering multiple times
for i in range(num_iterations):
    spectral = SpectralClustering(n_clusters=5, affinity='cosine', random_state=i)
    labels = spectral.fit_predict(normalized_features)

    # Calculate silhouette score to evaluate clustering performance
    silhouette_avg = silhouette_score(normalized_features, labels)
    print(f"Iteration {i+1}, Silhouette Score: {silhouette_avg}")

    # Keep track of the best result based on Silhouette Score
    if silhouette_avg > best_silhouette:
        best_silhouette = silhouette_avg
        best_labels = labels

# After iterations, best_labels will contain the labels with the best Silhouette Score
print(f"Best Silhouette Score: {best_silhouette}")

# Save the best clustering result
clusters = [[] for _ in range(5)]
for i, label in enumerate(best_labels):
    clusters[label].append(detection_ids[i])

# Save the best result to labels.json
with open("labels.json", "w") as f:
    json.dump(clusters, f)
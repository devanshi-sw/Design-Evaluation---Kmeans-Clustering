import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from google.colab import files
import io

# Load the dataset
df = pd.read_csv('evaluation.csv') 

# Calculate Z-scores for constraints and social value
df['z_constraint'] = (df['constraint'] - df['constraint'].mean()) / df['constraint'].std()
df['z_social_value'] = (df['social_value'] - df['social_value'].mean()) / df['social_value'].std()

#  processed data to a CSV 
df.to_csv('preprocessed_data.csv', index=False)

print(df.head())

# Read the uploaded file into a pandas DataFrame
data = pd.read_csv(io.BytesIO(uploaded['clusteranalysisalldata.csv']))

# Display the first few rows of the data to confirm
print(data.head())

print(data.columns)

wcss = []
for i in range(1, 11):  # checking for up to 10 clusters
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(data[['avg_z_constraints', 'avg_z_social_value']])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=8, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(data[['avg_z_constraints', 'avg_z_social_value']])
data['Cluster'] = clusters

plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='avg_z_constraints', y='avg_z_social_value', hue='Cluster', palette=sns.color_palette("hls", 8), legend="full")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label = 'Centroids')
plt.title('Clusters of Statements')
plt.xlabel('Average Z-score (Constraints)')
plt.ylabel('Average Z-score (Social Value)')
plt.legend()
plt.show()

from sklearn.metrics import silhouette_samples, silhouette_score

#from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(data, kmeans.labels_)

# Calculate silhouette values for each sample
silhouette_values = silhouette_samples(data, kmeans.labels_)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.set_xlim([-0.1, 1])
ax.set_ylim([0, len(data) + (4 + 1) * 10])  # Here, replace 4 with the number of clusters you've set.

y_lower = 10
for i in range(4):  # Here, replace 4 with the number of clusters you've set.
    cluster_silhouette_vals = silhouette_values[kmeans.labels_ == i]
    cluster_silhouette_vals.sort()

    size_cluster = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster

    color = cm.nipy_spectral(float(i) / 4)  # Here, replace 4 with the number of clusters you've set.
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, facecolor=color, edgecolor=color, alpha=0.7)

    y_lower = y_upper + 10

ax.set_xlabel("Silhouette Coefficient Values")
ax.set_ylabel("Cluster Label")
ax.axvline(x=silhouette_avg, color="red", linestyle="--")
ax.set_yticks([])  # Clear the yaxis labels / ticks
ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.show()

from sklearn.metrics import silhouette_score
silhouette_avg = silhouette_score(data, kmeans.labels_)

print("The average silhouette_score is :", silhouette_avg)

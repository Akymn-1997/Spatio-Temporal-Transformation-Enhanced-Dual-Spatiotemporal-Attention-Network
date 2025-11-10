import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import math
import os
import sklearn
from sklearn.cluster import KMeans
# Dataset 2
raw_csv_1 = np.array(pd.read_csv("hourly_wtbdata_245days.csv"))
cor_wt = np.array(pd.read_csv("sdwpf_baidukddcup2022_turb_location.csv"))

train_ratio = 0.6
valid_ratio = 0.2
print(cor_wt)

cor_wt = cor_wt[:,1:]
cor_wt = cor_wt.astype(float)

plt.figure(figsize=(8, 6))
plt.scatter(cor_wt[:, 0], cor_wt[:, 1], color='blue', alpha=0.5)
plt.title('Scatter Plot of 2D Data')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()

# Data Preprocessing
WS = raw_csv_1[1:,1::2]
WS = WS.astype(float)
WS = np.array(WS)
WS[np.isnan(WS)] = 0

#  Determine the best cluster number
from sklearn.metrics import silhouette_score
sse = []
silhouette_scores = []
WS_len = len(WS)
data = WS[:int(WS_len*.6)]
data_T = data.T
K_range = range(2, 15)

sse = []
silhouette_scores = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(data_T)
    sse.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(data_T, kmeans.labels_))
max_sse = max(sse)
max_sil = max(silhouette_scores)
normalized_sse = [(1 - x / max_sse)*1 for x in sse]
normalized_silhouette = [(x / max_sil) for x in silhouette_scores]  # 将范围从[-1,1]变换到[0,1]

composite_scores = [(sse + sil) / 2 for sse, sil in zip(normalized_sse, normalized_silhouette)]

plt.figure(figsize=(8, 6))

plt.plot(K_range, composite_scores, 'b-', label='Composite Scores', linewidth=2)  # Line in blue
plt.plot(K_range, composite_scores, 'ro', label='Data Points', markersize=8)  # Red circles for points

max_score_index = composite_scores.index(max(composite_scores))
max_score = composite_scores[max_score_index]
plt.plot(K_range[max_score_index], max_score, 'yp', markersize=10, label='Max Value')  # Green dot for max value

plt.axhline(y=max_score, color='orange', linestyle='--', linewidth=1)  # Horizontal line
plt.axvline(x=K_range[max_score_index], color='orange', linestyle='--', linewidth=1)  # Vertical line

plt.xlabel('Number of Subregions', fontsize=18)
plt.ylabel('EM', fontsize=18)

plt.xticks(fontsize=18, rotation=0)
plt.yticks(fontsize=18)

plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)

plt.tight_layout()
plt.show()

# Clustering of the wind turbines
WP_len = len(WS)
data = WS[:int(WP_len*train_ratio)]

n_clusters = 2
data_T = data.T
kmeans = KMeans(n_clusters=n_clusters, n_init=10 ,random_state=0).fit(data_T)
from matplotlib.cm import get_cmap
plt.figure(figsize=(8, 6))
cmap = get_cmap('tab10', n_clusters)

markers = ['o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', 'P']

cluster_heart = []
heart_seq = []
other_seq = []
kb_colc = []

for i in range(n_clusters):
    cluster_indices = np.where(kmeans.labels_ == i)[0]

    cluster_points = data_T[cluster_indices]

    cluster_center = kmeans.cluster_centers_[i]

    distances = np.linalg.norm(cluster_points - cluster_center, axis=1)

    min_distance_index = np.argmin(distances)

    closest_point_index = cluster_indices[min_distance_index]
    cluster_heart.append(closest_point_index)

    ######### The core step as the affine transformation ###################
    heart_seq.append(cluster_points[min_distance_index])
    other_seq.append(np.array([point for idx, point in enumerate(cluster_points) if idx != min_distance_index]))
    kb = []
    for p in other_seq[i]:
      pp = p.reshape(-1,1)
      pp0 = heart_seq[i].reshape(-1,1)
      pp1 = np.concatenate((pp0, np.ones((len(pp0), 1))), axis=1)
      pp2 = np.linalg.pinv(pp1)
      kb.append(pp2@pp)

    kb_colc.append(sum(kb))

    print("Cluster", i)
    print("Closest point index:", closest_point_index)
    print("Distance from center:", distances[min_distance_index])
    print(cluster_indices,cluster_indices.shape)
    cluster_coordinates = cor_wt[cluster_indices]
    plt.scatter(cluster_coordinates[:, 0], cluster_coordinates[:, 1],
                color=cmap(i),
                marker=markers[i],
                s = 60,
                label=f'Cluster {i + 1}')

plt.xlabel('Longitude',fontsize=18)
plt.ylabel('Latitude',fontsize=18)

plt.xticks(fontsize=18, rotation=0)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.legend(fontsize=15)
plt.grid(True)

plt.show()

print(cluster_heart)
print(kb_colc)
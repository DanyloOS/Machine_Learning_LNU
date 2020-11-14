import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


my_centers = [[20, 25], [60, 10], [10, 82]]
dots, my_clusters = make_blobs(n_samples=600, centers=my_centers, cluster_std=10, random_state=0)
df = pd.DataFrame(dots, columns=['X', 'Y'])


km = KMeans(n_clusters=3).fit(df)
k_centers = km.cluster_centers_
k_clusters = km.labels_
df['Cluster'] = k_clusters


clusters_group = df.groupby('Cluster')
print("\nDescribe: \n", clusters_group.describe().T)


plt.scatter(df.X, df.Y, c=k_clusters.astype(float), s=20, alpha=0.5)
plt.scatter(k_centers[:, 0], k_centers[:, 1], c='red', s=50, alpha=1)
plt.xlabel("X", size=20)
plt.ylabel("Y", size=20)
plt.title("n_clusters=3")
plt.show()

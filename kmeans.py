import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


x, y = make_blobs(n_samples=100, centers=4, n_features=2, cluster_std=[1, 1.5, 2, 2], random_state=7)

df = pd.DataFrame({
    'x1': x[:, 0],
    'x2': x[:, 1],
    'y': y
})


def plot_2d_clusters(x, y, ax):
    y_uniques = pd.Series(y).unique()

    for _ in y_uniques:
        x[y == _].plot(
            title=f'{len(y_uniques)} Clusters',
            kind='scatter',
            x='x1',
            y='x2',
            marker=f'${_}$',
            ax=ax
        )
        plt.show()


fig, ax = plt.subplots(1, 1, figsize=(15, 10))
x, y = df[['x1', 'x2']], df['y']
plot_2d_clusters(x, y, ax)

kmeans = KMeans(n_clusters=5, random_state=7)

y_prediction = kmeans.fit_predict(x)

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
plot_2d_clusters(x, y, axs[0])
plot_2d_clusters(x, y_prediction, axs[1])

axs[0].set_title(f'Actual {axs[0].get_title()}')
axs[1].set_title(f'K-means {axs[1].get_title()}')


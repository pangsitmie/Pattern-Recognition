import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kmeans_sample import KMeans

X, y = make_blobs(centers = 2, n_samples=500, n_features =2, shuffle=True, random_state = 30)
print(X.shape)

clusters = len(np.unique(y))
print(clusters)

k = KMeans(K=2, max_iters = 150, plot_steps=True)
y_pred = k.predict(X)

k.plot()
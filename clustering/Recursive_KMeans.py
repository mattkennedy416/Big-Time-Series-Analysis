# well don't think this is really smart to do, stop going down this path for now
#
# # so I have an idea about creating a recursive k-means where we start with two clusters,find if there's outliers,
# # and then split and retrain on the new data until an optimal number of clusters is reached?
#
# # maybe lets follow what dbscan does and have a cluster be -1 if it doesn't fit into any of the other clusters
#
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# import numpy as np
#
# class Recursive_KMeans():
#
#
#     def __init__(self, maxClusters=100):
#         self.maxClusters = maxClusters
#
#
#     def _clusterBounds(self, clusterData):
#         upperBound = np.zeros(shape=(clusterData.shape[1],))
#         lowerBound = np.zeros(shape=(clusterData.shape[1],))
#
#         for col in range(clusterData.shape[1]):
#             upperBound[col] = np.percentile(clusterData[:,col], 75)
#             lowerBound[col] = np.percentile(clusterData[:,col], 25)
#
#         # count up the percentage of time that each sample is outside the core 50% bounds
#         maskOutsideBounds = (clusterData > upperBound).astype(int) + (clusterData<lowerBound).astype(int)
#         percOutsideBounds = 100*np.sum(maskOutsideBounds, axis=1)/clusterData.shape[1]
#
#
#     def _clusterDensity(self, clusterData):
#         std = np.zeros(shape=(clusterData.shape[1]))
#
#         for col in range(clusterData.shape[1]):
#             std[col] = np.std(clusterData[:,col])
#
#         return np.mean(std)
#
#
#
#     def fit(self, data):
#
#         for n_clusters in range(2,20):
#             model = KMeans(init='k-means++', n_clusters=n_clusters)
#             model.fit(data)
#
#             clusterDensities = []
#             for cluster in set(model.labels_):
#                 clusterData = data[model.labels_==cluster]
#                 #self._clusterBounds( )
#                 clusterDensities.append(  self._clusterDensity(clusterData)  )
#             print(cluster, clusterDensities)
#
#             plt.plot( len(clusterDensities)*[n_clusters], clusterDensities, '*' )
#
#         plt.show()
#
#
#
#         print('')
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# if __name__ == '__main__':
#     from data.load import load
#
#     data = load().ekg()[:10000]  # already univariate
#
#     windowSize = 100
#
#     start = int(windowSize / 2)
#     end = data.shape[0] - int(windowSize / 2)
#     leftOffset = int(windowSize / 2)
#     rightOffset = int(windowSize / 2)
#
#     windows = np.zeros(shape=(end - start, windowSize))
#     stfts = np.zeros(shape=(end - start, windowSize))
#     for n in range(start, end):
#         windows[n - start, :] = data[n - leftOffset:n + rightOffset]
#         stfts[n - start, :] = np.fft.fft(np.hamming(windowSize) * windows[n - start, :])
#
#     model = Recursive_KMeans()
#     model.fit(stfts)
#
#

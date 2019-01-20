

from data.load import load

from sklearn.cluster import KMeans
import numpy as np

import matplotlib.pyplot as plt

data = load().TICC()[:,0] # just do univariate for now
data = load().ekg()[:10000] # already univariate
#data = load().s_and_p500()[:,0]


windowSize = 100

start = int(windowSize/2)
end = data.shape[0]-int(windowSize/2)
leftOffset = int(windowSize/2)
rightOffset = int(windowSize/2)


windows = np.zeros(shape=(end-start,windowSize))
stfts = np.zeros(shape=(end-start, windowSize))
for n in range(start, end):
    windows[n-start,:] = data[n-leftOffset:n+rightOffset]
    stfts[n-start,:] = np.fft.fft( np.hamming(windowSize)*windows[n-start,:])




clusterer = KMeans(n_clusters=10)
clusterer.fit(stfts)

# can we characterize




plt.plot(data)

labelsX = np.array(range(start,end))
plt.plot(labelsX, 100*(clusterer.labels_-5))
plt.show()



# for cluster, center in enumerate(clusterer.cluster_centers_):
#     sizeCluster = sum(clusterer.labels_ == cluster)
#     # plt.plot(center)
#     # plt.title(sizeCluster)
#     # plt.show()
#
#     clusterWindows = windows[clusterer.labels_ == cluster,:]
#     for n in range(100):
#         plt.plot(clusterWindows[n,:])
#
#     plt.show()


















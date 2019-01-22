

from data.load import load
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from external.sklearn.OPTICS import OPTICS
import numpy as np
import scipy

data = load().ekg()[0:1000]

corrThreshold = 0.90


windowSize = 100

for windowSize in range(10,250,10):
    leftOffset = int(windowSize/2)
    rightOffset = int(windowSize/2)

    start = leftOffset
    end = data.shape[0] - rightOffset

    windows = np.zeros(shape=(end - start, windowSize))
    #stft = np.zeros(shape=(end - start, windowSize))
    for n in range(start, end):
        windowValues = data[n - leftOffset:n + rightOffset]
        windows[n-start, :] = (windowValues - np.mean(windowValues)) / np.std(windowValues)

    model = OPTICS()
    results = model.fit_predict(windows)

    #print('Found', set(results), 'clusters')

    clusterCenters = np.zeros(shape=(len(set(model.labels_)), windowSize))
    for label in set(model.labels_):
        clusterData = windows[results==label,:]

        if label != -1:
            clusterCenters[label,:] = np.mean(clusterData,axis=0)


        # for row in range(clusterData.shape[0]):
        #     plt.plot(clusterData[row,:])
        # plt.title('Cluster '+str(label)+' with '+str(clusterData.shape[0])+' samples')
        # plt.show()



    # sig = np.repeat([0., 1., 1., 0., 1., 0., 0., 1.], 128)
    # sig_noise = sig + np.random.randn(len(sig))
    # corr = signal.correlate(sig_noise, np.ones(128), mode='same') / 128

    from scipy import signal

    crossCorrMatrix = np.zeros(shape=(clusterCenters.shape[0], clusterCenters.shape[0]))

    for n in range(clusterCenters.shape[0]):
        for m in range(clusterCenters.shape[0]):
            center1 = clusterCenters[n,:]
            center2 = clusterCenters[m,:]
            crossCorr = signal.correlate(center1, center2)/clusterCenters.shape[1]

            crossCorrMatrix[n,m] = np.max(crossCorr)


    # plt.matshow(crossCorrMatrix)
    # plt.show()



    # now we want to find the unique clusters, merging together cluster centers which have maximum cross correlations above our threshold
    uniqueClusters = []

    for cluster in range(clusterCenters.shape[0]):
        similarClusters = np.array(range(crossCorrMatrix.shape[1]))[crossCorrMatrix[cluster,:] > corrThreshold]

        alreadyHaveSimilarCluster = False
        for simCluster in similarClusters:
            if simCluster in uniqueClusters:
                alreadyHaveSimilarCluster = True
                break

        if not alreadyHaveSimilarCluster:
            uniqueClusters.append(cluster)

    print('Width:', windowSize, 'found', len(uniqueClusters),'unique clusters')



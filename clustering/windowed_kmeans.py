
# developed from: http://amid.fish/anomaly-detection-with-k-means-clustering

import matplotlib.pyplot as plt

import numpy as np
from scipy import stats, signal, spatial
from sklearn.cluster import KMeans


class Windowed_KMeans():
    # univariate clustering

    def __init__(self, window_size=30, training_slide_len=5, n_clusters=100, windowWeightingMethod='sine'):
        self.window_size = window_size
        self.training_slide_len = training_slide_len
        self.n_clusters = n_clusters

        self.windowWeightingMethod = windowWeightingMethod

        self.clusterer = KMeans(n_clusters = self.n_clusters)


    def _windowWeighting(self):

        if self.windowWeightingMethod == 'sine':
            # multiply each window by a sine wave
            window_rads = np.linspace(0, np.pi, self.window_size)
            window = np.sin(window_rads) ** 2
        else:
            raise NotImplementedError()

        return window



    def _generateSegments(self, data, slide_len):
        segments = []
        for start_pos in range(0, len(data), slide_len):
            end_pos = start_pos + self.window_size
            # make a copy so changes to 'segments' doesn't modify the original ekg_data
            segment = np.copy(data[start_pos:end_pos])
            # if we're at the end and we've got a truncated segment, drop it
            if len(segment) != self.window_size:
                continue
            segments.append(segment)

        print("Produced %d waveform segments" % len(segments))



        windowed_segments = []
        for segment in segments:
            windowed_segment = np.copy(segment) * self._windowWeighting()
            windowed_segments.append(windowed_segment)

        return np.array(windowed_segments)


    def _findClusterCorrelations(self, clusterCenters):

        # lets iterate through all pairs and find the maximum cross-correlation, where a large max correlation
        # would imply that these two clusters contain redundant information

        weightWindow = self._windowWeighting()
        unweightedClusterCenters = np.zeros(shape=clusterCenters.shape)
        for row in range(clusterCenters.shape[0]):
            unweightedClusterCenters[row,:] = clusterCenters[row,:] / weightWindow
            unweightedClusterCenters[row, [0,self.window_size-1]] = unweightedClusterCenters[row, [1,self.window_size-2]] # otherwise the ends will be nans from the divide-by-zero

        normalizedWeightedClusterCenters = np.zeros(shape=clusterCenters.shape)
        for row in range(clusterCenters.shape[0]):
            vals = clusterCenters[row,:]
            normalizedWeightedClusterCenters[row,:] = (vals - np.mean(vals)) / np.std(vals)

        normalizedUnweightedClusterCenters = np.zeros(shape=clusterCenters.shape)
        for row in range(clusterCenters.shape[0]):
            vals = unweightedClusterCenters[row,:]
            normalizedUnweightedClusterCenters[row,:] = (vals - np.mean(vals)) / np.std(vals)




        clusterDistances = np.zeros(shape=(len(clusterCenters), len(clusterCenters)))
        for n in range(len(clusterCenters)):
            for m in range(len(clusterCenters)):
                if n == m:
                    clusterDistances[n,m] = 0
                else:
                    crossCor = signal.correlate(normalizedUnweightedClusterCenters[n,:], normalizedUnweightedClusterCenters[m,:]) / clusterCenters.shape[1]

                    offset = np.argmax(crossCor) - (self.window_size-1)

                    if offset < 0:
                        # means cluster n is left of cluster m -- chop of start of cluster m and end of cluster n
                        cluster1 = normalizedUnweightedClusterCenters[ n, 0 : offset ]
                        cluster2 = normalizedUnweightedClusterCenters[ m, abs(offset) : ]

                    else:
                        cluster1 = normalizedUnweightedClusterCenters[ n, offset : ]
                        cluster2 = normalizedUnweightedClusterCenters[ m, 0:self.window_size - offset ]


                    clusterDistances[n,m] = np.linalg.norm(cluster1 - cluster2)

        # can we combine these together in a hierarchy?
        distance = 1.0
        hierarchy = []
        for n in range(clusterDistances.shape[0]):
            for m in range(clusterDistances.shape[1]):
                if clusterDistances[n,m] < distance:
                    pass

        # well I don't really know where I'm going with this any more
        # I'm imagining a re-combination of clusters with some hierarchical structure to get the most fundamental
        # and distinct segments of the time series


        plt.matshow(clusterDistances)
        plt.show()


    def fit(self, data):

        windowed_segments = self._generateSegments(data, slide_len=self.training_slide_len)

        segmentClusters = self.clusterer.fit_predict(windowed_segments)

        self._findClusterCorrelations(self.clusterer.cluster_centers_)


        print('done fitting')

        # calculate the cluster distributions
        for cluster, cluster_center in enumerate(self.clusterer.cluster_centers_):
            clusterSamples = windowed_segments[segmentClusters==cluster]

            numSamples = clusterSamples.shape[0]

            # lets find the 10th and 90th percentile value at each point in time?
            upperVal = np.zeros(shape=(clusterSamples.shape[1]))
            lowerVal = np.zeros(shape=(clusterSamples.shape[1]))
            for n in range(clusterSamples.shape[1]):
                upperVal[n] = stats.scoreatpercentile(clusterSamples[:,n], 90)
                lowerVal[n] = stats.scoreatpercentile(clusterSamples[:,n], 10)

            # import matplotlib
            # matplotlib.use('TKAgg')
            import matplotlib.pyplot as plt

            # for n in range(numSamples):
            #     plt.plot(clusterSamples[n,:])

            plt.plot(cluster_center)
            plt.plot(upperVal)
            plt.plot(lowerVal)
            plt.title(numSamples)
            plt.show()

            print('')



    def predict(self, data):

        windowed_segments = self._generateSegments(data, slide_len=1) # generate a window for each point
        results = self.clusterer.predict(windowed_segments)

        return results





if __name__ == '__main__':
    from data.load import load

    data = load().ekg()

    model = Windowed_KMeans()
    model.fit(data[0:10000])

    results = model.predict(data[10000:12000])






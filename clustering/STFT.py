



from sklearn.cluster import KMeans
import numpy as np






class STFT_Cluster():

    def __init__(self, n_clusters, windowSize, windowingType='center'):

        self.windowSize = windowSize
        self.n_clusters = n_clusters
        self.clusterer = KMeans(n_clusters=self.n_clusters)

        self.labels_ = None

        if windowingType == 'center':
            self.windowingType = 'center'
            self.leftOffset = int(self.windowSize/2)
            self.rightOffset = int(self.windowSize/2)
        elif windowingType == 'left':
            self.windowingType = 'left'
            self.leftOffset = 0
            self.rightOffset = self.windowSize
        else:
            raise ValueError('Unknown windowing type')


    def _createSTFTSWindows(self, data):
        start = self.leftOffset
        end = data.shape[0] - self.rightOffset

        #windows = np.zeros(shape=(end - start, windowSize))
        stft = np.zeros(shape=(end - start, self.windowSize))
        for n in range(start, end):
            windowVals = data[n - self.leftOffset:n + self.rightOffset]
            stft[n - start, :] = np.fft.fft(np.hamming(self.windowSize) * windowVals)

        return stft


    def _genLabelsFromModel(self, model):
        # need to append the labels with nans
        # note that we don't want to do this in-place because in predicting new data we'll NOT want to store these internally
        return self._genLabelsFromLabels(model.labels_)

    def _genLabelsFromLabels(self, labels):
        return np.array(self.leftOffset * [np.nan] + list(labels) + self.rightOffset * [np.nan])



    def fit(self, data):
        stft = self._createSTFTSWindows(data)
        self.clusterer.fit(stft)

        self.labels_ = self._genLabelsFromModel(model=self.clusterer)


    def fit_predict(self, data):

        self.fit(data)
        self.labels = self.clusterer.labels_

        return self._genLabelsFromModel(model=self.clusterer)


    def predict(self, data):
        # lets also implement a method for predicting on new data

        stfs = self._createSTFTSWindows(data)

        newLabels = self.clusterer.predict(stfs)
        return self._genLabelsFromLabels(newLabels)





# from data.load import load
# import matplotlib.pyplot as plt
#
# data = load().TICC()[:,0] # just do univariate for now
# data = load().ekg()[:10000] # already univariate
# #data = load().s_and_p500()[:,0]
#
# windowSize = 100
#
# start = int(windowSize/2)
# end = data.shape[0]-int(windowSize/2)
# leftOffset = int(windowSize/2)
# rightOffset = int(windowSize/2)
#
#
# windows = np.zeros(shape=(end-start,windowSize))
# stfts = np.zeros(shape=(end-start, windowSize))
# for n in range(start, end):
#     windows[n-start,:] = data[n-leftOffset:n+rightOffset]
#     stfts[n-start,:] = np.fft.fft( np.hamming(windowSize)*windows[n-start,:])
#
#
#
#
# clusterer = KMeans(n_clusters=10)
# clusterer.fit(stfts)
#
#
# plt.plot(data)
#
# labelsX = np.array(range(start,end))
# plt.plot(labelsX, 100*(clusterer.labels_-5))
# plt.show()



if __name__ == '__main__':
    from data.load import load
    import matplotlib.pyplot as plt

    data = load().TICC()[:, 0]  # just do univariate for now
    data = load().ekg()  # already univariate


    model = STFT_Cluster(n_clusters=10, windowSize=100)
    model.fit_predict(data[0:5000])

    plt.figure()
    plt.plot(data[0:5000])
    plt.plot(100*(model.labels_-5))

    plt.figure()
    results = model.predict(data[5000:6000])
    plt.plot(data[5000:6000])

    plt.plot(100*(results-5))
    plt.show()




















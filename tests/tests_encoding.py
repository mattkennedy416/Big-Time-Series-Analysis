import numpy as np
from data.load import load
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



plotSamples = False




allSamples = load().auslan()

wantedLabels = ['how-1', 'how-2', 'how-3', 'answer-3']



samples = []


for label in wantedLabels:
    data = allSamples[label]

    sample = data[['x1','y1','z1']].values

    u, s, vh = np.linalg.svd(sample)

    print(label, s)

    samples.append(sample)


if plotSamples:
    for sample in samples:
        plt.figure()
        plt.plot(sample)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(x1, y1, z1)
        # plt.title(label)
    plt.show()


from data.load import load
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from distance.lcss import lcss
from distance.discret_frechet import discret_frechet
from distance.traj_dist.frechet import frechet
from distance.traj_dist.sspd import e_sspd

plotSamples = False

runLCSS = False
runDiscretFretchet = True
runFretchet = False
runSSPD = False


allSamples = load().auslan()

wantedLabels = ['how-1', 'how-2', 'how-3', 'answer-3']



samples = []


for label in wantedLabels:
    data = allSamples[label]

    #x1, y1, z1 = data['x1'].values, data['y1'].values, data['z1'].values

    sample = data[['x1','y1','z1']].values

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



if True in [runSSPD, runFretchet, runDiscretFretchet]:

    if runSSPD:
        method = e_sspd
    elif runFretchet:
        method = frechet
    elif runDiscretFretchet:
        method = discret_frechet

    compares = [(0,1), (0,3)]

    for a,b in compares:
        t0 = samples[a][:, 0:2]
        t1 = samples[b][:, 0:2]
        dist = method(t0, t1)

        print(a, b, dist)








if runLCSS:
    # note that eps is the distance limit in space, not in time
    compares = [(0,1), (0,3)]

    import numpy as np
    epss = np.linspace(0.01, 0.5, 25)

    dists = {}
    for comp in compares:
        dists[comp] = []

    # so yeah this works if we can find an appropriate eps

    for eps in epss:
        for a, b in compares:
            dist = lcss(samples[a][:,0:3], samples[b][:,0:3], eps=eps)
            #print(a, b, dist)

            dists[(a,b)].append(dist)

    for comp in compares:
        plt.plot(epss, dists[comp], label=str(comp))
    plt.legend()
    plt.show()





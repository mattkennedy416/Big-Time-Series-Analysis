
from external.simple_segment import segment
from external.simple_segment import fit
import numpy as np

def PLA(data, windowScale=None, maxError=None, windowingType='top down', segmentType='interpolate'):

    # Implemented via Nick Foubert's simple-segment package
    # https://github.com/NickFoubert/simple-segment

    if maxError is not None:
        pass

    else:

        if windowScale is None:
            windowScale = len(data)/100
            print('WARNING: Neither error of window scale specified, assuming window scale of', windowScale, 'points.')

        import pandas as pd

        df = pd.DataFrame(data)
        stds = df.rolling(windowScale).std()
        meanSTDs = np.mean( stds.values[windowScale:])
        maxError = 100*meanSTDs # not really sure why, but this seems to work. Fundamental algorithms need to be re-written to make a bit more sense



    error = fit.sumsquared_error

    if windowingType == 'top down':
        window = segment.topdownsegment
    elif windowingType == 'bottom up':
        window = segment.bottomupsegment
    elif windowingType == 'sliding':
        window = segment.slidingwindowsegment
    else:
        raise ValueError('Unknown windowing type')

    if segmentType == 'interpolate':
        segmenting = fit.interpolate
    elif segmentType == 'regression':
        segmenting = fit.regression
    else:
        raise ValueError('Unknown segmenting type')

    return window(data, segmenting, error, maxError)



if __name__ == '__main__':
    from data.load import load

    data = load().s_and_p500()
    data = load().TICC()

    segments = PLA(data[:,0], windowScale=1000, windowingType='bottom up', segmentType='interpolate')

    print(segments)

    import matplotlib.pyplot as plt
    plt.plot(data[:,0])

    for seg in segments:
        plt.plot( [seg[0], seg[2]], [seg[1], seg[3]], c='r')

    plt.show()











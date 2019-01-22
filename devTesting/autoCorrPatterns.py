
from data.load import load
import matplotlib.pyplot as plt
import numpy as np

data = load().ekg()[0:1000]


windowSize = 100
leftOffset = 50
rightOffset = 50

start = leftOffset
end = data.shape[0] - rightOffset

windows = np.zeros(shape=(end - start, windowSize))
#stft = np.zeros(shape=(end - start, windowSize))
for n in range(start, end):
    windows[n-start, :] = data[n - leftOffset:n + rightOffset]
    #stft[n - start, :] = np.fft.fft(np.hamming(windowSize) * windowVals)

print('')

corrThreshold = 0.9



















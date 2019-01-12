

import gframe
import numpy as np

size = 25

values = np.random.rand(size, size)
myFrame = gframe(values)

other = np.random.rand(size, size)

myFrame.concat(other)

gpuResults = myFrame._retreive_array()

cpuResults = np.concatenate((values, other), axis=0)

difference = gpuResults -cpuResults

print(gpuResults.shape)
print(cpuResults.shape)
print(gpuResults - cpuResults)
print('Max Error:', np.max(difference))



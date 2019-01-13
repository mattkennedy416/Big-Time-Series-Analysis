
from gframe import gframe
import numpy as np

size = 25

values = np.random.rand(size, size)
myFrame = gframe(values)

other = np.random.rand(size, size)

# myFrame._gpuAddThis(0,size,0,size,other)

gpuResults = np.zeros((25, 25))
for col in range(size):
    c = myFrame[:,col]
    a = myFrame[:, col] + other[:, col]
    gpuResults[:, col] = a
    print('')
    # myFrame[:,col] += other[:,col]

print("Python: done with computations, lets pull the data back")
#gpuResults = myFrame._retreive_array()
cpuResult = values + other

print(gpuResults - cpuResult)
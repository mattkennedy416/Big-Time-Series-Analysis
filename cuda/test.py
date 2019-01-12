import gutil
import numpy as np
import numpy.testing as npt


# def test():
#     arr = np.array([1,2,2,2], dtype=np.int32)
#     adder = gutil.gframe(arr)
#     adder.increment()
#
#     adder.retreive_inplace()
#     results2 = adder.retreive()
#
#     npt.assert_array_equal(arr, [2,3,3,3])
#     npt.assert_array_equal(results2, [2,3,3,3])
#
#     print('tests complete')


#arr = np.array([1,2,2,2], dtype=np.float32)
arr = np.random.rand(5,5).astype(np.float32)

print(arr)


# for now lets flatten out the arrays in python and pass them to the cpp
# should probably figure out how to pass 2d arrays to the cpp and just have the flattened version in cuda
numRows, numColumns = arr.shape
arr = arr.reshape(arr.size,)
frame = gutil.gframe(arr, numRows, numColumns)

frame.operateOnSection(2,4,1,5)





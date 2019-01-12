
import numpy as np
import gutil
from copy import copy


class gframe_slice():
    # note that this is a "fake" array, doesn't actually contain any data itself
    # just a reference to the parent object and how it should be sliced
    # need to make certain that the parent object is actually a reference and not being copied every time

    def __init__(self, parent, rows, columns):
        self.parent = parent
        self.rows = rows
        self.columns = columns

        if isinstance(self.rows, slice):
            if self.rows.start is None:
                self.this_lowerRow = 0
            else:
                self.this_lowerRow = self.rows.start

            if self.rows.stop is None:
                self.this_upperRow = self.parent.numRows
            else:
                self.this_upperRow = self.rows.stop

        elif isinstance(self.rows,(int, np.int)):
            self.this_lowerRow = self.rows
            self.this_upperRow = self.rows + 1

        else:
            self.this_lowerRow = min(self.rows)
            self.this_upperRow = max(self.rows) + 1


        if isinstance(self.columns, slice):
            if self.columns.start is None:
                self.this_lowerCol = 0
            else:
                self.this_lowerCol = self.columns.start

            if self.columns.stop is None:
                self.this_upperCol = self.parent.numCols
            else:
                self.this_upperCol = self.columns.stop

        elif isinstance(self.columns,(int, np.int)):
            self.this_lowerCol = self.columns
            self.this_upperCol = self.columns + 1

        else:
            self.this_lowerCol = min(self.columns)
            self.this_upperCol = max(self.columns) + 1


    def __add__(self, other): # and this is the +
        return self.parent._gpuOperation_thisOther('add', self.this_lowerRow, self.this_upperRow, self.this_lowerCol, self.this_upperCol, other, inPlace=False)


    def __iadd__(self, other): # so this is the +=
        self.parent._gpuOperation_thisOther('add', self.this_lowerRow, self.this_upperRow, self.this_lowerCol, self.this_upperCol, other, inPlace=True)



    def __setitem__(self, key, value):
        # this is to handle to case where we're probably use this the most, which would be the "+=" operation?
        if value is None:
            return None



class gframe():

    def __init__(self, array, columns=None):

        self.numRows = None
        self.numCols = None
        self.columns = columns
        self.array = array # we'll keep this here for debugging but shouldn't be permanently

        self.numRows, self.numColumns = array.shape
        arr = self.array.reshape(array.size, ).astype(np.float32)

        # so interestingly if this array object gets passed around too much in python before being passed to the
        # c++ class, the c++ class will segfault... probably not happy about how python is internally keeping track of the object?
        # fortunately passing a copy of this object to the c++ class seems to fix any potential issues
        # --> fixed? think this was caused by us trying to directly write to the original python variable in c++
        self._frame = gutil.gframe(arr, self.numRows, self.numColumns, self.columns)



    def __getattr__(self, item):
        pass


    def __getitem__(self, item):
        if isinstance(item, int):
            return gframe_slice(parent=self, rows=item[0], columns=item[1])
        elif isinstance(item, str):
            colInd = self._frame.getColumn(item)
            return gframe_slice(parent=self, rows=slice(None,None,None), columns=colInd)


    def _gpuOperation_thisOther(self, operationType, this_lowerRow, this_upperRow, this_lowerCol, this_upperCol, other, inPlace):
        # "other" is an arbitrary vector which needs to be copied up to the gpu

        other = other.reshape(other.size, ).astype(np.float32)
        operationType = operationType.encode()

        self._frame.gpuOperation_this( operationType, this_lowerRow, this_upperRow, this_lowerCol, this_upperCol, other, inPlace )

        # so we don't necessarily know who called this, but results should always be the same ...
        if not inPlace:
            return self._frame.retreive_results()


    def _gpuOperation_thisThis(self, operationType, this1_lowerRow, this1_upperRow, this1_lowerCol, this1_upperCol, this2_lowerRow, this2_upperRow, this2_lowerCol, this2_upperCol):
        # "other" is another subset of this frame and already in gpu memory
        pass



    def _retreive_results(self):
        return self._frame.retreive_results()


    def _retreive_array(self):
        return self._frame.retreive_array()


    def __add__(self, other):
        raise NotImplementedError()


    def __setitem__(self, key, value):
        if value is None:
            pass


    def concat(self, other, axis=0):
        otherShape = other.shape

        rows = otherShape[0]
        if len(otherShape) == 1:
            cols = 1
        else:
            cols = otherShape[1]

        other = other.reshape(other.size, ).astype(np.float32)
        self._frame.concat(other, rows, cols, axis)




if __name__ == '__main__':

    import numpy as np

    size = 2500
    cols = ['a','b','c','d','e']
    values = np.random.rand(size, len(cols))
    myFrame = gframe(values, cols)

    other = np.random.rand(size, size)

    a = myFrame['b']
    print(a)


    # next step is going to be getting the column keys up and running










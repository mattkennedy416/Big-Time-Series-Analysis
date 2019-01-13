
import numpy as np
import gts
from copy import copy


def _indexingParser(input, maxLength):
    # input can be a slice, integer, or consecutive list
    # note that non-consecutive lists are not yet supported

    if isinstance(input, slice):
        if input.start is None:
            this_lower = 0
        else:
            this_lower = input.start

        if input.stop is None:
            this_upper = maxLength
        else:
            this_upper = input.stop

    elif isinstance(input, (int, np.int)):
        this_lower = input
        this_upper = input + 1

    else:
        this_lower = min(input)
        this_upper = max(input) + 1

    return this_lower, this_upper


class gframe_slice():
    # note that this is a "fake" array, doesn't actually contain any data itself
    # just a reference to the parent object and how it should be sliced
    # need to make certain that the parent object is actually a reference and not being copied every time

    def __init__(self, parent, rows, columns):
        self.parent = parent
        self.rows = rows # lets actually allow the rows and columns to retain their original value
        self.columns = columns


        # self.this_lowerRow, self.this_upperRow = _indexingParser(self.rows, self.parent.numRows)
        # self.this_lowerCol, self.this_upperCol = _indexingParser(self.columns, self.parent.numColumns)

    def getUID(self):
        return self.parent.getUID()


    def __add__(self, other): # and this is the +
        return self.parent._gpuOperation_thisOther('add', self.rows, self.columns, other, inPlace=False)


    def __iadd__(self, other): # so this is the +=
        self.parent._gpuOperation_thisOther('add', self.rows, self.columns, other, inPlace=True)



    def __setitem__(self, key, value):
        # this is to handle to case where we're probably use this the most, which would be the "+=" operation?
        if value is None:
            return None



class grframe_rolling():
    # similar to gframe_slice above this is going to be a pseudo-wrapper which basically just contains instructions for the gpu to process on demand?

    def __init__(self, parent, width, rows=None, columns=None, method='center'):
        # if a gframe_slice is input as the parent, then we already have the rows and columns
        # otherwise if a gframe is given as the parent, then rows and columns need to be explicitly defined

        methodTypes = ('backward','center','forward')
        if method not in methodTypes:
            raise ValueError('Method must be in: '+str(methodTypes))

        self.parent = parent
        self.window = width

        if isinstance(parent, gframe_slice):
            # self.lowerRow = parent.this_lowerRow
            # self.upperRow = parent.this_upperRow
            # self.lowerCol = parent.this_lowerCol
            # self.upperCol = parent.this_upperCol
            self.rows = parent.rows
            self.columns = parent.columns

        elif isinstance(parent, gframe):
            if rows is None or columns is None:
                raise ValueError('Rows and Columns must be specified')
            self.rows = rows
            self.columns = columns
            # self.this_lowerRow, self.this_upperRow = _indexingParser(self.rows, self.parent.numRows)
            # self.this_lowerCol, self.this_upperCol = _indexingParser(self.columns, self.parent.numColumns)

        else:
            raise NotImplementedError('Unable to parse input of type '+str(type(parent)))


    def mean_SMA(self):
        pass

    def mean_EMA(self):
        pass

    def min(self):
        pass

    def max(self):
        pass


    def getUID(self):
        return self.parent.getUID()




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
        self._frame = gts.gframe(arr, self.numRows, self.numColumns, self.columns)



    def __getattr__(self, item):
        pass


    def __getitem__(self, item):
        if isinstance(item, int):
            return gframe_slice(parent=self, rows=item[0], columns=item[1])
        elif isinstance(item, str):
            colInd = self._frame.getColumn(item)
            return gframe_slice(parent=self, rows=slice(None,None,None), columns=colInd)
        elif isinstance(item, list): # list of column inds
            colInds = []
            for subitem in item:
                if isinstance(subitem, int):
                    colInds.append(subitem)
                elif isinstance(subitem, str):
                    colInd = self._frame.getColumn(subitem)
                    colInds.append(colInd)
            return gframe_slice(parent=self, rows=slice(None,None,None), columns=colInds)

        elif isinstance(item, tuple): # we should assume this is (rows, columns)
            return gframe_slice(parent=self, rows=item[0], columns=item[1])


    # def _gpuOperation_thisOther(self, operationType, this_lowerRow, this_upperRow, this_lowerCol, this_upperCol, other, inPlace):
    #     # "other" is an arbitrary vector which needs to be copied up to the gpu
    #
    #     other = other.reshape(other.size, ).astype(np.float32)
    #     operationType = operationType.encode()
    #
    #     self._frame.gpuOperation_this( operationType, this_lowerRow, this_upperRow, this_lowerCol, this_upperCol, other, inPlace )
    #
    #     # so we don't necessarily know who called this, but results should always be the same ...
    #     if not inPlace:
    #         return self._frame.retreive_results()


    def _gpuOperation_thisOther(self, operationType, this_rowSelection, this_colSelection, other, inPlace):
        # "other" is an arbitrary vector which needs to be copied up to the gpu

        other = other.reshape(other.size, ).astype(np.float32)
        operationType = operationType.encode()

        self._frame.gpuOperation_thisOther( operationType, this_rowSelection, this_colSelection, other, inPlace )

        # so we don't necessarily know who called this, but results should always be the same ...
        if not inPlace:
            return self._frame.retreive_results()


    def _gpuOperation_thisThis(self, operationType, this1_lowerRow, this1_upperRow, this1_lowerCol, this1_upperCol, this2_lowerRow, this2_upperRow, this2_lowerCol, this2_upperCol):
        # "other" is another subset of this frame and already in gpu memory
        pass


    def getUID(self):
        return self._frame.getUID()


    def _retreive_results(self):
        return self._frame.retreive_results()


    def _retreive_array(self):
        return self._frame.retreive_array()


    def __add__(self, other):
        raise NotImplementedError()


    def __setitem__(self, key, value):
        if value is None:
            pass
        else:
            self.concat(other=value, axis=1, columns=key)


    def concat(self, other, axis=0, columns=None):
        #TODO: handle the case where we have two gframes with columns in differing orders

        otherShape = other.shape

        rows = otherShape[0]
        if len(otherShape) == 1:
            cols = 1
        else:
            cols = otherShape[1]

        otherFlat = other.reshape(other.size, ).astype(np.float32)
        self._frame.concat(otherFlat, rows, cols, axis, columns)








if __name__ == '__main__':

    import numpy as np

    size = 2500
    cols = ['a','b','c','d','e']
    values = np.random.rand(size, len(cols))
    myFrame = gframe(values, cols)

    other = np.random.rand(size, size)


    b = myFrame[:,[0,1,2]]

    a = myFrame[['b', 'c']]
    print(a)

    myFrame['g'] = np.random.rand(2500)

    frameValues = myFrame._frame.retreive_array()

    print('')


    # next step is going to be getting the column keys up and running










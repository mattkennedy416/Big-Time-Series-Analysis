import numpy as np
cimport numpy as np
import uuid
from libcpp cimport bool # to enable support for bools?

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/manager.hh":
    cdef cppclass C_GPUUtil "GPUUtil":
        C_GPUUtil(np.int32_t*, int)
        void increment()
        void retreive()
        void retreive_to(np.int32_t*, int)


cdef extern from "src/gframe.hh":
    cdef cppclass C_gframe "gframe":
        C_gframe(np.float32_t*, int, int)
        void gpuOperation_thisOther(char*, np.int32_t*, int, np.int32_t*, int, float*, int, bool)
        void retreive_results(np.float32_t*)
        void retreive_results_shape(np.int32_t*)
        void retreive_array(np.float32_t*)
        void concat(np.float32_t*, int, int, int)
        void cpuSort(int, np.int32_t*)
        void gpuOperation_rolling(char*, int, char*, np.int32_t*, int, np.int32_t*, int)
        void gpuOperation_isnan(np.int32_t*, int, np.int32_t*, int)
        void gpuOperation_nan2num(np.int32_t*, int, np.int32_t*, int, bool)



cdef class GPUUtil:
    cdef C_GPUUtil* g
    cdef int dim1

    def __cinit__(self, np.ndarray[ndim=1, dtype=np.int32_t] arr):
        self.dim1 = len(arr)
        self.g = new C_GPUUtil(&arr[0], self.dim1)

    def increment(self):
        self.g.increment()

    def retreive_inplace(self):
        self.g.retreive()

    def retreive(self):
        cdef np.ndarray[ndim=1, dtype=np.int32_t] a = np.zeros(self.dim1, dtype=np.int32)

        self.g.retreive_to(&a[0], self.dim1)

        return a


cdef class gframe:
    cdef C_gframe* g
    cdef int numRows
    cdef int numColumns
    cdef dict _columnKeys
    cdef str uid

    def __cinit__(self, np.ndarray[ndim=1, dtype=np.float32_t] arr, int numRows, int numColumns, columnKeys):
        self.numRows = numRows
        self.numColumns = numColumns

        # alright well we aren't having much luck in passing a char array to the c++
        # lets instead just store the strings as a dictionary in python, pointing to the column number
        # and for now lets do this conversion on the cython side of things of header -> col#
        self._columnKeys = {}
        if columnKeys is not None:
            for col, key in enumerate(columnKeys):
                self._columnKeys[key] = col

        # and lets generate a unique id for this frame instance
        self.uid = str(uuid.uuid1())

        self.g = new C_gframe(&arr[0], self.numRows, self.numColumns)


    def getColumnKeys(self):
        return self._columnKeys


    def getColumn(self, colKey):
        if colKey in self._columnKeys:
            return self._columnKeys[colKey]
        else:
            return None


    def getUID(self):
        return self.uid


    # def operateOnSection(self, int minRow, int maxRow, int minCol, int maxCol):
    #     self.g.operateOnSection(minRow, maxRow, minCol, maxCol)


    # def gpuOperation_this(self, char* opType, int this_lowerRow, int this_upperRow, int this_lowerCol, int this_upperCol, np.ndarray[ndim=1, dtype=np.float32_t] other, bool inPlace):
    #     print('shit is broken yo')
    #
    #     #self.g.gpuOperation_this('add'.encode(), this_lowerRow, this_upperRow, this_lowerCol, this_upperCol, &other[0], len(other), inPlace)
    #     #self.g.gpuOperation_thisOther(opType, this_lowerRow, this_upperRow, this_lowerCol, this_upperCol, &other[0], len(other), inPlace)

    def _indexSelectionToArray(self, selection):

        # simplest way to convert slice to array would be np.array(range(selection.stop))[slice]
        # is there a better way? probably

        if isinstance(selection, slice):
            if selection.stop is None:
                stop = self.numRows
            else:
                stop = selection.stop
            selectionArray = np.array( range(stop) )[selection]
        elif isinstance(selection, int):
            selectionArray = np.array([selection])
        elif isinstance(selection, list):
            selectionArray = np.array(selection)
        elif isinstance(selection, np.ndarray):
            selectionArray = selection

        print('Selection:', selection)
        print('SelectionArray:', selectionArray)

        return selectionArray.astype(np.int32)


    def gpuOperation_thisOther(self, char* opType, this_rowSelection, this_colSelection, np.ndarray[ndim=1, dtype=np.float32_t] other, bool inPlace):

        cdef np.ndarray[ndim=1, dtype=np.int32_t] this_rowArray = self._indexSelectionToArray(this_rowSelection)
        cdef np.ndarray[ndim=1, dtype=np.int32_t] this_colArray = self._indexSelectionToArray(this_colSelection)

        print(this_rowArray)
        print(this_colArray)


        #self.g.gpuOperation_this('add'.encode(), this_lowerRow, this_upperRow, this_lowerCol, this_upperCol, &other[0], len(other), inPlace)
        self.g.gpuOperation_thisOther(opType, &this_rowArray[0], len(this_rowArray), &this_colArray[0], len(this_colArray), &other[0], len(other), inPlace)


    def rolling(self, char* opType, int width, char* method, rowSelection, colSelection):
        cdef np.ndarray[ndim=1, dtype=np.int32_t] rowArray = self._indexSelectionToArray(rowSelection)
        cdef np.ndarray[ndim=1, dtype=np.int32_t] colArray = self._indexSelectionToArray(colSelection)

        self.g.gpuOperation_rolling(opType, width, method, &rowArray[0], len(rowArray), &colArray[0], len(colArray))

        return self.retreive_results()



    def isnan(self, rowSelection, colSelection):
        cdef np.ndarray[ndim=1, dtype=np.int32_t] rowArray = self._indexSelectionToArray(rowSelection)
        cdef np.ndarray[ndim=1, dtype=np.int32_t] colArray = self._indexSelectionToArray(colSelection)

        self.g.gpuOperation_isnan(&rowArray[0], len(rowArray), &colArray[0], len(colArray))

        return self.retreive_results().astype(np.bool)


    def nan2num(self, rowSelection, colSelection, bool inPlace):
        cdef np.ndarray[ndim=1, dtype=np.int32_t] rowArray = self._indexSelectionToArray(rowSelection)
        cdef np.ndarray[ndim=1, dtype=np.int32_t] colArray = self._indexSelectionToArray(colSelection)

        self.g.gpuOperation_nan2num(&rowArray[0], len(rowArray), &colArray[0], len(colArray), inPlace)

        if inPlace:
            pass
        else:
            return self.retreive_results()




    def _generateUniqueHeader(self, keyBase):
        # keyBase is the requested value, iterate on this until we find something unique
        key = keyBase
        iter = 1
        while key in self._columnKeys:
            key = keyBase + '-' + str(iter)
            iter += 1
            if iter >= 1000:
                raise ValueError('Reached maxed allowable iterations for header generation')
        return key


    def sort(self, columnKey, ascending=True):
        cdef np.ndarray[ndim=1, dtype=np.int32_t] index = np.array(range(self.numRows)).astype(np.int32)
        colInd = self.getColumn(columnKey)

        self.g.cpuSort(colInd, &index[0])

        values = self.retreive_array()

        if ascending:
            return values[index,:]
        else:
            return values[index[::-1],:]





    def concat(self, np.ndarray[ndim=1, dtype=np.float32_t] newArray, int newNumRows, int newNumCols, int axis, columnKeys):
        # concatenate newArray to existing array in memory

        if axis == 0: # number of columns must be the same?
            if newNumCols != self.numColumns:
                raise ValueError('Cannot concatenate new values of shape('+str(newNumRows)+','+str(newNumCols)+') to gframe of shape('+str(self.numRows)+','+str(self.numColumns)+')')
        elif axis == 1: # number of rows must be the same
            if newNumRows != self.numRows:
                raise ValueError('Cannot concatenate new values of shape('+str(newNumRows)+','+str(newNumCols)+') to gframe of shape('+str(self.numRows)+','+str(self.numColumns)+')')


        self.g.concat(&newArray[0], newNumRows, newNumCols, axis)
        if axis == 0:
            self.numRows += newNumRows
        elif axis == 1:

            if len(self._columnKeys) > 0: # so we have existing headers, need to assign headers to the newly added columns

                for n in range(newNumCols):
                    newColInd = self.numColumns + n

                    if columnKeys is not None and n < len(columnKeys):
                        keyBase = columnKeys[n]
                    else:
                        keyBase = str(n)

                    key = self._generateUniqueHeader(keyBase)
                    self._columnKeys[key] = newColInd


            self.numColumns += newNumCols # add this after the above headers have been taken care of


    def retreive_results(self):
        cdef np.ndarray[ndim=1, dtype=np.int32_t] shapeArray = np.zeros(2, dtype=np.int32)

        self.g.retreive_results_shape(&shapeArray[0])

        returnNumRows = shapeArray[0] # I think we want this mode of operation to return a subset of the data (size of "other") so we need to figure out what that size is
        returnNumCols = shapeArray[1]

        print("Found results array to be shape:", returnNumRows, returnNumCols)

        cdef np.ndarray[ndim=1, dtype=np.float32_t] flattenedArray = np.zeros(returnNumRows*returnNumCols, dtype=np.float32)
        self.g.retreive_results(&flattenedArray[0])


        # cdef np.ndarray[ndim=2, dtype=np.float32_t] unflattenedArray = np.zeros((returnNumRows, returnNumCols), dtype=np.float32)
        # unflattenedArray = flattenedArray.reshape((returnNumRows, returnNumCols))
        # return unflattenedArray

        # I think we want to return this as 1D if it's 1D (rather than a 1xN matrix)
        if returnNumCols == 1 or returnNumRows == 1:
            # well if it's 1D, should be equal to the already flattened array right?
            return flattenedArray

        #else:
        cdef np.ndarray[ndim=2, dtype=np.float32_t] unflattenedArray = np.zeros((returnNumRows, returnNumCols), dtype=np.float32)
        unflattenedArray = flattenedArray.reshape((returnNumRows, returnNumCols))
        return unflattenedArray



    def retreive_array(self):
        returnNumRows = self.numRows
        returnNumCols = self.numColumns

        print('Cython: retreiving array of shape(', returnNumRows, returnNumCols,')')

        cdef np.ndarray[ndim=1, dtype=np.float32_t] flattenedArray = np.zeros(returnNumRows*returnNumCols, dtype=np.float32)
        self.g.retreive_array(&flattenedArray[0])

        # I think we want to return this as 1D if it's 1D (rather than a 1xN matrix)
        if returnNumCols == 1 or returnNumRows == 1:
            # well if it's 1D, should be equal to the already flattened array right?
            return flattenedArray

        #else:
        cdef np.ndarray[ndim=2, dtype=np.float32_t] unflattenedArray = np.zeros((returnNumRows, returnNumCols), dtype=np.float32)
        unflattenedArray = flattenedArray.reshape((returnNumRows, returnNumCols))
        return unflattenedArray






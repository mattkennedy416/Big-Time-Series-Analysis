class gframe {
  // pointer to the GPU memory where the array is stored
  float* array_device;
  // pointer to the CPU memory where the array is stored
  float* array_host;

  float* results_device; // lets see if we can easily do a not-in-place operation where array is left untouched and results is written to
  float* results_host;
  int results_totalRows;
  int results_totalColumns;

  int this_totalRows;
  int this_totalColumns;
  int arraySize;

  bool array_device_modifiedSinceLastHostCopy; // only copy data back from GPU when requested and the host copy is different from device copy


public:
  /* By using the swig default names INPLACE_ARRAY1, DIM1 in the header
     file (these aren't the names in the implementation file), we're giving
     swig the info it needs to cast to and from numpy arrays.
     
     If instead the constructor line said
       gframe(int* myarray, int length);

     We would need a line like this in the swig.i file
       %apply (int* ARGOUT_ARRAY1, int DIM1) {(int* myarray, int length)}
   */

  gframe(float* INPLACE_ARRAY1, int DIM1, int DIM2); // constructor (copies to GPU)

  ~gframe(); // destructor

  void concat(float* newArray, int newNumRows, int newNumCols, int axis);

  void gpuOperation_this(char* operationType, int this_lowerRow, int this_upperRow, int this_lowerCol, int this_upperCol, float* other, int otherLength, bool inPlace);

  void operateOnSection(int minRow, int maxRow, int minCol, int maxCol);

  void updateHostFromGPU();

  void retreive_results(float* numpyArray);
  void retreive_results_shape(int* shapeArray);
  void retreive_array(float* numpyArray);


//  void increment(); // does operation inplace on the GPU
//
//  void retreive(); //gets results back from GPU, putting them in the memory that was passed in
//  // the constructor
//
//  //gets results back from the gpu, putting them in the supplied memory location
//  void retreive_to(int* INPLACE_ARRAY1, int DIM1);


};

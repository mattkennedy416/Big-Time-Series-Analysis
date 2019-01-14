/*
This is the central piece of code. This file implements a class
(interface in gframe.hh) that takes data in on the cpu side, copies
it to the gpu, and exposes functions (increment and retreive) that let
you perform actions with the GPU

This class will get translated into python via swig
*/

#include <gframe_kernel.cu>
#include <rolling_kernel.cu>
#include <gframe.hh>
#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <stdexcept>
#include <string.h>
using namespace std;


//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}


gframe::gframe (float* array_, int numRows_, int numColumns_) {
//  array_host = array_host_;
//  length = length_;
//  int size = length * sizeof(int);
//  cudaError_t err = cudaMalloc((void**) &array_device, size);
//  assert(err == 0);
//  err = cudaMemcpy(array_device, array_host, size, cudaMemcpyHostToDevice);
//  assert(err == 0);

	printf("Initialized object!\n");
	array_device_modifiedSinceLastHostCopy = false;

	// so we want to keep identical copies in both host memory and gpu memory

	this_totalRows = numRows_;
	this_totalColumns = numColumns_;
	arraySize = this_totalRows*this_totalColumns*sizeof(float);

	array_host = (float* )malloc(arraySize);
	for (int n=0; n<this_totalRows*this_totalColumns; n++)
		array_host[n] = array_[n];

	printf("array_host has size: (%i,%i)\n", this_totalRows, this_totalColumns);

	cudaError_t err = cudaMalloc((void**) &array_device, arraySize);
	assert(err==0);
	err = cudaMemcpy(array_device, array_host, arraySize, cudaMemcpyHostToDevice);
	cudaCheckError();

}





//void gframe::operateOnSection(int minRow, int maxRow, int minCol, int maxCol) {
//
//	printf("Operating on subsection ...\n");
//	kernal_printSubsection<<<1,1>>>(array_device, minRow, maxRow, minCol, maxCol, this_totalRows, this_totalColumns);
//	//testKernel<<<10,32>>>();
//	cudaCheckError();
//	cudaDeviceSynchronize();
//	printf("and we aren't crashing...\n");
//
//}


void gframe::cpuSort(int colInd, int* index ) {
	// index should be input as original index (ie not dummy values!)
	// will be reordered as needed
	
	printf("starting cpu sort\n");
	
	updateHostFromGPU();
	
	int i, j;
	for (i=0; i<this_totalRows-1; i++)
	{
		for (j=0; j<this_totalRows-1; j++)
		{
			int ind1 = index[j]*this_totalColumns + colInd;
			int ind2 = (index[j+1])*this_totalColumns + colInd;
			
			float val1 = array_host[ind1];
			float val2 = array_host[ind2];
			
			//printf("(i,j)=(%i,%i) -> index=(%i,%i) -> values=(%f,%f)\n", i,j, index[j], index[j+1], val1, val2);
			
			if (val1 > val2)
			{
				int temp = index[j];
				index[j] = index[j+1];
				index[j+1] = temp;
			}
			
		}
	}
		
}



void gframe::gpuOperation_rolling(char* operationType, int width, char* method, int* rowArray, int rowArrayLength, int* colArray, int colArrayLength) {
	
	
	printf("and we're rolling! abcd\n");
	
	int methodID;
	if (strcmp(method, "backward") == 0)
		methodID = 0;
	else if (strcmp(method, "center") == 0)
		methodID = 1;
	else
	{
		printf("Unknown rolling method");
		return;
	}
		
	
	
	// so I think we always want this operation to be not-in-place
	// need to cudaMalloc an array for results storing
	
	int numElements = rowArrayLength*colArrayLength;
	size_t sizeResults = numElements*sizeof(float);
	
	//float* results_device;
	printf("Allocating results_deivce to size %i\n", sizeResults);
	cudaMalloc((void **) &results_device, sizeResults);
	// data should already be copied up there though
	
	// so do we just now need to do our strcmps and everything else is handled in-kernel?
	// well we do first need to figure out how many blocks and threads based on the size of the data
	
	// and I think we need to cudaMalloc and copy up all arrays ...
	int* rowArray_device;
	int* colArray_device;
	cudaMalloc((void **) &rowArray_device, rowArrayLength*sizeof(int));
	cudaMalloc((void **) &colArray_device, colArrayLength*sizeof(int));
	cudaMemcpy(rowArray_device, rowArray, rowArrayLength*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(colArray_device, colArray, colArrayLength*sizeof(int), cudaMemcpyHostToDevice);
	
	
	dim3 threadsPerBlock(32,32);
	dim3 numBlocks(rowArrayLength/threadsPerBlock.x + 1, colArrayLength/threadsPerBlock.y + 1);
	
	printf("Lauching rolling kernel\n");
	if (strcmp(operationType, "mean_SMA") == 0)
	{
		kernel_meanSMA<<<numBlocks, threadsPerBlock>>>(array_device, methodID, width, rowArray_device, rowArrayLength, colArray_device, colArrayLength, this_totalColumns, this_totalRows, results_device);
		//(float* array_device, int window, int* rowArray, int rowArrayLength, int* colArray, int colArrayLength, float* results) 
	}
	
	cudaDeviceSynchronize();
	
	
	// lets actually just use the standard last-result retrieval
	// free(results_host);
	results_host = (float* )malloc(sizeResults);
	cudaMemcpy(results_host, results_device, sizeResults, cudaMemcpyDeviceToHost);
	cudaFree(results_device);
	
	printf("Setting results size information to: (%i,%i)\n", rowArrayLength, colArrayLength);
	results_totalRows = rowArrayLength;
	results_totalColumns = colArrayLength;
	
	
	
}


//void gframe::gpuOperation_thisOther(char* operationType, int this_lowerRow, int this_upperRow, int this_lowerCol, int this_upperCol, float* other, int otherLength, bool inPlace) {
void gframe::gpuOperation_thisOther(char* operationType, int* this_rowArray, int this_rowArrayLength, int* this_colArray, int this_colArrayLength, float* other, int otherLength, bool inPlace) {

	printf("Operation Type: %s\n", operationType);

	// add "other" to the values of this object stored in host and device memory
	// assume they're both float32s for now

	// use the lower/upper terminology since the upper values are not actually included (stops at previous index)
//	int section_numRows = (this_upperRow-this_lowerRow);
//	int section_numCols = (this_upperCol-this_lowerCol);
	int section_numRows = this_rowArrayLength;
	int section_numCols = this_colArrayLength;

	int this_requestedSize = (section_numRows)*(section_numCols);

	//printf("This size: (%i,%i,%i,%i)->(%i,%i)=%i -- other size: %i\n", this_lowerRow, this_upperRow, this_lowerCol,this_upperCol, section_numRows, section_numCols, this_requestedSize, otherLength);

	if (this_requestedSize != otherLength && otherLength != 1)
	{
		// though we do need to account for single value operations
		printf("Different sized arrays trying to be added together! %i vs %i\n", this_requestedSize, otherLength);
		return; // need to come up with a better way to handle invalid indexing, maybe this should actually be done on the python side
	}

	
	cudaError_t err;

	// I guess assume they've been flattened in the same way?
	// so lets first copy the new data up to the GPU, then add it

//	int other_totalCols = section_numCols; // assume they're the same shape
//	int other_totalRows = section_numRows;
	
	
	
	float* other_device;
//	if (otherLength == 1)
//	{
//		printf("Single value other operation\n");
//		other_device = other;
//	}
//	else
//	{
		err = cudaMalloc((void **) &other_device, otherLength*sizeof(float));
		err = cudaMemcpy(other_device, other, otherLength*sizeof(float), cudaMemcpyHostToDevice);
	//}
	



	if (inPlace == false)
	{
		err = cudaMalloc((void **) &results_device, section_numRows*section_numCols*sizeof(float));
		results_totalRows = section_numRows;
		results_totalColumns = section_numCols;
		// we don't need to copy anything up right?
	}
	else
	{
		// I think the cuda kernel doesn't like being passed null pointers
		err = cudaMalloc((void **) &results_device, 2*sizeof(float));
	}
	
	
	int* this_rowArray_device;
	int* this_colArray_device;
	cudaMalloc((void **) &this_rowArray_device, this_rowArrayLength*sizeof(int));
	cudaMalloc((void **) &this_colArray_device, this_colArrayLength*sizeof(int));
	cudaMemcpy(this_rowArray_device, this_rowArray, this_rowArrayLength*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(this_colArray_device, this_colArray, this_colArrayLength*sizeof(int), cudaMemcpyHostToDevice);
	
//	for (int n=0; n<this_rowArrayLength; n++)
//		printf("CPU: (%i,%i)\n", this_rowArray[n], this_colArray[n]);
//	printf("Total columns: %i\n", this_totalColumns);
	

	dim3 threadsPerBlock(32,32);
	dim3 numBlocks(section_numRows/threadsPerBlock.x + 1, section_numCols/threadsPerBlock.y + 1);
	printf("Num Blocks = (%i,%i)\n", numBlocks.x, numBlocks.y);
	
	
	int operationID; // so cuda can't parse strings easily, lets convert our string ID to an integer for the kernel
	if (strcmp(operationType, "add") == 0)
		operationID = 0;
	else if (strcmp(operationType, "subtract") == 0)
		operationID = 1;
	else if (strcmp(operationType, "multiply") == 0)
		operationID = 2;
	else if (strcmp(operationType, "divide") == 0)
		operationID = 3;
	else if (strcmp(operationType, "greaterThan") == 0)
		operationID = 4;
	
	
	

	
	kernel_gpuBasicOps<<<numBlocks, threadsPerBlock>>>(operationID, array_device, this_totalColumns, this_totalRows, this_rowArray_device, this_rowArrayLength, this_colArray_device, this_colArrayLength, other_device, section_numCols, section_numRows, otherLength, inPlace, results_device);
	
	cudaCheckError();

	cudaDeviceSynchronize();
	cudaCheckError();

	if (inPlace == false)
	{
		printf("Not in place -- copying results_device back to results_host ...\n");
		results_host = (float* )malloc(section_numRows*section_numCols*sizeof(float)); // need to malloc this before copying back
		cudaMemcpy(results_host, results_device, section_numRows*section_numCols*sizeof(float), cudaMemcpyDeviceToHost);
		cudaCheckError();
		// this should affect the array_device_modifiedSinceLastHostCopy (leave at current value either way)
		printf("Done copying!\n");
	}
	else
		array_device_modifiedSinceLastHostCopy = true;



	cudaFree(results_device); // store in results_host for retreival and free the device array
	cudaFree(other_device);

}


/*
void gframe::gpuOperation_thisThis(char* operationType, int this1_lowerRow, int this1_upperRow, int this1_lowerCol, int this1_upperCol, int this2_lowerRow, int this2_upperRow, int this2_lowerCol, int this2_upperCol, bool inPlace) {

	printf("Operation Type: %s\n", operationType);

	// add "other" to the values of this object stored in host and device memory
	// assume they're both float32s for now

	// use the lower/upper terminology since the upper values are not actually included (stops at previous index)
	int section_numRows = (this_upperRow-this_lowerRow);
	int section_numCols = (this_upperCol-this_lowerCol);

	int this_requestedSize = (section_numRows)*(section_numCols);

	printf("This size: (%i,%i,%i,%i)->(%i,%i)=%i -- other size: %i\n", this_lowerRow, this_upperRow, this_lowerCol,this_upperCol, section_numRows, section_numCols, this_requestedSize, otherLength);

	if (this_requestedSize != otherLength)
	{
		printf("Different sized arrays trying to be added together! %i vs %i\n", this_requestedSize, otherLength);
		return; // need to come up with a better way to handle invalid indexing, maybe this should actually be done on the python side
	}



	// I guess assume they've been flattened in the same way?
	// so lets first copy the new data up to the GPU, then add it

	int other_totalCols = section_numCols; // assume they're the same shape
	int other_totalRows = section_numRows;

	//float* other_device;
	//cudaError_t err = cudaMalloc((void **) &other_device, otherLength*sizeof(float));
	//err = cudaMemcpy(other_device, other, otherLength*sizeof(float), cudaMemcpyHostToDevice);


	if (inPlace == false)
	{
		err = cudaMalloc((void **) &results_device, otherLength*sizeof(float));
		results_totalRows = other_totalRows;
		results_totalColumns = other_totalCols;
		// we don't need to copy anything up right?
	}
	else
	{
		// I think the cuda kernel doesn't like being passed null pointers
		err = cudaMalloc((void **) &results_device, 2*sizeof(float));
	}


	dim3 threadsPerBlock(32,32);
	dim3 numBlocks(section_numRows/threadsPerBlock.x + 1, section_numCols/threadsPerBlock.y + 1);
	printf("Num Blocks = (%i,%i)\n", numBlocks.x, numBlocks.y);

	if (strcmp(operationType, "add") == 0)
		kernel_gpuAdd<<<numBlocks, threadsPerBlock>>>(array_device, this1_totalColumns, this_totalRows, this_totalColumns, this1_lowerRow, this1_lowerCol, this2_totalRows, this2_lowerRow, this2_lowerCol, inPlace, results_device);
	else if (strcmp(operationType, "subtract") == 0)
		kernel_gpuSubtract<<<numBlocks, threadsPerBlock>>>(array_device, this1_totalColumns, this_totalRows, this_totalColumns, this1_lowerRow, this1_lowerCol, this2_totalRows, this2_lowerRow, this2_lowerCol, inPlace, results_device);
	else if (strcmp(operationType, "multiply") == 0)
		kernel_gpuMultiply<<<numBlocks, threadsPerBlock>>>(array_device, this1_totalColumns, this_totalRows, this_totalColumns, this1_lowerRow, this1_lowerCol, this2_totalRows, this2_lowerRow, this2_lowerCol, inPlace, results_device);
	else if (strcmp(operationType, "divide") == 0)
		kernel_gpuDivide<<<numBlocks, threadsPerBlock>>>(array_device, this1_totalColumns, this_totalRows, this_totalColumns, this1_lowerRow, this1_lowerCol, this2_totalRows, this2_lowerRow, this2_lowerCol, inPlace, results_device);

	else
		throw std::invalid_argument("Unknown operation type");


	cudaCheckError();

	cudaDeviceSynchronize();
	cudaCheckError();

	if (inPlace == false)
	{
		printf("Not in place -- copying results_device back to results_host ...\n");
		results_host = (float* )malloc(otherLength*sizeof(float)); // need to malloc this before copying back
		cudaMemcpy(results_host, results_device, otherLength*sizeof(float), cudaMemcpyDeviceToHost);
		cudaCheckError();
		// this should affect the array_device_modifiedSinceLastHostCopy (leave at current value either way)
		printf("Done copying!\n");
	}
	else
		array_device_modifiedSinceLastHostCopy = true;



	cudaFree(results_device); // store in results_host for retreival and free the device array
	cudaFree(other_device);

}
*/



void gframe::concat(float* newArray, int newNumRows, int newNumCols, int axis) {

	// alright so I think another core operation is going to be concat, ie adding data
	// cuda realloc() doesn't exist, so think we need to do this one in C

	printf("Concatenating...\n");

	updateHostFromGPU(); // first make sure this is up to date

	int new_totalColumns;
	int new_totalRows;
	if (axis == 0)
	{
		// adding rows
		new_totalColumns = this_totalColumns; // enforce that the number of columns is the same?
		new_totalRows = this_totalRows + newNumRows;
	}
	else if (axis == 1)
	{
		// adding columns
		new_totalColumns = this_totalColumns + newNumCols;
		new_totalRows = this_totalRows; // enforce that the number of rows is the same?
	}
	else
	{
		throw std::out_of_range("axis must be 0 or 1");
	}


	// our new size is going to be the current size plus new size
	printf("Original Length: %i -- adding (%i,%i)=%i\n", arraySize/sizeof(float), newNumRows, newNumCols, newNumRows*newNumCols);
	size_t newSize = arraySize + newNumRows*newNumCols*sizeof(float);


	
	// I think the realloc() might be messing with the indexing of our original data, lets first copy it out and then back
	float* origDataTemp = (float* )malloc(arraySize);
	for (int n=0; n<arraySize/sizeof(float); n++)
		origDataTemp[n] = array_host[n];
	
	array_host = (float *)realloc(array_host, newSize); // and now realloc the original array
	
	// and lets just overwrite everything 
	// (yes this can be compressed more, but wrote everything out explicitly to ensure the logic makes since
	for (int n=0; n<new_totalRows; n++)
	{
		for (int m=0; m<new_totalColumns; m++)
		{
			// so here we have n and m being the final indices of our new array, we now need to figure out where that data should be pulled from
			if (axis==0 && n<this_totalRows) // adding rows and current ind is in the old data
			{
				int oldInd = n*this_totalColumns + m; // from original data
				int newInd = n*new_totalColumns + m; // to new array
				array_host[newInd] = origDataTemp[oldInd];
			}
			else if (axis == 0 && n >= this_totalRows) // adding new rows and current ind is in the new data
			{
				int oldInd = n*newNumCols + m; // relative to new data
				int newInd = n*new_totalColumns + m; // to new array
				array_host[newInd] = newArray[oldInd];
			}
			else if (axis==1 && m<this_totalColumns) // adding columns and current ind is in the old data
			{
				int oldInd = n*this_totalColumns + m; // from original data
				int newInd = n*new_totalColumns + m; // to new array
				array_host[newInd] = origDataTemp[oldInd];
			}
			else if (axis == 1 && m >= this_totalColumns) // adding new columns and current ind is in the new data
			{
				int oldInd = n*newNumCols + m; // relative to new data
				int newInd = n*new_totalColumns + m; // to ne array
				array_host[newInd] = newArray[oldInd];
			}
			
		}
	}
	
	free(origDataTemp);
	
	

	// lets update the global variables
	this_totalRows = new_totalRows;
	this_totalColumns = new_totalColumns;
	arraySize = this_totalRows*this_totalColumns*sizeof(float);
	printf("Globals updated to totalRows=%i -- totalCols=%i -- size=%i\n", this_totalRows, this_totalColumns, arraySize);


	// we now need to delete the old data off of cuda, reallocate, and copy the new stuff up
	cudaFree(array_device);

	cudaError_t err = cudaMalloc((void**) &array_device, arraySize);
	cudaCheckError();
	err = cudaMemcpy(array_device, array_host, arraySize, cudaMemcpyHostToDevice);
	cudaCheckError();

}



void gframe::updateHostFromGPU() {
	// this is expected to be an expensive operation, don't do it unless we actually need to
	// (allow the user to do a bunch of consecutive operations without copying back every time)
	printf("Array Size: %i\n", arraySize);

//	for (int n=0; n<this_totalRows*this_totalColumns; n++)
//	{
//		array_host[n] = 0; // lets try resetting this?
//		printf("Reseting index %i to zero\n", n);
//	}

	if (array_device_modifiedSinceLastHostCopy)
	{
		printf("copying device back to host ...\n");
		cudaMemcpy(array_host, array_device, arraySize, cudaMemcpyDeviceToHost);
		printf("done!\n");
		cudaCheckError();

		array_device_modifiedSinceLastHostCopy = false;
	}


}

void gframe::retreive_array(float* numpyArray) {
	// for now lets send back the entire array
	// so we already know what the length should be


	printf("C++: retreive_array of shape (%i,%i)\n\n", this_totalRows, this_totalColumns);

	updateHostFromGPU();


	//numpyArray = array_host; // will this just work?
	for (int n=0; n<this_totalRows*this_totalColumns; n++)
		numpyArray[n] = array_host[n];
}


void gframe::retreive_results_shape(int* shapeArray) {
	// since we don't necessarily know the shape of the results array when we're trying to return it, we need to make two calls here
	// first to figure out it's shape, then input a reference to a numpy array of the correct size to the other function
	printf("retreive_results_shape\n");
	shapeArray[0] = results_totalRows;
	shapeArray[1] = results_totalColumns;
}


void gframe::retreive_results(float* numpyArray) {
	printf("retreive_results\n");
	//numpyArray = array_host; // will this just work? not sure, but the below does
	for (int n=0; n<results_totalRows*results_totalColumns; n++)
		numpyArray[n] = results_host[n];

	printf("retreiving done!\n");
}





//void gframe::increment() {
////  kernel_add_one<<<64, 64>>>(array_device, length);
////  cudaError_t err = cudaGetLastError();
////  assert(err == 0);
//}
//
////void gframe::retreive() {
//////  int size = length * sizeof(int);
//////  cudaMemcpy(array_host, array_device, size, cudaMemcpyDeviceToHost);
//////  cudaError_t err = cudaGetLastError();
//////  if(err != 0) { cout << err << endl; assert(0); }
////}
//
//void gframe::retreive_to(int* array_host_, int length_) {
////  assert(length == length_);
////  int size = length * sizeof(int);
////  cudaMemcpy(array_host_, array_device, size, cudaMemcpyDeviceToHost);
////  cudaError_t err = cudaGetLastError();
////  assert(err == 0);
//}

gframe::~gframe() {
  cudaFree(array_device);
  printf("Cuda memory freed on object destruction\n");
}

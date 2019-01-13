/*
This is the central piece of code. This file implements a class
(interface in gframe.hh) that takes data in on the cpu side, copies
it to the gpu, and exposes functions (increment and retreive) that let
you perform actions with the GPU

This class will get translated into python via swig
*/

#include <gframe_kernel.cu>
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


void gframe::operateOnSection(int minRow, int maxRow, int minCol, int maxCol) {

	printf("Operating on subsection ...\n");
	kernal_printSubsection<<<1,1>>>(array_device, minRow, maxRow, minCol, maxCol, this_totalRows, this_totalColumns);
	//testKernel<<<10,32>>>();
	cudaCheckError();
	cudaDeviceSynchronize();
	printf("and we aren't crashing...\n");

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

	if (this_requestedSize != otherLength)
	{
		printf("Different sized arrays trying to be added together! %i vs %i\n", this_requestedSize, otherLength);
		return; // need to come up with a better way to handle invalid indexing, maybe this should actually be done on the python side
	}



	// I guess assume they've been flattened in the same way?
	// so lets first copy the new data up to the GPU, then add it

	int other_totalCols = section_numCols; // assume they're the same shape
	int other_totalRows = section_numRows;

	float* other_device;
	cudaError_t err = cudaMalloc((void **) &other_device, otherLength*sizeof(float));
	err = cudaMemcpy(other_device, other, otherLength*sizeof(float), cudaMemcpyHostToDevice);


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
	
	
	int* this_rowArray_device;
	int* this_colArray_device;
	cudaMalloc((void **) &this_rowArray_device, this_rowArrayLength*sizeof(int));
	cudaMalloc((void **) &this_colArray_device, this_colArrayLength*sizeof(int));
	cudaMemcpy(this_rowArray_device, this_rowArray, this_rowArrayLength*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(this_colArray_device, this_colArray, this_colArrayLength*sizeof(int), cudaMemcpyHostToDevice);
	
	for (int n=0; n<this_rowArrayLength; n++)
		printf("CPU: (%i,%i)\n", this_rowArray[n], this_colArray[n]);
	printf("Total columns: %i\n", this_totalColumns);
	


	dim3 threadsPerBlock(32,32);
	dim3 numBlocks(section_numRows/threadsPerBlock.x + 1, section_numCols/threadsPerBlock.y + 1);
	printf("Num Blocks = (%i,%i)\n", numBlocks.x, numBlocks.y);
	
	
	int operationID; // so cuda can't parse strings easily, lets convert our string ID to an integer for the kernel
	if (strcmp(operationType, "add") == 0)
		operationID = 0;
	
	

	
	kernel_gpuBasicOps<<<numBlocks, threadsPerBlock>>>(operationID, array_device, this_totalColumns, this_totalRows, this_rowArray_device, this_rowArrayLength, this_colArray_device, this_colArrayLength, other_device, other_totalCols, other_totalRows, inPlace, results_device);
	
//	if (strcmp(operationType, "add") == 0)
//		kernel_gpuAdd<<<numBlocks, threadsPerBlock>>>(array_device, this_totalColumns, this_totalRows, this_lowerRow, this_lowerCol, other_device, other_totalCols, other_totalRows, inPlace, results_device);
//	else if (strcmp(operationType, "subtract") == 0)
//		kernel_gpuSubtract<<<numBlocks, threadsPerBlock>>>(array_device, this_totalColumns, this_totalRows, this_lowerRow, this_lowerCol, other_device, other_totalCols, other_totalRows, inPlace, results_device);
//	else if (strcmp(operationType, "multiply") == 0)
//		kernel_gpuMultiply<<<numBlocks, threadsPerBlock>>>(array_device, this_totalColumns, this_totalRows, this_lowerRow, this_lowerCol, other_device, other_totalCols, other_totalRows, inPlace, results_device);
//	else if (strcmp(operationType, "divide") == 0)
//		kernel_gpuDivide<<<numBlocks, threadsPerBlock>>>(array_device, this_totalColumns, this_totalRows, this_lowerRow, this_lowerCol, other_device, other_totalCols, other_totalRows, inPlace, results_device);
//
//	else
//		throw std::invalid_argument("Unknown operation type");


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
	size_t newSize = arraySize + newNumRows*newNumCols*sizeof(float);

	// next lets try to realloc the host copy
	array_host = (float *)realloc(array_host, newSize);

	// so actually the problem with realloc is that they aren't going to be in the correct order anymore ... does it just straight up add it at the end I assume?
	// we need to go through this in two stages -- first get the original data in the new correct indices, and then fill in the new data

	printf("Moving original data...\n");
	for (int m=this_totalColumns; m>0; m--) // need to walk backwards to make sure we don't step on ourselves, also think we need to do m first?
	{
		for (int n=this_totalRows; n>0; n--)
		{
			int curInd = n*this_totalColumns + m; // current index
			int newInd = n*new_totalColumns + m; // new index we're moving to

			array_host[curInd] = array_host[newInd];
		}
	}

	printf("Adding new data ...\n");
	// so hopefully that's good ... now go through the new values
	for (int n=0; n<newNumRows; n++) // shouldn't matter what order we iterate through
	{
		for (int m=0; m<newNumCols; m++)
		{
			int relativeInd = n*newNumCols + m;

			// so these n and m are relative to the new matrix, need to convert them to absolute depending on the concatenation axis
			int absInd;
			if (axis==0)
				absInd = (n+this_totalRows)*new_totalColumns + m; // column will still be the same, just need to offset the row
			else if (axis==1)
				absInd = n*new_totalColumns + (m+this_totalColumns); // row will still be the same, just need to offset the column

			array_host[absInd] = newArray[relativeInd];
		}
	}

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


	printf("retreive_array\n");

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

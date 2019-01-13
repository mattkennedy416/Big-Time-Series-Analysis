#include <stdio.h>

//void __global__ kernel_add_one(int* a, int length) {
//    int gid = threadIdx.x + blockDim.x*blockIdx.x;
//
//    while(gid < length) {
//    	a[gid] += 1;
//        gid += blockDim.x*gridDim.x;
//    }
//}


void __global__ testKernel()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	printf("can we print from a kernel? -- %i\n", i);
}


void __global__ testKernel_doubleValues(float* array_device, int numElements, float* result) {
	
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (i < numElements)
	{
		result[i] = 2*array_device[i];	
	}
	
	
}




void __global__ kernal_printSubsection(float* array_device, int minRow, int maxRow, int minCol, int maxCol, int numRows, int numColumns){
	
	printf("Accessing from kernel:\n");
	for (int n=minRow; n<maxRow; n++)
	{
		for (int m=minCol; m<maxCol; m++)
		{
			int flattenedInd = n*numColumns + m;
			printf("(%i,%i)->%i  %f\n", n,m,flattenedInd, array_device[flattenedInd]);
		}
		printf("\n");
		
	}
	
}


void __global__ kernel_gpuBasicOps(int operationID, float* array_device, int this_totalColumns, int this_totalRows, int* this_rowArray, int this_rowArrayLength, int* this_colArray, int this_colArrayLength, float* other_device, int other_totalCols, int other_totalRows, bool inPlace, float* results_device) {
	//	// we're assuming that the other_totalCols and other_totalRows is the entire matrix
	//	// while "this" that's already in the gpu is a relative index
	
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (n < other_totalRows && m < other_totalCols)
	{
		int this_flattenedInd = this_rowArray[n]*this_totalColumns + this_colArray[m];
		int other_flattenedInd = n*other_totalCols + m;
		
		printf("accessing (%i,%i)<->(%i,%i)\n", this_rowArray[n], this_colArray[m], n, m);
		
		if (operationID == 0) // addition
		{
			if (inPlace)
				array_device[this_flattenedInd] = array_device[this_flattenedInd] + other_device[other_flattenedInd];
			else
				results_device[other_flattenedInd] = array_device[this_flattenedInd] + other_device[other_flattenedInd];
		}
	}
	
	
}

//void __global__ kernel_gpuAdd(float* array_device, int this_totalCols, int this_totalRows, int this_lowerRow, int this_lowerCol, float* other, int other_totalCols, int other_totalRows, bool inPlace, float* results_device) {
//	
//	// we're assuming that the other_totalCols and other_totalRows is the entire matrix
//	// while "this" that's already in the gpu is a relative index
//	
//	int n = blockIdx.x * blockDim.x + threadIdx.x;
//	int m = blockIdx.y * blockDim.y + threadIdx.y;
//	
//	if (n < other_totalRows && m < other_totalCols)
//	{
//		int this_flattenedInd = (n + this_lowerRow)*this_totalCols + (m + this_lowerCol);
//		int other_flattenedInd = n*other_totalCols + m;
//		
//		if (inPlace)
//			array_device[this_flattenedInd] = array_device[this_flattenedInd] + other[other_flattenedInd];
//		else
//			results_device[other_flattenedInd] = array_device[this_flattenedInd] + other[other_flattenedInd];
//		
//	}
//	
//}




void __global__ kernel_gpuSubtract(float* array_device, int this_totalCols, int this_totalRows, int this_lowerRow, int this_lowerCol, float* other, int other_totalCols, int other_totalRows, bool inPlace, float* results_device) {
	
	// we're assuming that the other_totalCols and other_totalRows is the entire matrix
	// while "this" that's already in the gpu is a relative index
	
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (n < other_totalRows && m < other_totalCols)
	{
		int this_flattenedInd = (n + this_lowerRow)*this_totalCols + (m + this_lowerCol);
		int other_flattenedInd = n*other_totalCols + m;
		
		if (inPlace)
			array_device[this_flattenedInd] = array_device[this_flattenedInd] - other[other_flattenedInd];
		else
			results_device[other_flattenedInd] = array_device[this_flattenedInd] - other[other_flattenedInd];
		
	}
	
}


void __global__ kernel_gpuMultiply(float* array_device, int this_totalCols, int this_totalRows, int this_lowerRow, int this_lowerCol, float* other, int other_totalCols, int other_totalRows, bool inPlace, float* results_device) {
	
	// we're assuming that the other_totalCols and other_totalRows is the entire matrix
	// while "this" that's already in the gpu is a relative index
	
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (n < other_totalRows && m < other_totalCols)
	{
		int this_flattenedInd = (n + this_lowerRow)*this_totalCols + (m + this_lowerCol);
		int other_flattenedInd = n*other_totalCols + m;
		
		if (inPlace)
			array_device[this_flattenedInd] = array_device[this_flattenedInd] * other[other_flattenedInd];
		else
			results_device[other_flattenedInd] = array_device[this_flattenedInd] * other[other_flattenedInd];
		
	}
	
}


void __global__ kernel_gpuDivide(float* array_device, int this_totalCols, int this_totalRows, int this_lowerRow, int this_lowerCol, float* other, int other_totalCols, int other_totalRows, bool inPlace, float* results_device) {
	
	// we're assuming that the other_totalCols and other_totalRows is the entire matrix
	// while "this" that's already in the gpu is a relative index
	
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (n < other_totalRows && m < other_totalCols)
	{
		int this_flattenedInd = (n + this_lowerRow)*this_totalCols + (m + this_lowerCol);
		int other_flattenedInd = n*other_totalCols + m;
		
		if (inPlace)
			array_device[this_flattenedInd] = array_device[this_flattenedInd] / other[other_flattenedInd];
		else
			results_device[other_flattenedInd] = array_device[this_flattenedInd] / other[other_flattenedInd];
		
	}
	
}





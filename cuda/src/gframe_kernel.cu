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


void __global__ kernel_gpuBasicOps(int operationID, float* array_device, int this_totalColumns, int this_totalRows, int* this_rowArray, int this_rowArrayLength, int* this_colArray, int this_colArrayLength, float* other_device, int section_numCols, int section_numRows, int otherLength, bool inPlace, float* results_device) {
	//	// we're assuming that the other_totalCols and other_totalRows is the entire matrix
	//	// while "this" that's already in the gpu is a relative index
	
	// so I think we'll have two different modes of operation here
	// 1st mode:
	//		other_device has same number of elements as whatever we're comparing it to in array_device
	// 2nd mode:
	// 		other_device has a single element, in which case we should compare all values in array_device to that single value (ie array_device[:,0] + 5)
	
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (n < this_rowArrayLength && m < this_colArrayLength)
	{
		printf("Kernel Operation on (%i,%i) -- otherLength=%i\n", n,m, otherLength);
		
		int this_flattenedInd;
		int other_flattenedInd;
		int results_flattenedInd;
		
		
		if (otherLength == 1)
		{
			this_flattenedInd = this_rowArray[n]*this_totalColumns + this_colArray[m];
			other_flattenedInd = 0;
			results_flattenedInd = n*section_numCols + m;
		}
		else
		{
			this_flattenedInd = this_rowArray[n]*this_totalColumns + this_colArray[m];
			other_flattenedInd = n*section_numCols + m;
			results_flattenedInd = n*section_numCols + m;
		}
		
		

		
		//printf("accessing (%i,%i)<->(%i,%i)\n", this_rowArray[n], this_colArray[m], n, m);
		
		// while I don't really like this solution, it does keep us from having to create separate kernels for every single operation
		if (operationID == 0) // **ADDITION**
		{
			printf("Adding %f to %f at location %i\n", other_device[other_flattenedInd], array_device[this_flattenedInd], this_flattenedInd);
			if (inPlace)
				array_device[this_flattenedInd] = array_device[this_flattenedInd] + other_device[other_flattenedInd];
			else
				results_device[results_flattenedInd] = array_device[this_flattenedInd] + other_device[other_flattenedInd];
		}
		
		else if (operationID == 1) // **SUBTRACTION**
		{
			if (inPlace)
				array_device[this_flattenedInd] = array_device[this_flattenedInd] - other_device[other_flattenedInd];
			else
				results_device[results_flattenedInd] = array_device[this_flattenedInd] - other_device[other_flattenedInd];
		}
		
		else if (operationID == 2) // **DIVISION**
		{
			if (inPlace)
				array_device[this_flattenedInd] = array_device[this_flattenedInd] / other_device[other_flattenedInd];
			else
				results_device[results_flattenedInd] = array_device[this_flattenedInd] / other_device[other_flattenedInd];
		}
		
		else if (operationID == 3) // **MULTIPLICATION**
		{
			if (inPlace)
				array_device[this_flattenedInd] = array_device[this_flattenedInd] * other_device[other_flattenedInd];
			else
				results_device[results_flattenedInd] = array_device[this_flattenedInd] * other_device[other_flattenedInd];
		}
		
		else if (operationID == 4) // GREATER THAN
		{
			if (inPlace)
				printf("GREATER THAN operation is only valid for not-in-place");
			else
			{
				bool val = array_device[this_flattenedInd] > other_device[other_flattenedInd];
				if (val)
					results_device[results_flattenedInd] = 1;
				else
					results_device[results_flattenedInd] = 0;
			}
				
		}

	}
}

























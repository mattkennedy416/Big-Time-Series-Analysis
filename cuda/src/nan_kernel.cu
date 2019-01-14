#include <stdio.h>
#include <float.h>

void __global__ kernel_isnan(float* array_device, int* rowArray, int rowArrayLength, int* colArray, int colArrayLength, int totalCols, int totalRows, float* results)
{
	
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (n < rowArrayLength && m < colArrayLength)
	{
		int arrayInd = n*totalCols + m;
		int resultsInd = n*colArrayLength + m;
		
		if (isnan(array_device[arrayInd]) || isinf(array_device[arrayInd])) // I think this is all we need to do
			results[resultsInd] = 1;
		else
			results[resultsInd] = 0;	
		
	}
}



void __global__ kernel_nan2num(float* array_device, int* rowArray, int rowArrayLength, int* colArray, int colArrayLength, int totalCols, int totalRows, bool inPlace, float* results)
{
	
	
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (n < rowArrayLength && m < colArrayLength)
	{
		
		
		int arrayInd = n*totalCols + m;
		int resultsInd = n*colArrayLength + m;
		
		
		if (inPlace)
		{
			if (isnan(array_device[arrayInd])) // I think this is all we need to do
				array_device[arrayInd] = 0;
			else if (isinf(array_device[arrayInd]))
				// so seems that isinf checks for both negative and positive infinity, and then we can see if the value is above or below zero?
				if (array_device[arrayInd] > 0)
					array_device[arrayInd] = FLT_MAX;
				else
					array_device[arrayInd] = -FLT_MAX;

		}
		else
		{
			if (isnan(array_device[arrayInd])) // I think this is all we need to do
				results[resultsInd] = 0;
			else if (isinf(array_device[arrayInd]))
				if (array_device[arrayInd] > 0)
					results[resultsInd] = FLT_MAX;
				else
					results[resultsInd] = -FLT_MAX; // note that FLT_MIN is the smallest float, ie E-38, NOT -E38
			else
				results[resultsInd] = array_device[arrayInd]; // otherwise just copy the value
		}
	
	}
	
	
	__syncthreads();
	
}




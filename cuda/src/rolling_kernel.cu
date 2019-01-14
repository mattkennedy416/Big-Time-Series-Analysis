#include <stdio.h>



void __global__ kernel_meanSMA(float* array_device, int methodID, int width, int* rowArray, int rowArrayLength, int* colArray, int colArrayLength, int totalCols, int totalRows, float* results) {
	
	// so doing it this way would probably have an advantage for calculating multiple columns at once, but a disadvantage for single column operations?
	// honestly I doubt it will be remotely noticable, but we could switch it to be 1D in a for loop or have both depending on how many columns are input?
	
	// as it's being calculated for every point, this will probably be the best way to do it even for very large window sizes, because in the vast majority of cases there will be more windows than available threads
	
	// also assume it's always in the row direction (in time)
	
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int m = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (n < rowArrayLength && m < colArrayLength)
	{
		// alright so this kernel thread is going to calculate the values for a single point
		// where n is the row and m is the column, want to iterate up and down n
		
		int start;
		int end;
		if (methodID == 0) // backward windowing
		{
			start = n - width + 1;
			end = n + 1; // to make sure we include the last point and still get the same number of indices
		}
		else if (methodID == 1) // center windowing
		{
			start = n - width/2;
			end = n + width/2; // if window is odd, I think integer division will round them both down?
		}
		
		// at the edges I guess just keep decreasing the window size?
		// other option is to make them NaNs ... and while making them NaNs is more technically correct, just decreasing window size at edges is probably more useful
		if (start < 0)
			start = 0;
		if (end > totalRows)
			end = totalRows;
		
		
		float total = 0;
		for (int nn=start; nn<end; nn++)
		{
			// now taking this to be our new row, need to get absolute/flattened index
			int arrayInd = nn*totalCols + m;
			total += array_device[arrayInd];
		}
		
		int resultsInd = n*colArrayLength + m;
		results[resultsInd] = total / (end - start);
	
	}
	
}



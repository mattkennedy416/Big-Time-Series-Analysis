#include <stdio.h>
#include <math.h>

void __global__ kernel_linearInterpolation(float* array_device, int* rowArray, int rowArrayLength, int col, int colArrayLength, int totalCols, int totalRows, float* originalTimes, int originalTimesLength, float* newTimes, int newTimesLength, float* results_device)
{
	
	// where original times and new times should probably be unix time? but technically won't matter probably
	
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	//int m = blockIdx.y * blockDim.y + threadIdx.y;
	
	int m = col;
	
	if (n < newTimesLength && m < colArrayLength)
	{
		// so we created a row-wise thread per element in the newTimes, with m being number of columns
		
		float newTime = newTimes[n];
		int resultsInd = n*colArrayLength + m;
		
		
		// need to potentially handle the first and last points in a special way ,,,
		if (n==0 && newTimes[0] < originalTimes[0])
		{
			// lets say if newTimes[0] is prior to originalTimes[0] less than the distance from originalTimes[1] to originalTimes[2], we'll accept it
			// otherwise call it a NaN
			if ( originalTimes[0] - newTimes[0] < originalTimes[1] - originalTimes[0] )
			{
				// continue slope from points 0 to 1?
				// ie assume there's a previous point at same slope and same sample rate
				int arrayInd1 = (0)*totalCols + m;
				int arrayInd2 = (1)*totalCols + m;
				float valueChange = array_device[arrayInd2] - array_device[arrayInd1];
				float timeDiff = originalTimes[1] - originalTimes[0];
				
				float pseudoPointTime = originalTimes[0] - timeDiff;
				float pseudoPointValue = array_device[arrayInd1] - valueChange;
				
				float perc = (newTime - pseudoPointTime) / timeDiff;
				
				results_device[resultsInd] = pseudoPointValue + perc*valueChange;
				
			}
			else
				results_device[resultsInd] = NAN;
			
			return;			
		}
		
		// ================================================
		// ================================================
		// ===== DO THIS SAME LOGIC FOR LAST POINT? =======
		// ================================================
		// ================================================
		
		// we now need to find the points before and after ...
		// do we need to iterate through the entire originalTimes vector or is there some way we can approximate the starting location?
		
		//float origSampleRate = (originalTimes[originalTimesLength-1] - originalTimes[0]) / originalTimesLength;
		//float newSampleRate = (newTimes[newTimesLength-1] - newTimes[0]) / newTimesLength;
		
		//int originalTimeLoc = (newTime - newTimes[0]) / (originalTimes[originalTimesLength-1] - originalTimes)
			
		float newTimePerc = (newTime - newTimes[0]) / (newTimes[newTimesLength-1] - newTimes[0]);
		int estimation = newTimePerc * originalTimesLength; // start at approximately the correct point
		
		int iterationsToFindMatch = 0;
		
		int originalTimeLoc;
		if (originalTimes[estimation] > newTime)
		{
			// time we guessed is greater than the time we want ... so walk backwards?
			for (originalTimeLoc=estimation; originalTimeLoc>0; originalTimeLoc--)
			{
				if (originalTimes[originalTimeLoc] < newTime)
					break;
				iterationsToFindMatch++;
			}
		}
		else
		{
			// the time we guessed is greater than the time we want ... so walk forwards?
			for (originalTimeLoc=estimation; originalTimeLoc<originalTimesLength; originalTimeLoc++)
			{
				if (originalTimes[originalTimeLoc] > newTime)
				{
					originalTimeLoc--; // want the left side point
					break;
				}
				iterationsToFindMatch++;
			}
		}
		
		printf("Loc %i needed %i iterations, starting from %i -> %i\n", n, iterationsToFindMatch, estimation, originalTimeLoc);
		
		
		int arrayInd1 = originalTimeLoc*totalCols + m;
		int arrayInd2 = (originalTimeLoc+1)*totalCols + m;				
		
		float valueChange = (array_device[arrayInd2] - array_device[arrayInd1]);
		
		// we need to figure out what percentage the newTime is between the two points
		float origTime1 = originalTimes[originalTimeLoc];
		float origTime2 = originalTimes[originalTimeLoc+1];
		
		float perc = (newTime - origTime1) / (origTime2 - origTime1);
		
		results_device[resultsInd] = array_device[arrayInd1] + perc*valueChange;
		
	
	}
	
}



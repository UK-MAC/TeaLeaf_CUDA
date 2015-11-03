#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include "ext_shared_cuda.hpp"
#include "ext_cuda_chunk.hpp"

// Plots a three-dimensional dat file.
void TeaLeafCudaChunk::Plot3d(double* dBuffer, std::string name)
{
	// Open the plot file
	FILE* fp = fopen("plot2d.dat", "wb");
	if(!fp) { fprintf(stderr, "Could not open plot file.\n"); }

	int size = xCells*yCells*sizeof(double);
	double* hBuffer = (double*)malloc(size);
	if(hBuffer == NULL)
	{
		fprintf(stderr, "Could not allocate hBuffer.\n");
		exit(1);
	}

	cudaDeviceSynchronize();
	cudaMemcpy(hBuffer, dBuffer, size, cudaMemcpyDeviceToHost);
	CheckErrors(__LINE__, __FILE__);

	double bSum = 0.0;

	// Plot the data structure
	for(int jj = 0; jj < yCells; ++jj)
	{
		for(int kk = 0; kk < xCells; ++kk)
		{
			double val = hBuffer[kk+jj*xCells];
			fprintf(fp, "%d %d %.12E\n", kk, jj, val);
			bSum+=val;
		}
	}

	printf("%s: %.12E\n", name.c_str(), bSum);

	free(hBuffer);
	fclose(fp);
	cudaDeviceSynchronize();
}

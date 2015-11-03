#include <cstdio>
#include "ext_cuda_chunk.hpp"
#include "kernels/solver_methods.cuknl"

/*
 *		SHARED SOLVER METHODS
 */

// Entry point to copy U.
extern "C"
void ext_solver_copy_u_(
		const int* chunk)
{
	Chunks[*chunk-1]->CopyU();
}

// Entry point for calculating residual.
extern "C"
void ext_calculate_residual_(
		const int* chunk)
{
	Chunks[*chunk-1]->CalculateResidual();
}

// Entry point for calculating 2norm.
extern "C"
void ext_calculate_2norm_(
		const int* chunk,
		const int* normArray,
	   	double* normOut)
{
	Chunks[*chunk-1]->Calculate2Norm(*normArray, normOut);
}

// Entry point for finalising solution.
extern "C"
void ext_solver_finalise_(
		const int* chunk)
{
	Chunks[*chunk-1]->Finalise();
}

// Determines the rx, ry values.
void TeaLeafCudaChunk::CalcRxRy(
		const double dt,
		double* rxOut,
		double* ryOut)
{
	double dx, dy;

	cudaMemcpy(&dx, dCellDx, sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&dy, dCellDy, sizeof(double), cudaMemcpyDeviceToHost);
	TeaLeafCudaChunk::CheckErrors(__LINE__,__FILE__);

	*rxOut = dt/(dx*dx);
	*ryOut = dt/(dy*dy);
}

// Copies the value of u
void TeaLeafCudaChunk::CopyU()
{
	PRE_KERNEL(2*HALO_PAD);

	CuKnlCopyU<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, xCells,  dU, dU0);

	POST_KERNEL("Copy U");
}

// Calculates the current residual value.
void TeaLeafCudaChunk::CalculateResidual()
{
	PRE_KERNEL(2*HALO_PAD);

	CuKnlCalculateResidual<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, xCells, 
			dU, dU0, dKx, dKy, dR);

	POST_KERNEL("Calc Residual");
}

// Calculates the 2norm of an array
void TeaLeafCudaChunk::Calculate2Norm(
		const bool normArray,
		double* normOut)
{
	PRE_KERNEL(2*HALO_PAD);

	CuKnlCalculate2Norm<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, xCells,
			normArray ? dR : dU0, dReduceBuffer1);

	SumReduce(dReduceBuffer1, normOut, numBlocks);
	POST_KERNEL("Calc 2 Norm");
}

// Reduces residual values
void TeaLeafCudaChunk::SumReduce(
		double* buffer,
		double* result,
		int len)
{
	while(len > 1)
	{
		int numBlocks = std::ceil(len/(float)BLOCK_SIZE);
		CuKnlSumReduce<<<numBlocks,BLOCK_SIZE>>>(len, buffer);
		len = numBlocks;
	}

	cudaMemcpy(result, buffer, sizeof(double), cudaMemcpyDeviceToHost);
	CheckErrors(__LINE__,__FILE__);
}

// Finalises the solution.
void TeaLeafCudaChunk::Finalise()
{
	PRE_KERNEL(2*HALO_PAD);

	CuKnlFinalise<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, xCells, dDensity, dU, dEnergy1);

	POST_KERNEL("Finalise");
}

// Loads alphas and betas into device memory
void TeaLeafCudaChunk::LoadAlphaBeta(
		const double* alphas,
		const double* betas,
		const int numCoefs)
{
	size_t length = numCoefs*sizeof(double);
	cudaMalloc((void**) &dAlphas, length);
	cudaMalloc((void**) &dBetas, length);
	cudaMemcpy(dAlphas, alphas, length, cudaMemcpyHostToDevice);
	cudaMemcpy(dBetas, betas, length, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	CheckErrors(__LINE__,__FILE__);
}

#include <cstdio>
#include <math.h>
#include "ext_cuda_chunk.hpp"
#include "kernels/jacobi_solve.cuknl"

/*
 *		JACOBI SOLVER KERNEL
 */

// Entry point for Jacobi initialisation.
extern "C"
void ext_jacobi_kernel_init_(
		const int* chunk,
		const int* coefficient,
		const double* dt,
		double* rx,
		double* ry)
{
	Chunks[*chunk-1]->JacobiInit(*dt, rx, ry, *coefficient);
}

// Entry point for Jacobi solver main method.
extern "C"
void ext_jacobi_kernel_solve_(
		const int* chunk,
		double* error)
{
	Chunks[*chunk-1]->JacobiSolve(error);
}

// Jacobi solver initialisation method.
void TeaLeafCudaChunk::JacobiInit(
		const double dt,
		double* rx,
		double* ry,
		const int coefficient)
{
	if(coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
	{
		Abort(__LINE__, __FILE__, "Coefficient %d is not valid.\n", coefficient);
	}

	CalcRxRy(dt, rx, ry);

	PRE_KERNEL(HALO_PAD);

	CuKnlJacobiInit<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, xCells, 
			dDensity, dEnergy1, *rx, *ry, dKx, dKy, 
			dU0, dU, coefficient);

	POST_KERNEL("Jacobi Initialise");
}

void TeaLeafCudaChunk::JacobiCopyU()
{
	PRE_KERNEL(0);

	CuKnlJacobiCopyU<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, dU, dR);

	POST_KERNEL("Jacobi Copy U");
}

// Main Jacobi solver method.
void TeaLeafCudaChunk::JacobiSolve(
		double* error)
{
	JacobiCopyU();

	PRE_KERNEL(2*HALO_PAD);

	CuKnlJacobiSolve<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, xCells, dKx, dKy, dU0, dR, dU, dReduceBuffer1);

	SumReduce(dReduceBuffer1, error, numBlocks);

	POST_KERNEL("Jacobi Solve");
}


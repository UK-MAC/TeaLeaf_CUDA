#include <cstdio>
#include <math.h>
#include "ext_cuda_chunk.hpp"
#include "kernels/cheby_solve.cuknl"

/*
 *		CHEBYSHEV SOLVER KERNEL
 */

// Entry point for Chebyshev initialisation.
extern "C"
void ext_cheby_solver_init_(
		const int* chunk,
		const double* alphas, 
		const double* betas,
		int* numCoefs,
		const double* theta,
		const int* preconditioner)
{
	Chunks[*chunk-1]->ChebyInit(
			alphas, betas, *numCoefs, *theta, *preconditioner);
}

// Entry point for the main Chebyshev iteration
extern "C"
void ext_cheby_solver_iterate_(
		const int* chunk,
		const int* chebyCalcStep)
{
	Chunks[*chunk-1]->ChebyIterate(*chebyCalcStep);
}

// Initialises the Chebyshev solver
void TeaLeafCudaChunk::ChebyInit(
		const double* alphas, 
		const double* betas,
		int numCoefs,
		const double theta,
		const bool preconditionerOn)
{
	preconditioner = preconditionerOn;

	LoadAlphaBeta(alphas, betas, numCoefs);

	PRE_KERNEL(2*HALO_PAD);

	CuKnlChebyInitP<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, xCells, dU, dU0,
			dMi, dKx, dKy, theta, preconditioner, dP, dR, dW);

	POST_KERNEL("Cheby Init P");

	START_PROFILING();

	CuKnlChebyCalcU<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, xCells, dP, dU);

	POST_KERNEL("Cheby Calc U");
}

// The main Chebyshev iteration
void TeaLeafCudaChunk::ChebyIterate(
		const int chebyCalcStep)
{
	PRE_KERNEL(2*HALO_PAD);

	CuKnlChebyCalcP<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, xCells, dU, dU0,
			dMi, dKx, dKy, dAlphas, dBetas, 
			preconditioner, chebyCalcStep-1, dP, dR, dW);

	POST_KERNEL("Cheby Iterate");

	START_PROFILING();

	CuKnlChebyCalcU<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, xCells, dP, dU);

	POST_KERNEL("Cheby Calc U");
}

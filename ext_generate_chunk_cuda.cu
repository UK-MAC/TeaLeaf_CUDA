#include <stdio.h>
#include "ext_cuda_chunk.hpp"
#include "kernels/generate_chunk.cuknl"

/*
 *		GENERATE CHUNK KERNEL
 */

// Entry point for the the chunk generation method.
extern "C"
void ext_generate_chunk_(
		const int* chunk,
		const int* numberOfStates,
		const double* stateDensity,
		const double* stateEnergy,
		const double* stateXMin,
		const double* stateXMax,
		const double* stateYMin,
		const double* stateYMax,
		const double* stateRadius,
		const int* stateGeometry,
		const int* rectParam,
		const int* circParam,
		const int* pointParam)
{
	Chunks[*chunk-1]->GenerateChunk(*numberOfStates, stateDensity,
			stateEnergy, stateXMin, stateXMax, stateYMin, stateYMax,
			stateRadius, stateGeometry, *rectParam, *circParam, *pointParam);
}

// Sets up a chunk, setting state provided.
void TeaLeafCudaChunk::GenerateChunk(
		const int numberOfStates,
		const double* stateDensity,
		const double* stateEnergy,
		const double* stateXMin,
		const double* stateXMax,
		const double* stateYMin,
		const double* stateYMax,
		const double* stateRadius,
		const int* stateGeometry,
		const int rectParam,
		const int circParam,
		const int pointParam)
{
	PRE_KERNEL(0);

	CuKnlGenerateInitial<<<numBlocks, BLOCK_SIZE>>>(
			numThreads, stateEnergy[0], stateDensity[0], dEnergy0, dDensity);

	POST_KERNEL("Generate Chunk Initial");

	for(int state=1; state < numberOfStates; ++state)
	{
		START_PROFILING();

		CuKnlGenerateChunk<<<numBlocks, BLOCK_SIZE>>>(
				xCells, yCells, state, rectParam, circParam, pointParam, 
				dU, dEnergy0, dDensity, stateEnergy[state], stateDensity[state], 
				stateGeometry[state], stateRadius[state], stateXMin[state], 
				stateYMin[state], stateXMax[state], stateYMax[state], 
				dVertexX, dVertexY, dCellX, dCellY);

		POST_KERNEL("Generate Chunk Final");
	}
}

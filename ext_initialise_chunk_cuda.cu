#include <cstdio>
#include <algorithm>
#include "ext_cuda_chunk.hpp"
#include "kernels/initialise_chunk.cuknl"

/*
 * 		INITIALISE CHUNK KERNEL
 */

// Extended CUDA kernel for the chunk initialisation
extern "C"
void ext_initialise_chunk_( 
		const int* chunk,
		const double* xMin,
		const double* yMin,
		const double* dx,
		const double* dy)
{
	Chunks[*chunk-1]->InitialiseChunk(*xMin, *yMin, *dx, *dy);
}

// Initialises the chunk's primary data fields.
void TeaLeafCudaChunk::InitialiseChunk( 
		const double xMin,
		const double yMin,
		const double dx,
		const double dy)
{
	int numCells = 1+std::max(xCells, yCells);
	int numBlocks = std::ceil((float)numCells/(float)BLOCK_SIZE);

	START_PROFILING();

	CuKnlInitialiseChunkVertices<<<numBlocks, BLOCK_SIZE>>>(
			xCells, yCells,  xMin, yMin, dx, dy,
			dVertexX, dVertexY, dVertexDx, dVertexDy);

	POST_KERNEL("Initialise Chunk Vertices");

	numCells = (xCells+1)*(yCells+1);
	numBlocks = std::ceil((float)numCells/(float)BLOCK_SIZE);

	START_PROFILING();

	CuKnlInitialiseChunk<<<numBlocks, BLOCK_SIZE>>>(
			xCells, yCells, dx, dy, dVertexX, dVertexY, 
			dCellX, dCellY, dCellDx, dCellDy, dVolume, dXArea, dYArea);

	POST_KERNEL("Initialise Chunk Final");
}

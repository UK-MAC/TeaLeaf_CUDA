#include <cstdio>
#include <iostream>
#include "ext_cuda_chunk.hpp"
#include "kernels/update_halo.cuknl"

/*
 * 		UPDATE HALO KERNEL
 */	

using std::ceil;

// Entry point for updating halos
extern "C"
void ext_update_halo_kernel_(
		const int* chunk,
		const int* chunkNeighbours,
		const int* fields,
		const int* depth)
{
	Chunks[*chunk-1]->UpdateHalo(chunkNeighbours, fields, *depth);
}

// Performs halo updates for all fields
void TeaLeafCudaChunk::UpdateHalo(
		const int* chunkNeighbours,
		const int* fields,
		const int depth)
{
#define LAUNCH_UPDATE(index, buffer, depth)\
	if(fields[index-1])\
	{\
		UpdateFace(chunkNeighbours, depth, buffer);\
	}

	LAUNCH_UPDATE(FIELD_P, dP, depth);
	LAUNCH_UPDATE(FIELD_DENSITY, dDensity, depth);
	LAUNCH_UPDATE(FIELD_ENERGY0, dEnergy0, depth);
	LAUNCH_UPDATE(FIELD_ENERGY1, dEnergy1, depth);
	LAUNCH_UPDATE(FIELD_U, dU, depth);
	LAUNCH_UPDATE(FIELD_SD, dSd, depth);
}

// Invokes a halo update for a particular field on all requested faces
void TeaLeafCudaChunk::UpdateFace(
		const int* chunkNeighbours,
		const int depth,
		double* buffer)
{
#define UPDATE_FACE(face, kernelName, updateKernel) \
	if(chunkNeighbours[face-1] == EXTERNAL_FACE)\
	{\
		START_PROFILING();\
		updateKernel<<<numBlocks, BLOCK_SIZE>>>(\
				xCells, yCells, depth, buffer);\
		POST_KERNEL(kernelName);\
	}

	int numBlocks = ceil((xCells*depth)/(float)BLOCK_SIZE);
	UPDATE_FACE(CHUNK_TOP, "Halo Top", CuKnlUpdateTop);
	UPDATE_FACE(CHUNK_BOTTOM, "Halo Bottom", CuKnlUpdateBottom);

	numBlocks = ceil((yCells*depth)/(float)BLOCK_SIZE);
	UPDATE_FACE(CHUNK_RIGHT, "Halo Right", CuKnlUpdateRight);
	UPDATE_FACE(CHUNK_LEFT, "Halo Left", CuKnlUpdateLeft);
}

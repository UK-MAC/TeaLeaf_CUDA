#include <stdio.h>
#include "ext_cuda_chunk.hpp"
#include "kernels/set_field.cuknl"

/*
 * 		SET FIELD KERNEL
 * 		Sets energy1 to energy0.
 */	

// Entry point for the the set field method.
extern "C"
void ext_set_field_kernel_(const int* chunk)
{
	Chunks[*chunk-1]->SetField();
}

// Copies energy0 into energy1.
void TeaLeafCudaChunk::SetField()
{
	PRE_KERNEL(0);

	CuKnlSetField<<<numBlocks, BLOCK_SIZE>>>(
			innerX, innerY, dEnergy0, dEnergy1);

	POST_KERNEL("Set Field");
}


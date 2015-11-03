#include <stdio.h>
#include "ext_cuda_chunk.hpp"
#include "kernels/field_summary.cuknl"

/*
 * 		FIELD SUMMARY KERNEL
 * 		Calculates aggregates of values in field.
 */	

// Entry point for field summary method.
extern "C"
void ext_field_summary_kernel_(
		const int* chunk,
		double* volOut,
		double* massOut,
		double* ieOut,
		double* tempOut)
{
	Chunks[*chunk-1]->FieldSummary(volOut, massOut, ieOut, tempOut);
}

// Calculates key values from the current field.
void TeaLeafCudaChunk::FieldSummary(
		double* volOut,
		double* massOut,
		double* ieOut,
		double* tempOut)
{
	PRE_KERNEL(2*HALO_PAD);

	CuKnlFieldSummary<<<numBlocks,BLOCK_SIZE>>>(
			xCells, yCells, innerX, innerY, dVolume, dDensity, dEnergy0,
			dU, dReduceBuffer1, dReduceBuffer2, dReduceBuffer3, dReduceBuffer4);

	SumReduce(dReduceBuffer1, volOut, numBlocks);
	SumReduce(dReduceBuffer2, massOut, numBlocks);
	SumReduce(dReduceBuffer3, ieOut, numBlocks);
	SumReduce(dReduceBuffer4, tempOut, numBlocks);

	POST_KERNEL("Field Summary");
}

#include <cstdio>
#include <math.h>
#include "ext_cuda_chunk.hpp"
#include "kernels/cg_solve.cuknl"

/*
 *      CONJUGATE GRADIENT SOLVER KERNEL
 */

// Entry point for CG initialisation.
extern "C"
void ext_cg_solver_init_(
        const int* chunk,
        const int* coefficient,
        const int* preconditioner,
        double* dt,
        double* rx,
        double* ry,
        double* rro)
{
    Chunks[*chunk-1]->CGInit(
            *coefficient, *preconditioner, *dt, rx, ry, rro);
}

// Entry point for calculating w
extern "C"
void ext_cg_calc_w_(
        const int* chunk,
        double* pw)
{
    Chunks[*chunk-1]->CGCalcW(pw);
}

// Entry point for calculating u and r
extern "C"
void ext_cg_calc_ur_(
        const int* chunk,
        const double* alpha,
        double* rrn)
{
    Chunks[*chunk-1]->CGCalcUr(*alpha, rrn);
}

// Entry point for calculating p
extern "C"
void ext_cg_calc_p_(
        const int* chunk,
        const double* beta)
{
    Chunks[*chunk-1]->CGCalcP(*beta);
}

// Initialises the CG solver
void TeaLeafCudaChunk::CGInit(
        const int coefficient,
        const bool enablePreconditioner,
        const double dt,
        double* rx,
        double* ry,
        double* rro)
{
    if(coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
    {
        Abort(__LINE__, __FILE__, "Coefficient %d is not valid.\n", coefficient);
    }

    preconditioner = enablePreconditioner;

    CalcRxRy(dt, rx, ry);
    CGInitU(coefficient);
    CGInitDirections(*rx, *ry);
    CGInitOthers(rro);
}

// Initialises u and w
void TeaLeafCudaChunk::CGInitU(
        const int coefficient)
{
    PRE_KERNEL(0);

    CuKnlCGInitU<<<numBlocks, BLOCK_SIZE>>>(
            innerX, innerY, coefficient, dDensity,
            dEnergy1, dU, dP, dR, dW);

    POST_KERNEL("CG Init U and W");
}

// Initialises the directions kx and ky
void TeaLeafCudaChunk::CGInitDirections(
        double rx, 
        double ry)
{
    PRE_KERNEL(3);

    CuKnlCGInitDirections<<<numBlocks, BLOCK_SIZE>>>(
            innerX, innerY, xCells,
            dW, dKx, dKy, rx, ry);

    POST_KERNEL("CG Init K");
}

// Initialises other cg variables
void TeaLeafCudaChunk::CGInitOthers(
        double* rro)
{
    PRE_KERNEL(2*HALO_PAD);

    CuKnlCGInitOthers<<<numBlocks, BLOCK_SIZE>>>(
            innerX, innerY, xCells, 
            dU, dKx, dKy, preconditioner, 
            dReduceBuffer1, dP, dR, dW, dMi, dZ);

    SumReduce(dReduceBuffer1, rro, numBlocks);
    POST_KERNEL("CG Calc RRO");
}

// Calculates new value for w
void TeaLeafCudaChunk::CGCalcW(
        double* pw)
{
    PRE_KERNEL(2*HALO_PAD);

    CuKnlCGCalcW<<<numBlocks, BLOCK_SIZE>>>(
            innerX, innerY, xCells,
            dKx, dKy, dP, dReduceBuffer2, dW);

    SumReduce(dReduceBuffer2, pw, numBlocks);
    POST_KERNEL("CG Calc W");
}

// Calculates new values for u and r
void TeaLeafCudaChunk::CGCalcUr(
        const double alpha,
        double* rrn)
{
    PRE_KERNEL(2*HALO_PAD);

    CuKnlCGCalcUr<<<numBlocks, BLOCK_SIZE>>>(
            innerX, innerY, xCells, preconditioner, alpha, 
            dMi, dP, dW, dU, dZ, dR, dReduceBuffer3);

    SumReduce(dReduceBuffer3, rrn, numBlocks);
    POST_KERNEL("CG Calc UR");
}

// Calculates a new value for p
void TeaLeafCudaChunk::CGCalcP(
        const double beta)
{
    PRE_KERNEL(2*HALO_PAD);

    CuKnlCGCalcP<<<numBlocks, BLOCK_SIZE>>>(
            innerX, innerY, xCells,
            preconditioner, beta, dR, dZ, dP);

    POST_KERNEL("CG Calc P");
}

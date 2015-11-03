#ifndef __CUDA_CHUNK
#define __CUDA_CHUNK

#include <vector>
#include <map>
#include "ext_shared_cuda.hpp"
#include "kernels/cudaknl_shared.hpp"
#include "thrust/device_allocator.h"
#include "thrust/extrema.h"

// The core Tealeaf-CUDA interface class.
struct TeaLeafCudaChunk
{
    TeaLeafCudaChunk(
            const int xMax, 
            const int yMax, 
            const int rank);

    ~TeaLeafCudaChunk();

    // Initialisation
    void FieldSummary(
            double* volOut,
            double* massOut,
            double* ieOut,
            double* tempOut);

    void GenerateChunk(
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
            const int pointParam);

    void InitialiseChunk(
            const double xMin,
            const double yMin,
            const double dx,
            const double dy);

    void PackUnpackKernel(
            const int* fields,
            const int* offsets,
            const int depth,
            const int face,
            double* buffer, 
            const bool pack);

    void SetField();

    void UpdateHalo(
            const int* chunkNeighbours,
            const int* fields,
            const int depth);

    void UpdateFace(
            const int* chunkNeighbours,
            const int depth,
            double* buffer);

    // Jacobi
    void JacobiSolve(
            double* error);

    void JacobiCopyU();

    void JacobiInit(
            const double dt,
            double* rx,
            double* ry,
            const int coefficient);

    // Chebyshev
    void ChebyInit(
            const double* chAlphas, 
            const double* chBetas,
            int numcoefs,
            const double theta,
            const bool preconditioner);

    void ChebyIterate(
            const int chebyCalcStep);

    // Conjugate Gradient
    void CGInit(
            const int coefficient,
            const bool preconditioner,
            const double dt,
            double* rx,
            double* ry,
            double* rro);

    void CGInitU(
            const int coefficient);

    void CGInitDirections(
            double rx,
            double ry);

    void CGInitOthers(
            double* rro);

    void CGCalcW(
            double* pw);

    void CGCalcUr(
            const double alpha,
            double* rrn);

    void CGCalcP(
            const double beta);

    // PPCG
    void PPCGInit(
            const bool preconditionerOn,
            const double* alphas,
            const double* betas,
            const int numSteps);

    void PPCGInitP(
            double* rro);

    void PPCGInitSd(
            const double theta);

    void PPCGInner(
            const int currentStep);

    // Shared solver methods
    void CalcRxRy(
            const double dt,
            double* rx,
            double* ry);

    void CalculateResidual();

    void Calculate2Norm(
            const bool normArray,
            double* norm);

    void SumReduce(
            double* buffer,
            double* result,
            int n);

    void LoadAlphaBeta(
            const double* alphas,
            const double* betas,
            const int numCoefs);

    void CopyU();

    void Finalise();

    void Plot3d(
            double* buffer, 
            std::string name);

    // Helpers
    static void Abort(
            int lineNum, 
            const char* file, 
            const char* format, 
            ...);

    static void CheckErrors(
            int lineNum, 
            const char* file);

    static const char* CudaCodes(int code);

    // Key shared
    int xCells;
    int yCells;
    int rank;
    int deviceId;
    int maxBlocks;
    bool preconditioner;

    // Profiling objects
    float span;
    std::map<std::string, double> kernelTimes;
    std::map<std::string, int> kernelCalls;
    cudaEvent_t start;
    cudaEvent_t stop;

    // Shared device pointers
    double* dCellX;
    double* dCellDx;
    double* dCellY;
    double* dCellDy;
    double* dD;
    double* dDensity;
    double* dEnergy0;
    double* dEnergy1;
    double* dKx;
    double* dKy;
    double* dMi;
    double* dP;
    double* dR;
    double* dSd;
    double* dU;
    double* dU0;
    double* dVertexX;
    double* dVertexDx;
    double* dVertexY;
    double* dVertexDy;
    double* dVolume;
    double* dW;
    double* dXArea;
    double* dYArea;
    double* dZ;

    // MPI Buffers
    double* dLeftBuffer;
    double* dRightBuffer;
    double* dTopBuffer;
    double* dBottomBuffer;

    // Chebyshev
    double* dAlphas;
    double* dBetas;

    // For reductions
    double* dReduceBuffer1;
    double* dReduceBuffer2;
    double* dReduceBuffer3;
    double* dReduceBuffer4;
};

// Globally stored list of chunks.
extern std::vector<TeaLeafCudaChunk*> Chunks;

#endif

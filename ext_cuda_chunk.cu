#include <cstdio>
#include <cstdarg>
#include "ext_cuda_chunk.hpp"

// Globally shared data structure.
std::vector<TeaLeafCudaChunk*> Chunks;

// Entry point for the initialisation of the CUDA extension
extern "C"
void ext_init_cuda_(
		int* xMax, 
		int* yMax, 
		int* rank)
{
	Chunks.push_back(new TeaLeafCudaChunk(*xMax, *yMax, *rank));
}

// Entry point for the finalisation of the CUDA extension
extern "C"
void ext_finalise_()
{
	for(int ii = 0; ii != Chunks.size(); ++ii)
	{
		delete Chunks[ii];
	}
}

TeaLeafCudaChunk::TeaLeafCudaChunk(
		int xMax, 
		int yMax, 
		int rank)
: xCells(xMax+HALO_PAD*2), 
	yCells(yMax+HALO_PAD*2),
	rank(rank)
{
	// Naive assumption that devices are paired even and odd
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	deviceId = rank%numDevices;

	int result = cudaSetDevice(deviceId);
	if(result != cudaSuccess)
	{
		Abort(__LINE__,__FILE__,"Could not allocate CUDA device %d.\n", deviceId);
	}

	struct cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, deviceId);

	printf("Rank %d using %s device id %d\n", rank, properties.name, deviceId);

	const int block = xCells*yCells;

#define CUDA_MALLOC(buf, size) 		\
	cudaMalloc((void**)&buf, size);	\
	CheckErrors(__LINE__,__FILE__);	\
	cudaDeviceSynchronize();		\
	cudaMemset(buf, 0, size);		\
	cudaDeviceSynchronize();		\
	CheckErrors(__LINE__,__FILE__);

	CUDA_MALLOC(dU, block*sizeof(double));
	CUDA_MALLOC(dU0, block*sizeof(double));
	CUDA_MALLOC(dSd, block*sizeof(double));
	CUDA_MALLOC(dR, block*sizeof(double));
	CUDA_MALLOC(dW, block*sizeof(double));
	CUDA_MALLOC(dZ, block*sizeof(double));
	CUDA_MALLOC(dP, block*sizeof(double));
	CUDA_MALLOC(dMi, block*sizeof(double));
	CUDA_MALLOC(dD, block*sizeof(double));
	CUDA_MALLOC(dKx, block*sizeof(double));
	CUDA_MALLOC(dKy, block*sizeof(double));
	CUDA_MALLOC(dDensity, block*sizeof(double));
	CUDA_MALLOC(dEnergy0, block*sizeof(double));
	CUDA_MALLOC(dEnergy1, block*sizeof(double));
	CUDA_MALLOC(dVolume, block*sizeof(double));
	CUDA_MALLOC(dCellX, xCells*sizeof(double));
	CUDA_MALLOC(dCellY, yCells*sizeof(double));
	CUDA_MALLOC(dCellDx, xCells*sizeof(double));
	CUDA_MALLOC(dCellDy, yCells*sizeof(double));
	CUDA_MALLOC(dVertexX, (xCells+1)*sizeof(double));
	CUDA_MALLOC(dVertexY, (yCells+1)*sizeof(double));
	CUDA_MALLOC(dVertexDx, (xCells+1)*sizeof(double));
	CUDA_MALLOC(dVertexDy, (yCells+1)*sizeof(double));
	CUDA_MALLOC(dXArea, (xCells+1)*yCells*sizeof(double));
	CUDA_MALLOC(dYArea, xCells*(yCells+1)*sizeof(double));
	CUDA_MALLOC(dTopBuffer, (xCells+1)*MAX_DEPTH*NUM_FIELDS*sizeof(double));
	CUDA_MALLOC(dBottomBuffer, (xCells+1)*MAX_DEPTH*NUM_FIELDS*sizeof(double));
	CUDA_MALLOC(dLeftBuffer, (yCells+1)*MAX_DEPTH*NUM_FIELDS*sizeof(double));
	CUDA_MALLOC(dRightBuffer, (yCells+1)*MAX_DEPTH*NUM_FIELDS*sizeof(double));

	const size_t reduceBufferLength = xCells*yCells*sizeof(double);
	CUDA_MALLOC(dReduceBuffer1, reduceBufferLength);
	CUDA_MALLOC(dReduceBuffer2, reduceBufferLength);
	CUDA_MALLOC(dReduceBuffer3, reduceBufferLength);
	CUDA_MALLOC(dReduceBuffer4, reduceBufferLength);
#undef CUDA_MALLOC
}

TeaLeafCudaChunk::~TeaLeafCudaChunk()
{
#ifdef ENABLE_PROFILING
	double totalTime = 0.0;

	fprintf(stdout, "%30s %7s %5s %9s\n", "Kernel name", "runtime", "calls", "bandwidth");
	std::map<std::string, double>::iterator ii = kernelTimes.begin();
	std::map<std::string, int>::iterator jj = kernelCalls.begin();

	for( ; ii != kernelTimes.end(); ++ii, ++jj)
	{
		totalTime += ii->second;

		fprintf(stdout, "%30s %9.3f %5d %7.5f\n", 
				ii->first.c_str(), ii->second, jj->second, 0.0);
	}

	fprintf(stdout, "Total kernel time %f ms\n", totalTime);
#endif
}

// Synchronises and checks for most recent CUDA error.
void TeaLeafCudaChunk::CheckErrors(int lineNum, const char* file)
{
	cudaDeviceSynchronize();
	int result = cudaGetLastError();

	if(result != cudaSuccess)
	{
		Abort(lineNum, file, "Error in %s - return code %d (%s)\n", file, result, CudaCodes(result));
	}
}

// Aborts the application.
void TeaLeafCudaChunk::Abort(int lineNum, const char* file, const char* format, ...)
{
	fprintf(stderr, "\x1b[31m");
	fprintf(stderr, "\nError at line %d in %s:", lineNum, file);
	fprintf(stderr, "\x1b[0m \n");

	va_list arglist;
	va_start(arglist, format);
	vfprintf(stderr, format, arglist);
	va_end(arglist);

	exit(1);
}

// Enumeration for the set of potential CUDA error codes.
const char* TeaLeafCudaChunk::CudaCodes(int code)
{
	switch(code)
	{
		case cudaSuccess: return "cudaSuccess"; // 0
		case cudaErrorMissingConfiguration: return "cudaErrorMissingConfiguration"; // 1
		case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation"; // 2
		case cudaErrorInitializationError: return "cudaErrorInitializationError"; // 3
		case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure"; // 4
		case cudaErrorPriorLaunchFailure: return "cudaErrorPriorLaunchFailure"; // 5
		case cudaErrorLaunchTimeout: return "cudaErrorLaunchTimeout"; // 6
		case cudaErrorLaunchOutOfResources: return "cudaErrorLaunchOutOfResources"; // 7
		case cudaErrorInvalidDeviceFunction: return "cudaErrorInvalidDeviceFunction"; // 8
		case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration"; // 9
		case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice"; // 10
		case cudaErrorInvalidValue: return "cudaErrorInvalidValue";// 11
		case cudaErrorInvalidPitchValue: return "cudaErrorInvalidPitchValue";// 12
		case cudaErrorInvalidSymbol: return "cudaErrorInvalidSymbol";// 13
		case cudaErrorMapBufferObjectFailed: return "cudaErrorMapBufferObjectFailed";// 14
		case cudaErrorUnmapBufferObjectFailed: return "cudaErrorUnmapBufferObjectFailed";// 15
		case cudaErrorInvalidHostPointer: return "cudaErrorInvalidHostPointer";// 16
		case cudaErrorInvalidDevicePointer: return "cudaErrorInvalidDevicePointer";// 17
		case cudaErrorInvalidTexture: return "cudaErrorInvalidTexture";// 18
		case cudaErrorInvalidTextureBinding: return "cudaErrorInvalidTextureBinding";// 19
		case cudaErrorInvalidChannelDescriptor: return "cudaErrorInvalidChannelDescriptor";// 20
		case cudaErrorInvalidMemcpyDirection: return "cudaErrorInvalidMemcpyDirection";// 21
		case cudaErrorAddressOfConstant: return "cudaErrorAddressOfConstant";// 22
		case cudaErrorTextureFetchFailed: return "cudaErrorTextureFetchFailed";// 23
		case cudaErrorTextureNotBound: return "cudaErrorTextureNotBound";// 24
		case cudaErrorSynchronizationError: return "cudaErrorSynchronizationError";// 25
		case cudaErrorInvalidFilterSetting: return "cudaErrorInvalidFilterSetting";// 26
		case cudaErrorInvalidNormSetting: return "cudaErrorInvalidNormSetting";// 27
		case cudaErrorMixedDeviceExecution: return "cudaErrorMixedDeviceExecution";// 28
		case cudaErrorCudartUnloading: return "cudaErrorCudartUnloading";// 29
		case cudaErrorUnknown: return "cudaErrorUnknown";// 30
		case cudaErrorNotYetImplemented: return "cudaErrorNotYetImplemented";// 31
		case cudaErrorMemoryValueTooLarge: return "cudaErrorMemoryValueTooLarge";// 32
		case cudaErrorInvalidResourceHandle: return "cudaErrorInvalidResourceHandle";// 33
		case cudaErrorNotReady: return "cudaErrorNotReady";// 34
		case cudaErrorInsufficientDriver: return "cudaErrorInsufficientDriver";// 35
		case cudaErrorSetOnActiveProcess: return "cudaErrorSetOnActiveProcess";// 36
		case cudaErrorInvalidSurface: return "cudaErrorInvalidSurface";// 37
		case cudaErrorNoDevice: return "cudaErrorNoDevice";// 38
		case cudaErrorECCUncorrectable: return "cudaErrorECCUncorrectable";// 39
		case cudaErrorSharedObjectSymbolNotFound: return "cudaErrorSharedObjectSymbolNotFound";// 40
		case cudaErrorSharedObjectInitFailed: return "cudaErrorSharedObjectInitFailed";// 41
		case cudaErrorUnsupportedLimit: return "cudaErrorUnsupportedLimit";// 42
		case cudaErrorDuplicateVariableName: return "cudaErrorDuplicateVariableName";// 43
		case cudaErrorDuplicateTextureName: return "cudaErrorDuplicateTextureName";// 44
		case cudaErrorDuplicateSurfaceName: return "cudaErrorDuplicateSurfaceName";// 45
		case cudaErrorDevicesUnavailable: return "cudaErrorDevicesUnavailable";// 46
		case cudaErrorInvalidKernelImage: return "cudaErrorInvalidKernelImage";// 47
		case cudaErrorNoKernelImageForDevice: return "cudaErrorNoKernelImageForDevice";// 48
		case cudaErrorIncompatibleDriverContext: return "cudaErrorIncompatibleDriverContext";// 49
		case cudaErrorPeerAccessAlreadyEnabled: return "cudaErrorPeerAccessAlreadyEnabled";// 50
		case cudaErrorPeerAccessNotEnabled: return "cudaErrorPeerAccessNotEnabled";// 51
		case cudaErrorDeviceAlreadyInUse: return "cudaErrorDeviceAlreadyInUse";// 52
		case cudaErrorProfilerDisabled: return "cudaErrorProfilerDisabled";// 53
		case cudaErrorProfilerNotInitialized: return "cudaErrorProfilerNotInitialized";// 54
		case cudaErrorProfilerAlreadyStarted: return "cudaErrorProfilerAlreadyStarted";// 55
		case cudaErrorProfilerAlreadyStopped: return "cudaErrorProfilerAlreadyStopped";// 56
		case cudaErrorAssert: return "cudaErrorAssert";// 57
		case cudaErrorTooManyPeers: return "cudaErrorTooManyPeers";// 58
		case cudaErrorHostMemoryAlreadyRegistered: return "cudaErrorHostMemoryAlreadyRegistered";// 59
		case cudaErrorHostMemoryNotRegistered: return "cudaErrorHostMemoryNotRegistered";// 60
		case cudaErrorOperatingSystem: return "cudaErrorOperatingSystem";// 61
		case cudaErrorStartupFailure: return "cudaErrorStartupFailure";// 62
		case cudaErrorApiFailureBase: return "cudaErrorApiFailureBase";// 63
		default: return "Unknown error";
	}
}


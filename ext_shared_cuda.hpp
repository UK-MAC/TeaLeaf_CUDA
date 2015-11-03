#ifndef __CUDA_SHARED
#define __CUDA_SHARED

#define HALO_PAD 2
#define CHUNK_LEFT 1
#define CHUNK_RIGHT 2
#define CHUNK_BOTTOM 3
#define CHUNK_TOP 4
#define NUM_FACES 4
#define EXTERNAL_FACE -1

#define FIELD_DENSITY 1
#define FIELD_ENERGY0 2
#define FIELD_ENERGY1 3
#define FIELD_U 4
#define FIELD_P 5
#define FIELD_SD 6
#define NUM_FIELDS 6
#define MAX_DEPTH 2

#define CONDUCTIVITY 1
#define RECIP_CONDUCTIVITY 2

#define SMVP(a) \
	(1.0 + (Kx[index+1]+Kx[index])\
	 + (Ky[index+xMax]+Ky[index]))*a[index]\
	 - (Kx[index+1]*a[index+1]+Kx[index]*a[index-1])\
	 - (Ky[index+xMax]*a[index+xMax]+Ky[index]*a[index-xMax]);

#ifdef ENABLE_PROFILING
#define START_PROFILING()\
	{\
		cudaEventCreate(&start);\
		cudaEventCreate(&stop);\
		cudaEventRecord(start, 0);\
	}

#define STOP_PROFILING(kernelName)\
	{\
		cudaEventRecord(stop, 0);\
		cudaEventSynchronize(stop);\
		cudaEventElapsedTime(&span, start, stop);\
		cudaEventDestroy(start);\
		cudaEventDestroy(stop);\
		std::string _name(kernelName);\
		if (kernelTimes.end() != kernelTimes.find(_name))\
		{\
			kernelTimes.at(_name) += span;\
			kernelCalls.at(_name) += 1;\
		}\
		else\
		{\
			kernelTimes[_name] = span;\
			kernelCalls[_name] = 1;\
		}\
	}
#else
#define START_PROFILING();
#define STOP_PROFILING(kernelName);
#endif

#define PRE_KERNEL(pad)\
	const int innerX = xCells-pad;\
	const int innerY = yCells-pad;\
	const int numThreads = innerX*innerY;\
	const int numBlocks = std::ceil(numThreads/(float)BLOCK_SIZE);\
	START_PROFILING();

#define POST_KERNEL(kernelName)\
	STOP_PROFILING(kernelName);\
	CheckErrors(__LINE__,__FILE__);

typedef void (*CuKnlUpdateHaloType)(
		const int xMax,
		const int yMax,
		const int depth,
		double* buffer);

typedef void (*CuKnlPackType)(
		const int xMax,
		const int yMax,
		const int innerX,
		const int innerY,
		double* field,
		double* buffer,
		const int depth);

#endif

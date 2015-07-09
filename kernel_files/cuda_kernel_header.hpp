#include "../ftocmacros.h"

// size of workgroup/block - 256 seems to be optimal
#ifndef BLOCK_SZ 
    #define BLOCK_SZ 128
#endif

#define LOCAL_Y (4)
#define LOCAL_X (BLOCK_SZ/LOCAL_Y)

const static dim3 block_shape(LOCAL_X, LOCAL_Y);

#define WITHIN_BOUNDS \
    (row >= (y_min + 1) - 0 && row <= (y_max + 1) + 0 \
    && column >= (x_min + 1) - 0 && column <= (x_max + 1) + 0)

/*
*  access a value in a 2d array given the x and y offset from current thread
*  index, adding or subtracting a bit more if it is one of the arrays with
*  bigger rows
*/
#define THARR2D(x_offset, y_offset, big_row)\
    ( glob_id                               \
    + (x_offset)                            \
    + ((y_offset) * (x_max + 4))            \
    + ((big_row) * (row + (y_offset))) )

// kernel indexes uses in all kernels
#define __kernel_indexes                    \
    __attribute__((__unused__)) const int x_min = kernel_info.x_min; \
    __attribute__((__unused__)) const int x_max = kernel_info.x_max; \
    __attribute__((__unused__)) const int y_min = kernel_info.y_min; \
    __attribute__((__unused__)) const int y_max = kernel_info.y_max; \
    __attribute__((__unused__)) const int HALO_DEPTH = kernel_info.halo_depth; \
    __attribute__((__unused__)) const int row = blockIdx.y*blockDim.y + threadIdx.y + kernel_info.y_offset; \
    __attribute__((__unused__)) const int column = blockIdx.x*blockDim.x + threadIdx.x + kernel_info.x_offset; \
    __attribute__((__unused__)) const int glob_id = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x*blockDim.y + \
        threadIdx.y*blockDim.x + threadIdx.x; \
    __attribute__((__unused__)) const int lid = threadIdx.x;

__device__ inline double SUM(double a, double b){return a+b;}

template < typename T, int offset >
class Reduce
{
public:
    __device__ inline static void run
    (T* array, T* out, T(*func)(T, T))
    {
        // only need to synch if not working within a warp
        if (offset > 16)
        {
            __syncthreads();
        }

        // only continue if it's in the lower half
        if (threadIdx.x < offset)
        {
            array[threadIdx.x] = func(array[threadIdx.x], array[threadIdx.x + offset]);
            Reduce< T, offset/2 >::run(array, out, func);
        }
    }
};

template < typename T >
class Reduce < T, 0 >
{
public:
    __device__ inline static void run
    (T* array, T* out, T(*func)(T, T))
    {
        out[blockIdx.x] = array[0];
    }
};


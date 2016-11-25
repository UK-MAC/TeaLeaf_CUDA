#ifndef __CUDA_KERNEL_HEADER
#define __CUDA_KERNEL_HEADER
#include "../ftocmacros.h"
#include <cstdio>

// size of workgroup/block
#ifndef BLOCK_SZ 
    #define BLOCK_SZ 128
#endif

#define JACOBI_BLOCK_SIZE 4

#define LOCAL_Y (JACOBI_BLOCK_SIZE)
#define LOCAL_X (BLOCK_SZ/LOCAL_Y)

const static dim3 block_shape(LOCAL_X, LOCAL_Y);

#define WITHIN_BOUNDS \
    (row <= (y_max - 1) + HALO_DEPTH + kernel_info.kernel_y_max \
    && column <= (x_max - 1) + HALO_DEPTH + kernel_info.kernel_x_max)

/*
*  access a value in a 2d array given the x and y offset from current thread
*  index, adding or subtracting a bit more if it is one of the arrays with
*  bigger rows
*/
#define THARR2D(x_offset, y_offset, big_row)                \
    ( (column + x_offset)                                   \
    + (row + y_offset)*(x_max + 2*HALO_DEPTH + big_row))

static __device__ int get_global_id
(int dim)
{
    if (dim == 0)
    {
        return blockIdx.x*blockDim.x + threadIdx.x;
    }
    else if (dim == 1)
    {
        return blockIdx.y*blockDim.y + threadIdx.y;
    }
    return 0;
}

static __device__ int get_global_size
(int dim)
{
    if (dim == 0)
    {
        return blockDim.x*gridDim.x;
    }
    else if (dim == 1)
    {
        return blockDim.y*gridDim.y;
    }
    return 0;
}

static __device__ int get_local_id
(int dim)
{
    if (dim == 0)
    {
        return threadIdx.x;
    }
    else if (dim == 1)
    {
        return threadIdx.y;
    }
    return 0;
}

// kernel indexes uses in all kernels
#define __kernel_indexes                    \
    __attribute__((__unused__)) const int x_min = kernel_info.x_min; \
    __attribute__((__unused__)) const int x_max = kernel_info.x_max; \
    __attribute__((__unused__)) const int y_min = kernel_info.y_min; \
    __attribute__((__unused__)) const int y_max = kernel_info.y_max; \
    __attribute__((__unused__)) const int HALO_DEPTH = kernel_info.halo_depth; \
    __attribute__((__unused__)) const int PRECONDITIONER = kernel_info.preconditioner_type; \
    __attribute__((__unused__)) const int column = get_global_id(0) + kernel_info.x_offset; \
    __attribute__((__unused__)) const int row = get_global_id(1) + kernel_info.y_offset; \
    __attribute__((__unused__)) const int glob_id = row*get_global_size(0) + column; \
    __attribute__((__unused__)) const int loc_column = get_local_id(0); \
    __attribute__((__unused__)) const int loc_row = get_local_id(1); \
    __attribute__((__unused__)) const int lid = loc_row*LOCAL_X + loc_column; \
    __attribute__((__unused__)) const int block_id = blockIdx.x + blockIdx.y*gridDim.x;

    //__attribute__((__unused__)) const int glob_id = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x*blockDim.y + \
        threadIdx.y*blockDim.x + threadIdx.x; \

template < typename T >
__device__ inline T SUM(T a, T b){return a+b;}
template < typename T >
__device__ inline T MAXIMUM(T a, T b){return a < b ? b : a;}
template < typename T >
__device__ inline T MINIMUM(T a, T b){return a < b ? a : b;}

template < typename T, int offset >
class Reduce
{
public:
    __device__ inline static void run
    (T* array, T* out, T(*func)(T, T))
    {
        __attribute__((__unused__)) const int loc_column = get_local_id(0); \
        __attribute__((__unused__)) const int loc_row = get_local_id(1); \
        __attribute__((__unused__)) const int lid = loc_row*LOCAL_X + loc_column; \
        __attribute__((__unused__)) const int block_id = blockIdx.x + blockIdx.y*gridDim.x;

        // only need to synch if not working within a warp
        if (offset > 16)
        {
            __syncthreads();
        }

        // only continue if it's in the lower half
        if (lid < offset)
        {
            array[lid] = func(array[lid], array[lid + offset]);
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
        __attribute__((__unused__)) const int loc_column = get_local_id(0); \
        __attribute__((__unused__)) const int loc_row = get_local_id(1); \
        __attribute__((__unused__)) const int lid = loc_row*LOCAL_X + loc_column; \
        __attribute__((__unused__)) const int block_id = blockIdx.x + blockIdx.y*gridDim.x;

        out[block_id] = array[0];
    }
};
#endif //__CUDA_KERNEL_HEADER

/*Crown Copyright 2012 AWE.
 *
 * This file is part of CloverLeaf.
 *
 * CloverLeaf is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * CloverLeaf is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * CloverLeaf. If not, see http://www.gnu.org/licenses/.
 */

/*
 *  @brief CUDA initialisation
 *  @author Michael Boulton NVIDIA Corporation
 *  @details Initialises CUDA devices and global storage
 */

#if defined(MPI_HDR)
extern "C" void tea_get_rank_(int*);
#endif

#include "cuda_common.hpp"
#include "cuda_strings.hpp"

#include <sstream>
#include <cstdio>
#include <cassert>

CloverleafCudaChunk cuda_chunk;

extern "C" void initialise_cuda_
(INITIALISE_ARGS)
{
    cuda_chunk = CloverleafCudaChunk(in_x_min,
                                in_x_max,
                                in_y_min,
                                in_y_max,
                                in_profiler_on);
}

CloverleafCudaChunk::CloverleafCudaChunk
(void)
{
    ;
}

CloverleafCudaChunk::CloverleafCudaChunk
(INITIALISE_ARGS)
:x_min(*in_x_min),
x_max(*in_x_max),
y_min(*in_y_min),
y_max(*in_y_max),
profiler_on(*in_profiler_on),
num_blocks((((*in_x_max)+5)*((*in_y_max)+5))/BLOCK_SZ)
{
    // FIXME (and opencl really)
    // make a better platform agnostic way of selecting devices

    int rank;
#if defined(MPI_HDR)
    tea_get_rank_(&rank);
#else
    rank = 0;
#endif

    // Read in from file - easier than passing in from fortran
    FILE* input = fopen("tea.in", "r");
    if (NULL == input)
    {
        // should never happen
        DIE("Input file not found\n");
    }

    // find out which solver to use
    bool tl_use_jacobi = clover::paramEnabled(input, "tl_use_jacobi");
    bool tl_use_cg = clover::paramEnabled(input, "tl_use_cg");
    bool tl_use_chebyshev = clover::paramEnabled(input, "tl_use_chebyshev");
    bool tl_use_ppcg = clover::paramEnabled(input, "tl_use_ppcg");
    preconditioner_on = clover::paramEnabled(input, "tl_preconditioner_on");

    if(!rank)fprintf(stdout, "Solver to use: ");
    if (tl_use_ppcg)
    {
        tea_solver = TEA_ENUM_PPCG;
        if(!rank)fprintf(stdout, "PPCG\n");
    }
    else if (tl_use_chebyshev)
    {
        tea_solver = TEA_ENUM_CHEBYSHEV;
        if(!rank)fprintf(stdout, "Chebyshev + CG\n");
    }
    else if (tl_use_cg)
    {
        tea_solver = TEA_ENUM_CG;
        if(!rank)fprintf(stdout, "Conjugate gradient\n");
    }
    else if (tl_use_jacobi)
    {
        tea_solver = TEA_ENUM_JACOBI;
        if(!rank)fprintf(stdout, "Jacobi\n");
    }
    else
    {
        tea_solver = TEA_ENUM_JACOBI;
        if(!rank)fprintf(stdout, "Jacobi (no solver specified in tea.in)\n");
    }

    int device_id = clover::preferredDevice(input);
    device_id = device_id < 0 ? 0 : device_id;

    fclose(input);

//#ifdef MANUALLY_CHOOSE_GPU
    // choose device 0 unless specified
    cudaThreadExit();
    int num_devices;
    cudaGetDeviceCount(&num_devices);

    fprintf(stdout, "%d devices available in rank %d - would use %d - adding %d - choosing %d\n",
            num_devices, rank, device_id, rank%num_devices, device_id + rank % num_devices);
    fflush(stdout);
    device_id += rank % num_devices;

    int err = cudaSetDevice(device_id);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Setting device id to %d in rank %d failed with error code %d\n", device_id, rank, err);
        errorHandler(__LINE__, __FILE__);
    }
//#endif

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::cout << "CUDA using " << prop.name << std::endl;

    #define CUDA_ARRAY_ALLOC(arr, size)     \
            cudaMalloc((void**) &arr, size);\
            errorHandler(__LINE__, __FILE__);\
            cudaDeviceSynchronize();        \
            cudaMemset(arr, 0, size);       \
            cudaDeviceSynchronize();        \
            CUDA_ERR_CHECK;

    CUDA_ARRAY_ALLOC(volume, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(soundspeed, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(viscosity, BUFSZ2D(0, 0));

    CUDA_ARRAY_ALLOC(density, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(energy0, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(energy1, BUFSZ2D(0, 0));

    CUDA_ARRAY_ALLOC(xarea, BUFSZ2D(1, 0));
    CUDA_ARRAY_ALLOC(yarea, BUFSZ2D(0, 1));

    CUDA_ARRAY_ALLOC(cellx, BUFSZX(0));
    CUDA_ARRAY_ALLOC(celldx, BUFSZX(0));
    CUDA_ARRAY_ALLOC(vertexx, BUFSZX(1));
    CUDA_ARRAY_ALLOC(vertexdx, BUFSZX(1));

    CUDA_ARRAY_ALLOC(celly, BUFSZY(0));
    CUDA_ARRAY_ALLOC(celldy, BUFSZY(0));
    CUDA_ARRAY_ALLOC(vertexy, BUFSZY(1));
    CUDA_ARRAY_ALLOC(vertexdy, BUFSZY(1));

    CUDA_ARRAY_ALLOC(u, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(u0, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(z, BUFSZ2D(0, 0));

    CUDA_ARRAY_ALLOC(vector_p, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(vector_r, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(vector_w, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(vector_Mi, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(vector_Kx, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(vector_Ky, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(vector_sd, BUFSZ2D(0, 0));

    CUDA_ARRAY_ALLOC(left_buffer, BUFSZ2D(0, 0)/(x_max/2));
    CUDA_ARRAY_ALLOC(right_buffer, BUFSZ2D(0, 0)/(x_max/2));
    CUDA_ARRAY_ALLOC(bottom_buffer, BUFSZ2D(0, 0)/(y_max/2));
    CUDA_ARRAY_ALLOC(top_buffer, BUFSZ2D(0, 0)/(y_max/2));

    CUDA_ARRAY_ALLOC(reduce_buf_1, num_blocks*sizeof(double));
    CUDA_ARRAY_ALLOC(reduce_buf_2, num_blocks*sizeof(double));
    CUDA_ARRAY_ALLOC(reduce_buf_3, num_blocks*sizeof(double));
    CUDA_ARRAY_ALLOC(reduce_buf_4, num_blocks*sizeof(double));

    reduce_ptr_1 = thrust::device_ptr< double >(reduce_buf_1);
    reduce_ptr_2 = thrust::device_ptr< double >(reduce_buf_2);
    reduce_ptr_3 = thrust::device_ptr< double >(reduce_buf_3);
    reduce_ptr_4 = thrust::device_ptr< double >(reduce_buf_4);

    thr_cellx = thrust::device_ptr< double >(cellx);
    thr_celly = thrust::device_ptr< double >(celly);
    thr_density = thrust::device_ptr< double >(density);
    thr_energy0 = thrust::device_ptr< double >(energy0);
    thr_soundspeed = thrust::device_ptr< double >(soundspeed);

    #undef CUDA_ARRAY_ALLOC

#define ADD_BUFFER_DBG_MAP(name) arr_names[#name] = name;
    ADD_BUFFER_DBG_MAP(volume);
    ADD_BUFFER_DBG_MAP(soundspeed);
    ADD_BUFFER_DBG_MAP(viscosity);

    ADD_BUFFER_DBG_MAP(u);
    arr_names["p"] = vector_p;

    ADD_BUFFER_DBG_MAP(vector_r);
    ADD_BUFFER_DBG_MAP(vector_w);
    ADD_BUFFER_DBG_MAP(vector_Mi);
    ADD_BUFFER_DBG_MAP(vector_Kx);
    ADD_BUFFER_DBG_MAP(vector_Ky);

    ADD_BUFFER_DBG_MAP(density);
    ADD_BUFFER_DBG_MAP(energy0);
    ADD_BUFFER_DBG_MAP(energy1);
    ADD_BUFFER_DBG_MAP(xarea);
    ADD_BUFFER_DBG_MAP(yarea);

    ADD_BUFFER_DBG_MAP(cellx);
    ADD_BUFFER_DBG_MAP(celly);
    ADD_BUFFER_DBG_MAP(celldx);
    ADD_BUFFER_DBG_MAP(celldy);
    ADD_BUFFER_DBG_MAP(vertexx);
    ADD_BUFFER_DBG_MAP(vertexy);
    ADD_BUFFER_DBG_MAP(vertexdx);
    ADD_BUFFER_DBG_MAP(vertexdy);
#undef ADD_BUFFER_DBG_MAP
}


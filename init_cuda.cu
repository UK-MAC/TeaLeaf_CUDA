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

#include "cuda_common.hpp"
#include "cuda_strings.hpp"

#include "mpi.h"
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
                                in_y_max);
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
y_max(*in_y_max)
{
    // FIXME (and opencl really)
    // make a better platform agnostic way of selecting devices

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Read in from file - easier than passing in from fortran
    std::ifstream input("tea.in");
    input.exceptions(std::ifstream::badbit);

    if (!input.is_open())
    {
        // should never happen
        DIE("Input file not found\n");
    }

    profiler_on = paramEnabled(input, "profiler_on");

    int device_id = readInt(input, "cuda_device");
    device_id = (device_id < 0) ? 0 : device_id;

    cudaThreadExit();

    // account for MPI
    int num_devices;
    cudaGetDeviceCount(&num_devices);

    if (num_devices < device_id)
    {
        DIE("Device id %d specified in tea.in, but only %d devices available", device_id, num_devices);
    }

    int err = cudaSetDevice(device_id);

    if (err != cudaSuccess)
    {
        DIE("Setting device id to %d in rank %d failed with error code %d\n", device_id, rank, err);
    }

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::cout << "CUDA in rank " << rank << " using " << prop.name << std::endl;

    int file_halo_depth = readInt(input, "halo_depth");
    halo_exchange_depth = file_halo_depth;

    if (halo_exchange_depth < 1)
    {
        DIE("Halo exchange depth unspecified or was too small");
    }

    bool tl_use_jacobi = paramEnabled(input, "tl_use_jacobi");
    bool tl_use_cg = paramEnabled(input, "tl_use_cg");
    bool tl_use_chebyshev = paramEnabled(input, "tl_use_chebyshev");
    bool tl_use_ppcg = paramEnabled(input, "tl_use_ppcg");

    // set solve
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

    std::string desired_preconditioner = readString(input, "tl_preconditioner_type");

    // set preconditioner type
    if(!rank)fprintf(stdout, "Preconditioner to use: ");
    if (desired_preconditioner.find("jac_diag") != std::string::npos)
    {
        preconditioner_type = TL_PREC_JAC_DIAG;
        if(!rank)fprintf(stdout, "Diagonal Jacobi\n");
    }
    else if (desired_preconditioner.find("jac_block") != std::string::npos)
    {
        preconditioner_type = TL_PREC_JAC_BLOCK;
        if(!rank)fprintf(stdout, "Block Jacobi\n");
    }
    else if (desired_preconditioner.find("none") != std::string::npos)
    {
        preconditioner_type = TL_PREC_NONE;
        if(!rank)fprintf(stdout, "None\n");
    }
    else
    {
        preconditioner_type = TL_PREC_NONE;
        if(!rank)fprintf(stdout, "None (no preconditioner specified in tea.in)\n");
    }

    initSizes();
    initBuffers();
}

void CloverleafCudaChunk::initSizes
(void)
{
    grid_dim = dim3(
        std::ceil((x_max + 2.0*halo_exchange_depth)/LOCAL_X),
        std::ceil((y_max + 2.0*halo_exchange_depth)/LOCAL_Y));
    num_blocks = grid_dim.x*grid_dim.y;

    #define UPDATE_HALO_SIZE 32

    for (int depth = 1; depth <= 2; depth++)
    {
        update_bt_block_sizes[depth] = dim3(UPDATE_HALO_SIZE, 1);
        update_lr_block_sizes[depth] = dim3(1, UPDATE_HALO_SIZE);
    }

    update_bt_block_sizes[halo_exchange_depth] = update_bt_block_sizes[1];
    update_lr_block_sizes[halo_exchange_depth] = update_lr_block_sizes[1];

    std::map<int, dim3>::iterator typedef irangeit;
    for (irangeit key = update_lr_block_sizes.begin();
        key != update_lr_block_sizes.end(); key++)
    {
        int depth = key->first;

        int min_update_bt_grid_dim = x_max + 2*depth;
        int min_update_lr_grid_dim = y_max + 2*depth;

        int num_blocks_bt = 1;
        int num_blocks_lr = 1;

        while (update_bt_block_sizes[depth].x*num_blocks_bt < min_update_bt_grid_dim)
            num_blocks_bt++;
        while (update_lr_block_sizes[depth].y*num_blocks_lr < min_update_lr_grid_dim)
            num_blocks_lr++;

        update_bt_num_blocks[depth] = dim3(num_blocks_bt, depth);
        update_lr_num_blocks[depth] = dim3(depth, num_blocks_lr);
    }

    kernel_info_t kernel_info_generic;

    kernel_info_generic.x_min = x_min;
    kernel_info_generic.x_max = x_max;
    kernel_info_generic.y_min = y_min;
    kernel_info_generic.y_max = y_max;

    kernel_info_generic.halo_depth = halo_exchange_depth;
    kernel_info_generic.preconditioner_type = preconditioner_type;

    kernel_info_generic.x_offset = halo_exchange_depth;
    kernel_info_generic.y_offset = halo_exchange_depth;

    kernel_info_map["device_initialise_chunk_kernel"] = kernel_info_t(kernel_info_generic, -halo_exchange_depth, halo_exchange_depth, -halo_exchange_depth, halo_exchange_depth);

    kernel_info_map["device_initialise_chunk_kernel_vertex"] = kernel_info_t(kernel_info_generic, -halo_exchange_depth, halo_exchange_depth, -halo_exchange_depth, halo_exchange_depth);
    kernel_info_map["device_generate_chunk_init"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
    kernel_info_map["device_generate_chunk_kernel"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
    kernel_info_map["device_generate_chunk_init_u"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
    kernel_info_map["device_generate_chunk"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);

    kernel_info_map["device_set_field_kernel"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
    kernel_info_map["device_field_summary_kernel"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);

    kernel_info_map["device_update_halo_top"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
    kernel_info_map["device_update_halo_bottom"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
    kernel_info_map["device_update_halo_left"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
    kernel_info_map["device_update_halo_right"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);

    kernel_info_map["device_pack_left_buffer"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
    kernel_info_map["device_unpack_left_buffer"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
    kernel_info_map["device_pack_right_buffer"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
    kernel_info_map["device_unpack_right_buffer"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
    kernel_info_map["device_pack_bottom_buffer"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
    kernel_info_map["device_unpack_bottom_buffer"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
    kernel_info_map["device_pack_top_buffer"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
    kernel_info_map["device_unpack_top_buffer"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);

    if (tea_solver == TEA_ENUM_CG ||
    tea_solver == TEA_ENUM_CHEBYSHEV ||
    tea_solver == TEA_ENUM_PPCG)
    {
        kernel_info_map["device_tea_leaf_cg_solve_calc_w"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
        kernel_info_map["device_tea_leaf_cg_solve_calc_ur"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
        kernel_info_map["device_tea_leaf_cg_solve_calc_p"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
        kernel_info_map["device_tea_leaf_cg_solve_init_p"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);

        if (tea_solver == TEA_ENUM_CHEBYSHEV)
        {
            kernel_info_map["device_tea_leaf_cheby_solve_init_p"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
            kernel_info_map["device_tea_leaf_cheby_solve_calc_u"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
            kernel_info_map["device_tea_leaf_cheby_solve_calc_p"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
        }
        else if (tea_solver == TEA_ENUM_PPCG)
        {
            kernel_info_map["device_tea_leaf_ppcg_solve_init_sd"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
            kernel_info_map["device_tea_leaf_ppcg_solve_calc_sd"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
            kernel_info_map["device_tea_leaf_ppcg_solve_update_r"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
        }
    }
    else
    {
        kernel_info_map["device_tea_leaf_jacobi_copy_u"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
        kernel_info_map["device_tea_leaf_jacobi_solve"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
    }

    kernel_info_map["device_tea_leaf_finalise"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
    kernel_info_map["device_tea_leaf_calc_residual"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
    kernel_info_map["device_tea_leaf_calc_2norm"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);

    kernel_info_map["device_tea_leaf_block_init"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);
    kernel_info_map["device_tea_leaf_block_solve"] = kernel_info_t(kernel_info_generic, 0, 0, 0, 0);

    kernel_info_map["device_tea_leaf_init_common"] = kernel_info_t(kernel_info_generic, -halo_exchange_depth, halo_exchange_depth, -halo_exchange_depth, halo_exchange_depth);
    kernel_info_map["device_tea_leaf_zero_boundaries"] = kernel_info_t(kernel_info_generic, -halo_exchange_depth, halo_exchange_depth, -halo_exchange_depth, halo_exchange_depth);
    kernel_info_map["device_tea_leaf_init_jac_diag"] = kernel_info_t(kernel_info_generic, -halo_exchange_depth, halo_exchange_depth, -halo_exchange_depth, halo_exchange_depth);
}

void CloverleafCudaChunk::initBuffers
(void)
{
    #define CUDA_ARRAY_ALLOC(arr, size)     \
            cudaMalloc((void**) &arr, size);\
            errorHandler(__LINE__, __FILE__);\
            cudaDeviceSynchronize();        \
            cudaMemset(arr, 0, size);       \
            cudaDeviceSynchronize();        \
            CUDA_ERR_CHECK;

    // number of bytes to allocate for x size array
    #define BUFSZX(x_extra)   \
        ( ((x_max) + 2*halo_exchange_depth + x_extra)       \
        * sizeof(double) )

    // number of bytes to allocate for y size array
    #define BUFSZY(y_extra)   \
        ( ((y_max) + 2*halo_exchange_depth + y_extra)       \
        * sizeof(double) )

    // number of bytes to allocate for 2d array
    #define BUFSZ2D(x_extra, y_extra)   \
        ( ((x_max) + 2*halo_exchange_depth + x_extra)       \
        * ((y_max) + 2*halo_exchange_depth + y_extra)       \
        * sizeof(double) )

    CUDA_ARRAY_ALLOC(volume, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(soundspeed, BUFSZ2D(0, 0));

    CUDA_ARRAY_ALLOC(tri_cp, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(tri_bfp, BUFSZ2D(0, 0));

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
    CUDA_ARRAY_ALLOC(vector_z, BUFSZ2D(0, 0));

    CUDA_ARRAY_ALLOC(vector_p, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(vector_r, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(vector_w, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(vector_Mi, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(vector_Kx, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(vector_Ky, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(vector_sd, BUFSZ2D(0, 0));

    CUDA_ARRAY_ALLOC(left_buffer, (y_max+2*halo_exchange_depth)*halo_exchange_depth*NUM_BUFFERED_FIELDS*sizeof(double));
    CUDA_ARRAY_ALLOC(right_buffer, (y_max+2*halo_exchange_depth)*halo_exchange_depth*NUM_BUFFERED_FIELDS*sizeof(double));
    CUDA_ARRAY_ALLOC(bottom_buffer, (x_max+2*halo_exchange_depth)*halo_exchange_depth*NUM_BUFFERED_FIELDS*sizeof(double));
    CUDA_ARRAY_ALLOC(top_buffer, (x_max+2*halo_exchange_depth)*halo_exchange_depth*NUM_BUFFERED_FIELDS*sizeof(double));

    CUDA_ARRAY_ALLOC(reduce_buf_1, num_blocks*sizeof(double));
    CUDA_ARRAY_ALLOC(reduce_buf_2, num_blocks*sizeof(double));
    CUDA_ARRAY_ALLOC(reduce_buf_3, num_blocks*sizeof(double));
    CUDA_ARRAY_ALLOC(reduce_buf_4, num_blocks*sizeof(double));

    // To make sure memory is allocated later on
    ch_alphas_device = NULL;
    ch_betas_device = NULL;

    #undef CUDA_ARRAY_ALLOC

#define ADD_BUFFER_DBG_MAP(name) arr_names[#name] = name;
    ADD_BUFFER_DBG_MAP(volume);
    ADD_BUFFER_DBG_MAP(soundspeed);

    ADD_BUFFER_DBG_MAP(u);
    ADD_BUFFER_DBG_MAP(u0);

    ADD_BUFFER_DBG_MAP(vector_p);
    ADD_BUFFER_DBG_MAP(vector_r);
    ADD_BUFFER_DBG_MAP(vector_w);
    ADD_BUFFER_DBG_MAP(vector_sd);
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


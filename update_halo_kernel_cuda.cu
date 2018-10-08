/*Crown Copyright 2012 AWE.
 *
 * This file is part of TeaLeaf.
 *
 * TeaLeaf is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * TeaLeaf is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * TeaLeaf. If not, see http://www.gnu.org/licenses/.
 */

/*
 *  @brief CUDA kernel to update the external halo cells in a chunk.
 *  @author Michael Boulton NVIDIA Corporation
 *  @details Updates halo cells for the required fields at the required depth
 *  for any halo cells that lie on an external boundary. The location and type
 *  of data governs how this is carried out. External boundaries are always
 *  reflective.
 */

#include "cuda_common.hpp"
#include "kernel_files/update_halo_kernel.cuknl"

extern "C" void update_halo_kernel_cuda_
(const int* chunk_neighbours,
const int* fields,
const int* depth)
{
    cuda_chunk.update_halo_kernel(fields, *depth, chunk_neighbours);
}

void TealeafCudaChunk::update_array_boundary
(cell_info_t const& grid_type,
const int* chunk_neighbours,
double* cur_array_d,
int depth)
{
    #define CHECK_LAUNCH(face, side, dir)                               \
    if (EXTERNAL_FACE == chunk_neighbours[CHUNK_ ## face - 1])          \
    {                                                                   \
        TIME_KERNEL_BEGIN;  \
        device_update_halo_kernel_##face                         \
        <<<update_##side##_num_blocks[depth], update_##side##_block_sizes[depth] >>> \
        (kernel_info_map.at("device_update_halo_kernel_"#face), \
            grid_type, cur_array_d, depth); \
        CUDA_ERR_CHECK;                                                 \
        TIME_KERNEL_END(device_update_halo_kernel_##face); \
    }
    CHECK_LAUNCH(bottom, bt, x);
    CHECK_LAUNCH(top, bt, x);
    CHECK_LAUNCH(left, lr, y);
    CHECK_LAUNCH(right, lr, y);

    #undef CHECK_LAUNCH
}

void TealeafCudaChunk::update_halo_kernel
(const int* fields,
const int depth,
const int* chunk_neighbours)
{
    #define HALO_UPDATE_RESIDENT(arr, grid_type)        \
    {if (1 == fields[FIELD_##arr - 1])                  \
    {                                                   \
        update_array_boundary(grid_type, chunk_neighbours, arr, depth);   \
    }}

    HALO_UPDATE_RESIDENT(density, CELL);
    HALO_UPDATE_RESIDENT(energy0, CELL);
    HALO_UPDATE_RESIDENT(energy1, CELL);

    HALO_UPDATE_RESIDENT(u, CELL);
    HALO_UPDATE_RESIDENT(vector_p, CELL);
    HALO_UPDATE_RESIDENT(vector_sd, CELL);
    HALO_UPDATE_RESIDENT(vector_r, CELL);

    #undef HALO_UPDATE_RESIDENT
}


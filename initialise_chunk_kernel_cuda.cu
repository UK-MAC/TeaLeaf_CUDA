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
 *  @brief CUDA driver for chunk initialisation.
 *  @author Michael Boulton NVIDIA Corporation
 *  @details Invokes the user specified chunk initialisation kernel.
 */

#include "cuda_common.hpp"

extern "C" void initialise_chunk_kernel_cuda_
(double* d_xmin, double* d_ymin, double* d_dx, double* d_dy)
{
    chunk.initialise_chunk_kernel(*d_xmin, *d_ymin, *d_dx, *d_dy);
}

void CloverleafCudaChunk::initialise_chunk_kernel
(double d_xmin, double d_ymin, double d_dx, double d_dy)
{
    CUDA_BEGIN_PROFILE;

    device_initialise_chunk_kernel_vertex_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min,x_max,y_min,y_max, d_xmin, d_ymin, d_dx, d_dy, 
        vertexx, vertexdx, vertexy, vertexdy);
    CUDA_ERR_CHECK;

    device_initialise_chunk_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min,x_max,y_min,y_max, d_xmin, d_ymin, d_dx, d_dy, 
        vertexx, vertexdx, vertexy, vertexdy,
        cellx, celldx, celly, celldy,
        volume, xarea, yarea);
    CUDA_ERR_CHECK;

    CUDA_END_PROFILE;
}


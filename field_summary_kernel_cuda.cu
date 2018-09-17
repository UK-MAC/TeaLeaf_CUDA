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
 *  @brief CUDA field summary kernel
 *  @author Michael Boulton NVIDIA Corporation
 *  @details The total mass, internal energy, kinetic energy and volume weighted
 *  pressure for the chunk is calculated.
 */

#include "cuda_common.hpp"
#include "kernel_files/field_summary_kernel.cuknl"
#include "host_reductions_kernel_cuda.hpp"

extern "C" void field_summary_kernel_cuda_
(double* vol, double* mass, double* ie, double* temp)
{
    cuda_chunk.field_summary_kernel(vol, mass, ie, temp);
}

void TealeafCudaChunk::field_summary_kernel
(double* vol, double* mass, double* ie, double* temp)
{
    CUDALAUNCH(device_field_summary_kernel, volume, density,
        energy1, u, reduce_buf_1, reduce_buf_2, reduce_buf_3,
        reduce_buf_4);
    ReduceToHost<double>::sum(reduce_buf_1, vol,  num_blocks);
    ReduceToHost<double>::sum(reduce_buf_2, mass, num_blocks);
    ReduceToHost<double>::sum(reduce_buf_3, ie,   num_blocks);
    ReduceToHost<double>::sum(reduce_buf_4, temp, num_blocks);
}


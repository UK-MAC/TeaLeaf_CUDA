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
 *  @brief CUDA revert kernel.
 *  @author Michael Boulton NVIDIA Corporation
 *  @details Takes the half step field data used in the predictor and reverts
 *  it to the start of step data, ready for the corrector.
 *  Note that this does not seem necessary in this proxy-app but should be
 *  left in to remain relevant to the full method.
 */

#include "cuda_common.hpp"

extern "C" void revert_kernel_cuda_
(void)
{
    chunk.revert_kernel();
}

void CloverleafCudaChunk::revert_kernel
(void)
{
    CUDA_BEGIN_PROFILE;

    device_revert_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min,x_max,y_min,y_max, density0, density1, energy0, energy1);
    CUDA_ERR_CHECK;

    CUDA_END_PROFILE;
}


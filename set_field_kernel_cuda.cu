#include "cuda_common.hpp"
#include "kernel_files/set_field_kernel.cuknl"

extern "C" void set_field_kernel_cuda_
(void)
{
    cuda_chunk.set_field_kernel();
}

void CloverleafCudaChunk::set_field_kernel
(void)
{
    CUDA_BEGIN_PROFILE;
    device_set_field_kernel_cuda<<< num_blocks, BLOCK_SZ >>>
    (x_min,x_max,y_min,y_max, density0, density1, energy0, energy1);
    CUDA_END_PROFILE;
}


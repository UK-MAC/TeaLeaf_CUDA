#include "cuda_common.hpp"
#include "kernel_files/set_field_kernel.cuknl"

extern "C" void set_field_kernel_cuda_
(void)
{
    cuda_chunk.set_field_kernel();
}

void TealeafCudaChunk::set_field_kernel
(void)
{
    CUDALAUNCH(device_set_field_kernel, energy0, energy1);
}


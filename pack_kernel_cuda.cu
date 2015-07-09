#include "cuda_common.hpp"
#include <cstdio>
#include <numeric>

#include "kernel_files/pack_kernel.cuknl"

extern "C" void cuda_pack_buffers_
(int fields[NUM_FIELDS], int offsets[NUM_FIELDS], int * depth,
 int * face, double * host_buffer)
{
    cuda_chunk.packUnpackAllBuffers(fields, offsets, *depth, *face, 1, host_buffer);
}

extern "C" void cuda_unpack_buffers_
(int fields[NUM_FIELDS], int offsets[NUM_FIELDS], int * depth,
 int * face, double * host_buffer)
{
    cuda_chunk.packUnpackAllBuffers(fields, offsets, *depth, *face, 0, host_buffer);
}

void CloverleafCudaChunk::packUnpackAllBuffers
(int fields[NUM_FIELDS], int offsets[NUM_FIELDS],
 const int depth, const int face, const int pack,
 double * host_buffer)
{
    const int n_exchanged = std::accumulate(fields, fields + NUM_FIELDS, 0);

    if (n_exchanged < 1)
    {
        return;
    }

    // which buffer is being used for this operation
    double * device_buffer = NULL;

    switch (face)
    {
    case CHUNK_LEFT:
        device_buffer = left_buffer;
        break;
    case CHUNK_RIGHT:
        device_buffer = right_buffer;
        break;
    case CHUNK_BOTTOM:
        device_buffer = bottom_buffer;
        break;
    case CHUNK_TOP:
        device_buffer = top_buffer;
        break;
    default:
        DIE("Invalid face identifier %d passed to mpi buffer packing\n", face);
    }

    pack_func_t pack_kernel = NULL;

    // set which kernel to call
    if (pack)
    {
        switch (face)
        {
        case CHUNK_LEFT:
            pack_kernel = &device_packleftBuffer;
            break;
        case CHUNK_RIGHT:
            pack_kernel = &device_packrightBuffer;
            break;
        case CHUNK_BOTTOM:
            pack_kernel = &device_packbottomBuffer;
            break;
        case CHUNK_TOP:
            pack_kernel = &device_packtopBuffer;
            break;
        default:
            DIE("Invalid face identifier %d passed to pack\n", face);
        }
    }
    else
    {
        switch (face)
        {
        case CHUNK_LEFT:
            pack_kernel = &device_unpackleftBuffer;
            break;
        case CHUNK_RIGHT:
            pack_kernel = &device_unpackrightBuffer;
            break;
        case CHUNK_BOTTOM:
            pack_kernel = &device_unpackbottomBuffer;
            break;
        case CHUNK_TOP:
            pack_kernel = &device_unpacktopBuffer;
            break;
        default:
            DIE("Invalid face identifier %d passed to unpack\n", face);
        }
    }

    // size of this buffer
    int side_size = 0;
    // actual number of elements in column/row
    int needed_launch_size = 0;
    // launch sizes for packing/unpacking arrays
    dim3 pack_global, pack_local;

    switch (face)
    {
    // pad it to fit in 32 local work group size (always on NVIDIA hardware)
    case CHUNK_LEFT:
    case CHUNK_RIGHT:
        needed_launch_size = (y_max + 5);
        pack_global = dim3(depth, needed_launch_size + (32 - (needed_launch_size % 32)));
        pack_local = dim3(1, 32);
        break;
    case CHUNK_BOTTOM:
    case CHUNK_TOP:
        needed_launch_size = (x_max + 5);
        pack_global = dim3(needed_launch_size + (32 - (needed_launch_size % 32)), depth);
        pack_local = dim3(32, 1);
        break;
    default:
        DIE("Invalid face identifier %d passed to mpi buffer packing\n", face);
    }

    side_size = sizeof(double)*needed_launch_size*depth;

    if (!pack)
    {
        cudaMemcpy(device_buffer, host_buffer, n_exchanged*depth*side_size,
            cudaMemcpyHostToDevice);

        CUDA_ERR_CHECK;
    }

    for (int ii = 0; ii < NUM_FIELDS; ii++)
    {
        int which_field = ii+1;

        if (fields[ii])
        {
            if (offsets[ii] < 0 || offsets[ii] > NUM_FIELDS*side_size)
            {
                DIE("Tried to pack/unpack field %d but invalid offset %d given\n",
                    ii, offsets[ii]);
            }

            int x_inc = 0, y_inc = 0;

            // x_inc / y_inc set to 0 in tea leaf

            double * device_array = NULL;

            #define CASE_BUF(which_array)   \
            case FIELD_##which_array:       \
            {                               \
                device_array = which_array; \
            }

            switch (which_field)
            {
            CASE_BUF(density); break;
            CASE_BUF(energy0); break;
            CASE_BUF(energy1); break;
            CASE_BUF(u); break;
            CASE_BUF(vector_p); break;
            CASE_BUF(vector_sd); break;
            default:
                DIE("Invalid face %d passed to pack buffer function\n", which_field);
            }

            #undef CASE_BUF

            pack_kernel<<< pack_global, pack_local >>>(x_min, x_max, y_min, y_max,
                x_inc, y_inc, device_array, device_buffer+offsets[ii], depth);

            CUDA_ERR_CHECK;
        }
    }

    if (pack)
    {
        cudaMemcpy(host_buffer, device_buffer, n_exchanged*depth*side_size,
            cudaMemcpyDeviceToHost);

        CUDA_ERR_CHECK;
    }
}


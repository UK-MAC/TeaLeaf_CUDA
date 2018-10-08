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

void TealeafCudaChunk::packUnpackAllBuffers
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

    kernel_info_t kernel_info;

    // set which kernel to call
    if (pack)
    {
        switch (face)
        {
        case CHUNK_LEFT:
            pack_kernel = &device_pack_left_buffer;
            kernel_info = kernel_info_map.at("device_pack_left_buffer");
            break;
        case CHUNK_RIGHT:
            pack_kernel = &device_pack_right_buffer;
            kernel_info = kernel_info_map.at("device_pack_right_buffer");
            break;
        case CHUNK_BOTTOM:
            pack_kernel = &device_pack_bottom_buffer;
            kernel_info = kernel_info_map.at("device_pack_bottom_buffer");
            break;
        case CHUNK_TOP:
            pack_kernel = &device_pack_top_buffer;
            kernel_info = kernel_info_map.at("device_pack_top_buffer");
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
            pack_kernel = &device_unpack_left_buffer;
            kernel_info = kernel_info_map.at("device_unpack_left_buffer");
            break;
        case CHUNK_RIGHT:
            pack_kernel = &device_unpack_right_buffer;
            kernel_info = kernel_info_map.at("device_unpack_right_buffer");
            break;
        case CHUNK_BOTTOM:
            pack_kernel = &device_unpack_bottom_buffer;
            kernel_info = kernel_info_map.at("device_unpack_bottom_buffer");
            break;
        case CHUNK_TOP:
            pack_kernel = &device_unpack_top_buffer;
            kernel_info = kernel_info_map.at("device_unpack_top_buffer");
            break;
        default:
            DIE("Invalid face identifier %d passed to unpack\n", face);
        }
    }

    kernel_info.x_offset = halo_exchange_depth - depth;
    kernel_info.y_offset = halo_exchange_depth - depth;

    // size of this buffer
    int side_size = 0;
    // launch sizes for packing/unpacking arrays
    dim3 pack_num_blocks, pack_block_size;

    switch (face)
    {
    // pad it to fit in 32 local work group size (always on NVIDIA hardware)
    case CHUNK_LEFT:
    case CHUNK_RIGHT:
        side_size = depth*(y_max + 2*depth);
        pack_num_blocks = update_lr_num_blocks[depth];
        pack_block_size = update_lr_block_sizes[depth];
        break;
    case CHUNK_BOTTOM:
    case CHUNK_TOP:
        side_size = depth*(x_max + 2*depth);
        pack_num_blocks = update_bt_num_blocks[depth];
        pack_block_size = update_bt_block_sizes[depth];
        break;
    default:
        DIE("Invalid face identifier %d passed to mpi buffer packing\n", face);
    }

    side_size *= sizeof(double);

    if (!pack)
    {
        cudaMemcpy(device_buffer, host_buffer, n_exchanged*side_size,
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

            #define CASE_BUF(which_array)   \
            case FIELD_##which_array:       \
            {                               \
                device_array = which_array; \
            }

            double * device_array = NULL;

            switch (which_field)
            {
            CASE_BUF(density); break;
            CASE_BUF(energy0); break;
            CASE_BUF(energy1); break;
            CASE_BUF(u); break;
            CASE_BUF(vector_p); break;
            CASE_BUF(vector_sd); break;
            CASE_BUF(vector_r); break;
            default:
                DIE("Invalid face %d passed to pack buffer function\n", which_field);
            }

            #undef CASE_BUF

            pack_kernel<<< pack_num_blocks, pack_block_size >>>(kernel_info,
                x_inc, y_inc, device_array, device_buffer, depth, offsets[ii]);

            CUDA_ERR_CHECK;
        }
    }

    if (pack)
    {
        cudaMemcpy(host_buffer, device_buffer, n_exchanged*side_size,
            cudaMemcpyDeviceToHost);

        CUDA_ERR_CHECK;
    }
}


#include "cuda_common.hpp"
#include <numeric>

#include "kernel_files/pack_kernel.cuknl"

extern "C" void cuda_pack_buffers_
(int fields[NUM_FIELDS], int offsets[NUM_FIELDS], int * depth,
 int * face, double * buffer)
{
    cuda_chunk.packUnpackAllBuffers(fields, offsets, *depth, *face, 1, buffer);
}

extern "C" void cuda_unpack_buffers_
(int fields[NUM_FIELDS], int offsets[NUM_FIELDS], int * depth,
 int * face, double * buffer)
{
    cuda_chunk.packUnpackAllBuffers(fields, offsets, *depth, *face, 0, buffer);
}

void CloverleafCudaChunk::packUnpackAllBuffers
(int fields[NUM_FIELDS], int offsets[NUM_FIELDS],
 const int depth, const int face, const int pack,
 double * buffer)
{
    const int n_exchanged = std::accumulate(fields, fields + NUM_FIELDS, 0);

    if (n_exchanged < 1)
    {
        return;
    }

    // which buffer is being used for this operation
    double * side_buffer = NULL;

    switch (face)
    {
    case CHUNK_LEFT:
        side_buffer = left_buffer;
        break;
    case CHUNK_RIGHT:
        side_buffer = right_buffer;
        break;
    case CHUNK_BOTTOM:
        side_buffer = bottom_buffer;
        break;
    case CHUNK_TOP:
        side_buffer = top_buffer;
        break;
    default:
        DIE("Invalid face identifier %d passed to mpi buffer packing\n", face);
    }

    pack_func_t * pack_kernel = NULL;

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
    size_t side_size = 0;
    // reuse the halo update kernels sizes to launch packing kernels
    dim3 pack_global, pack_local;

    switch (face)
    {
    case CHUNK_LEFT:
    case CHUNK_RIGHT:
        side_size = lr_mpi_buf_sz;
        pack_global = update_lr_global_size[depth-1];
        pack_local = update_lr_local_size[depth-1];
        break;
    case CHUNK_BOTTOM:
    case CHUNK_TOP:
        side_size = bt_mpi_buf_sz;
        pack_global = update_ud_global_size[depth-1];
        pack_local = update_ud_local_size[depth-1];
        break;
    default:
        DIE("Invalid face identifier %d passed to mpi buffer packing\n", face);
    }

    if (!pack)
    {
        // FIXME write buffer from host to device
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

            // set x/y/z inc for array
            switch (which_field)
            {
            case FIELD_density:
            case FIELD_energy0:
            case FIELD_energy1:
            case FIELD_u:
            case FIELD_p:
            case FIELD_sd:
                break;
            default:
                DIE("Invalid field number %d in choosing _inc values\n", which_field);
            }

            #define CASE_BUF(which_array)   \
            case FIELD_##which_array:       \
            {                               \
                device_array = which_array;\
            }

            double * device_array = NULL;

            switch (which_field)
            {
            CASE_BUF(density); break;
            CASE_BUF(energy0); break;
            CASE_BUF(energy1); break;
            CASE_BUF(u); break;
            CASE_BUF(work_array_1); break;
            CASE_BUF(work_array_8); break;
            default:
                DIE("Invalid face %d passed to left/right pack buffer\n", which_field);
            }

            #undef CASE_BUF

            // FIXME launch kernel
        }
    }

    if (pack)
    {
        // FIXME read buffer
    }
}


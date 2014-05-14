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
                                in_y_max,
                                in_profiler_on);
}

CloverleafCudaChunk::CloverleafCudaChunk
(void)
{
    ;
}

static std::string matchParam
(FILE * input,
 const char* param_name)
{
    std::string param_string;
    static char name_buf[101];
    rewind(input);
    /* read in line from file */
    while (NULL != fgets(name_buf, 100, input))
    {
        /* if it has the parameter name, its the line we want */
        if (NULL != strstr(name_buf, param_name))
        {
            if (NULL != strstr(name_buf, "="))
            {
                *(strstr(name_buf, "=")) = ' ';
                char param_buf[100];
                sscanf(name_buf, "%*s %s", param_buf);
                param_string = std::string(param_buf);
                break;
            }
            else
            {
                param_string = std::string("NO_SETTING");
                break;
            }
        }
    }

    return param_string;
}

int preferredDevice
(void)
{
    FILE* input;
    assert(input = fopen("tea.in", "r"));

    std::string param_string = matchParam(input, "cuda_device");

    int preferred_device;

    if (param_string.size() == 0)
    {
        // not found in file
        preferred_device = 0;
        std::cout << "CUDA device not specifiefd in file - using 0" << std::endl;
    }
    else
    {
        std::stringstream converter(param_string);

        if (!(converter >> preferred_device))
        {
            preferred_device = -1;
        }
    }

    fclose(input);

    return preferred_device;
}

CloverleafCudaChunk::CloverleafCudaChunk
(INITIALISE_ARGS)
:x_min(*in_x_min),
x_max(*in_x_max),
y_min(*in_y_min),
y_max(*in_y_max),
profiler_on(*in_profiler_on),
num_blocks((((*in_x_max)+5)*((*in_y_max)+5))/BLOCK_SZ)
{
    // FIXME (and opencl really)
    // make a better platform agnostic way of selecting devices

    // choose device 0 unless specified
    cudaThreadExit();
    int device_id = preferredDevice();
    cudaSetDevice(device_id); 

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::cout << "CUDA using " << prop.name << std::endl;

    #define CUDA_ARRAY_ALLOC(arr, size)     \
            cudaMalloc((void**) &arr, size);\
            cudaDeviceSynchronize();        \
            cudaMemset(arr, 0, size);       \
            cudaDeviceSynchronize();        \
            CUDA_ERR_CHECK;

    CUDA_ARRAY_ALLOC(volume, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(soundspeed, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(pressure, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(viscosity, BUFSZ2D(0, 0));

    CUDA_ARRAY_ALLOC(density0, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(density1, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(energy0, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(energy1, BUFSZ2D(0, 0));

    CUDA_ARRAY_ALLOC(u, BUFSZ2D(0, 0));
    CUDA_ARRAY_ALLOC(u0, BUFSZ2D(0, 0));

    CUDA_ARRAY_ALLOC(xvel0, BUFSZ2D(1, 1));
    CUDA_ARRAY_ALLOC(xvel1, BUFSZ2D(1, 1));
    CUDA_ARRAY_ALLOC(yvel0, BUFSZ2D(1, 1));
    CUDA_ARRAY_ALLOC(yvel1, BUFSZ2D(1, 1));

    CUDA_ARRAY_ALLOC(xarea, BUFSZ2D(1, 0));
    CUDA_ARRAY_ALLOC(vol_flux_x, BUFSZ2D(1, 0));
    CUDA_ARRAY_ALLOC(mass_flux_x, BUFSZ2D(1, 0));

    CUDA_ARRAY_ALLOC(yarea, BUFSZ2D(0, 1));
    CUDA_ARRAY_ALLOC(vol_flux_y, BUFSZ2D(0, 1));
    CUDA_ARRAY_ALLOC(mass_flux_y, BUFSZ2D(0, 1));

    CUDA_ARRAY_ALLOC(cellx, BUFSZX(0));
    CUDA_ARRAY_ALLOC(celldx, BUFSZX(0));
    CUDA_ARRAY_ALLOC(vertexx, BUFSZX(1));
    CUDA_ARRAY_ALLOC(vertexdx, BUFSZX(1));

    CUDA_ARRAY_ALLOC(celly, BUFSZY(0));
    CUDA_ARRAY_ALLOC(celldy, BUFSZY(0));
    CUDA_ARRAY_ALLOC(vertexy, BUFSZY(1));
    CUDA_ARRAY_ALLOC(vertexdy, BUFSZY(1));

    CUDA_ARRAY_ALLOC(work_array_1, BUFSZ2D(1, 1));
    CUDA_ARRAY_ALLOC(work_array_2, BUFSZ2D(1, 1));
    CUDA_ARRAY_ALLOC(work_array_3, BUFSZ2D(1, 1));
    CUDA_ARRAY_ALLOC(work_array_4, BUFSZ2D(1, 1));
    CUDA_ARRAY_ALLOC(work_array_5, BUFSZ2D(1, 1));
    CUDA_ARRAY_ALLOC(work_array_6, BUFSZ2D(1, 1));

    CUDA_ARRAY_ALLOC(reduce_buf_1, num_blocks*sizeof(double));
    CUDA_ARRAY_ALLOC(reduce_buf_2, num_blocks*sizeof(double));
    CUDA_ARRAY_ALLOC(reduce_buf_3, num_blocks*sizeof(double));
    CUDA_ARRAY_ALLOC(reduce_buf_4, num_blocks*sizeof(double));
    CUDA_ARRAY_ALLOC(reduce_buf_5, num_blocks*sizeof(double));
    CUDA_ARRAY_ALLOC(reduce_buf_6, num_blocks*sizeof(double));

    reduce_ptr_1 = thrust::device_ptr< double >(reduce_buf_1);
    reduce_ptr_2 = thrust::device_ptr< double >(reduce_buf_2);
    reduce_ptr_3 = thrust::device_ptr< double >(reduce_buf_3);
    reduce_ptr_4 = thrust::device_ptr< double >(reduce_buf_4);
    reduce_ptr_5 = thrust::device_ptr< double >(reduce_buf_5);
    reduce_ptr_6 = thrust::device_ptr< double >(reduce_buf_6);

    CUDA_ARRAY_ALLOC(pdv_reduce_array, num_blocks*sizeof(int));
    reduce_pdv = thrust::device_ptr< int >(pdv_reduce_array);

    thr_cellx = thrust::device_ptr< double >(cellx);
    thr_celly = thrust::device_ptr< double >(celly);
    thr_xvel0 = thrust::device_ptr< double >(xvel0);
    thr_yvel0 = thrust::device_ptr< double >(yvel0);
    thr_xvel1 = thrust::device_ptr< double >(xvel1);
    thr_yvel1 = thrust::device_ptr< double >(yvel1);
    thr_density0 = thrust::device_ptr< double >(density0);
    thr_energy0 = thrust::device_ptr< double >(energy0);
    thr_pressure = thrust::device_ptr< double >(pressure);
    thr_soundspeed = thrust::device_ptr< double >(soundspeed);

    CUDA_ARRAY_ALLOC(dev_left_send_buffer, sizeof(double)*(y_max+5)*2);
    CUDA_ARRAY_ALLOC(dev_right_send_buffer, sizeof(double)*(y_max+5)*2);
    CUDA_ARRAY_ALLOC(dev_top_send_buffer, sizeof(double)*(x_max+5)*2);
    CUDA_ARRAY_ALLOC(dev_bottom_send_buffer, sizeof(double)*(x_max+5)*2);

    CUDA_ARRAY_ALLOC(dev_left_recv_buffer, sizeof(double)*(y_max+5)*2);
    CUDA_ARRAY_ALLOC(dev_right_recv_buffer, sizeof(double)*(y_max+5)*2);
    CUDA_ARRAY_ALLOC(dev_top_recv_buffer, sizeof(double)*(x_max+5)*2);
    CUDA_ARRAY_ALLOC(dev_bottom_recv_buffer, sizeof(double)*(x_max+5)*2);

    #undef CUDA_ARRAY_ALLOC

#define ADD_BUFFER_DBG_MAP(name) arr_names[#name] = name;
    ADD_BUFFER_DBG_MAP(volume);
    ADD_BUFFER_DBG_MAP(soundspeed);
    ADD_BUFFER_DBG_MAP(pressure);
    ADD_BUFFER_DBG_MAP(viscosity);

    ADD_BUFFER_DBG_MAP(u);
    arr_names["p"] = work_array_1;

    ADD_BUFFER_DBG_MAP(density0);
    ADD_BUFFER_DBG_MAP(density1);
    ADD_BUFFER_DBG_MAP(energy0);
    ADD_BUFFER_DBG_MAP(energy1);
    ADD_BUFFER_DBG_MAP(xvel0);
    ADD_BUFFER_DBG_MAP(xvel1);
    ADD_BUFFER_DBG_MAP(yvel0);
    ADD_BUFFER_DBG_MAP(yvel1);
    ADD_BUFFER_DBG_MAP(xarea);
    ADD_BUFFER_DBG_MAP(yarea);
    ADD_BUFFER_DBG_MAP(vol_flux_x);
    ADD_BUFFER_DBG_MAP(vol_flux_y);
    ADD_BUFFER_DBG_MAP(mass_flux_x);
    ADD_BUFFER_DBG_MAP(mass_flux_y);

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


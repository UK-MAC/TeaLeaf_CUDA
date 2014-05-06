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
 *  @brief CUDA common file
 *  @author Michael Boulton NVIDIA Corporation
 *  @details Contains common elements for cuda kernels
 */

#ifndef __CUDA_COMMON_INC
#define __CUDA_COMMON_INC

#include "thrust/device_allocator.h"
#include "thrust/extrema.h"
#include "kernel_files/cuda_kernel_header.hpp"

// used in update_halo and for copying back to host for mpi transfers
#define FIELD_density0      1
#define FIELD_density1      2
#define FIELD_energy0       3
#define FIELD_energy1       4
#define FIELD_pressure      5
#define FIELD_viscosity     6
#define FIELD_soundspeed    7
#define FIELD_xvel0         8
#define FIELD_xvel1         9
#define FIELD_yvel0         10
#define FIELD_yvel1         11
#define FIELD_vol_flux_x    12
#define FIELD_vol_flux_y    13
#define FIELD_mass_flux_x   14
#define FIELD_mass_flux_y   15
#define FIELD_u             16
#define FIELD_p             17
#define NUM_FIELDS          17
#define FIELD_work_array_1 FIELD_p

// which side to pack - keep the same as in fortran file
#define CHUNK_LEFT 1
#define CHUNK_left 1
#define CHUNK_RIGHT 2
#define CHUNK_right 2
#define CHUNK_BOTTOM 3
#define CHUNK_bottom 3
#define CHUNK_TOP 4
#define CHUNK_top 4
#define EXTERNAL_FACE       (-1)

#define CELL_DATA   1
#define VERTEX_DATA 2
#define X_FACE_DATA 3
#define Y_FACE_DATA 4

#define INITIALISE_ARGS \
    /* values used to control operation */\
    int* in_x_min, \
    int* in_x_max, \
    int* in_y_min, \
    int* in_y_max, \
    bool* in_profiler_on

/*******************/

// disable checking for errors after kernel calls / memory allocation
#ifdef NO_ERR_CHK

// do nothing instead
#define CUDA_ERR_CHECK ;

#else

#include <iostream>

#define CUDA_ERR_CHECK errorHandler(__LINE__, __FILE__);

static const char* errorCodes
(int err_code)
{
    switch(err_code)
    {
        case cudaSuccess: return "cudaSuccess"; // 0
        case cudaErrorMissingConfiguration: return "cudaErrorMissingConfiguration"; // 1
        case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation"; // 2
        case cudaErrorInitializationError: return "cudaErrorInitializationError"; // 3
        case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure"; // 4
        case cudaErrorPriorLaunchFailure: return "cudaErrorPriorLaunchFailure"; // 5
        case cudaErrorLaunchTimeout: return "cudaErrorLaunchTimeout"; // 6
        case cudaErrorLaunchOutOfResources: return "cudaErrorLaunchOutOfResources"; // 7
        case cudaErrorInvalidDeviceFunction: return "cudaErrorInvalidDeviceFunction"; // 8
        case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration"; // 9
        case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice"; // 10
        case cudaErrorInvalidValue: return "cudaErrorInvalidValue";// 11
        case cudaErrorInvalidPitchValue: return "cudaErrorInvalidPitchValue";// 12
        case cudaErrorInvalidSymbol: return "cudaErrorInvalidSymbol";// 13
        case cudaErrorMapBufferObjectFailed: return "cudaErrorMapBufferObjectFailed";// 14
        case cudaErrorUnmapBufferObjectFailed: return "cudaErrorUnmapBufferObjectFailed";// 15
        case cudaErrorInvalidHostPointer: return "cudaErrorInvalidHostPointer";// 16
        case cudaErrorInvalidDevicePointer: return "cudaErrorInvalidDevicePointer";// 17
        case cudaErrorInvalidTexture: return "cudaErrorInvalidTexture";// 18
        case cudaErrorInvalidTextureBinding: return "cudaErrorInvalidTextureBinding";// 19
        case cudaErrorInvalidChannelDescriptor: return "cudaErrorInvalidChannelDescriptor";// 20
        case cudaErrorInvalidMemcpyDirection: return "cudaErrorInvalidMemcpyDirection";// 21
        case cudaErrorAddressOfConstant: return "cudaErrorAddressOfConstant";// 22
        case cudaErrorTextureFetchFailed: return "cudaErrorTextureFetchFailed";// 23
        case cudaErrorTextureNotBound: return "cudaErrorTextureNotBound";// 24
        case cudaErrorSynchronizationError: return "cudaErrorSynchronizationError";// 25
        case cudaErrorInvalidFilterSetting: return "cudaErrorInvalidFilterSetting";// 26
        case cudaErrorInvalidNormSetting: return "cudaErrorInvalidNormSetting";// 27
        case cudaErrorMixedDeviceExecution: return "cudaErrorMixedDeviceExecution";// 28
        case cudaErrorCudartUnloading: return "cudaErrorCudartUnloading";// 29
        case cudaErrorUnknown: return "cudaErrorUnknown";// 30
        case cudaErrorNotYetImplemented: return "cudaErrorNotYetImplemented";// 31
        case cudaErrorMemoryValueTooLarge: return "cudaErrorMemoryValueTooLarge";// 32
        case cudaErrorInvalidResourceHandle: return "cudaErrorInvalidResourceHandle";// 33
        case cudaErrorNotReady: return "cudaErrorNotReady";// 34
        case cudaErrorInsufficientDriver: return "cudaErrorInsufficientDriver";// 35
        case cudaErrorSetOnActiveProcess: return "cudaErrorSetOnActiveProcess";// 36
        case cudaErrorInvalidSurface: return "cudaErrorInvalidSurface";// 37
        case cudaErrorNoDevice: return "cudaErrorNoDevice";// 38
        case cudaErrorECCUncorrectable: return "cudaErrorECCUncorrectable";// 39
        case cudaErrorSharedObjectSymbolNotFound: return "cudaErrorSharedObjectSymbolNotFound";// 40
        case cudaErrorSharedObjectInitFailed: return "cudaErrorSharedObjectInitFailed";// 41
        case cudaErrorUnsupportedLimit: return "cudaErrorUnsupportedLimit";// 42
        case cudaErrorDuplicateVariableName: return "cudaErrorDuplicateVariableName";// 43
        case cudaErrorDuplicateTextureName: return "cudaErrorDuplicateTextureName";// 44
        case cudaErrorDuplicateSurfaceName: return "cudaErrorDuplicateSurfaceName";// 45
        case cudaErrorDevicesUnavailable: return "cudaErrorDevicesUnavailable";// 46
        case cudaErrorInvalidKernelImage: return "cudaErrorInvalidKernelImage";// 47
        case cudaErrorNoKernelImageForDevice: return "cudaErrorNoKernelImageForDevice";// 48
        case cudaErrorIncompatibleDriverContext: return "cudaErrorIncompatibleDriverContext";// 49
        case cudaErrorPeerAccessAlreadyEnabled: return "cudaErrorPeerAccessAlreadyEnabled";// 50
        case cudaErrorPeerAccessNotEnabled: return "cudaErrorPeerAccessNotEnabled";// 51
        case cudaErrorDeviceAlreadyInUse: return "cudaErrorDeviceAlreadyInUse";// 52
        case cudaErrorProfilerDisabled: return "cudaErrorProfilerDisabled";// 53
        case cudaErrorProfilerNotInitialized: return "cudaErrorProfilerNotInitialized";// 54
        case cudaErrorProfilerAlreadyStarted: return "cudaErrorProfilerAlreadyStarted";// 55
        case cudaErrorProfilerAlreadyStopped: return "cudaErrorProfilerAlreadyStopped";// 56
        case cudaErrorAssert: return "cudaErrorAssert";// 57
        case cudaErrorTooManyPeers: return "cudaErrorTooManyPeers";// 58
        case cudaErrorHostMemoryAlreadyRegistered: return "cudaErrorHostMemoryAlreadyRegistered";// 59
        case cudaErrorHostMemoryNotRegistered: return "cudaErrorHostMemoryNotRegistered";// 60
        case cudaErrorOperatingSystem: return "cudaErrorOperatingSystem";// 61
        case cudaErrorStartupFailure: return "cudaErrorStartupFailure";// 62
        case cudaErrorApiFailureBase: return "cudaErrorApiFailureBase";// 63
        default: return "Unknown error";
    }
}

inline void errorHandler
(int line_num, std::string const& file)
{
    cudaDeviceSynchronize();
    int l_e = cudaGetLastError();
    if (cudaSuccess != l_e)
    {
        std::cout << "error on line " << line_num << " of ";
        std::cout << file << std::endl;
        std::cout << "return code " << l_e; 
        std::cout << " (" << errorCodes(l_e) << ")";
        std::cout << std::endl;
        exit(l_e);
    }
}

#endif //NO_ERR_CHK

// whether to time kernel run times
#ifdef TIME_KERNELS

// use same timer as fortran/c
extern "C" void timer_c_(double*);

// beginning of profiling bit
#define CUDA_BEGIN_PROFILE \
    static float time; \
    static cudaEvent_t __t_0, __t_1;          \
    cudaEventCreate(&__t_0); \
    cudaEventRecord(&__t_0);

// end of profiling bit
#define CUDA_END_PROFILE \
    cudaDeviceSynchronize();                        \
    timer_c_(&__t_1); \
    std::cout << "[PROFILING] " << __t_1 - __t_0  \
    << " to calculate " << __FILE__  << std::endl;

#else

#define CUDA_BEGIN_PROFILE ;
#define CUDA_END_PROFILE if (profiler_on) cudaDeviceSynchronize();

#endif // TIME_KERNELS

typedef struct cell_info {
    const int x_e;
    const int y_e;
    const int x_i;
    const int y_i;
    const int x_f;
    const int y_f;
    const int grid_type;

    cell_info
    (int x_extra, int y_extra,
    int x_invert, int y_invert,
    int x_face, int y_face,
    int in_type)
    :x_e(x_extra), y_e(y_extra),
    x_i(x_invert), y_i(y_invert),
    x_f(x_face), y_f(y_face),
    grid_type(in_type)
    {
        ;
    }

} cell_info_t;

// types of array data
const static cell_info_t CELL(    0, 0,  1,  1, 0, 0, CELL_DATA);
const static cell_info_t VERTEX_X(1, 1, -1,  1, 0, 0, VERTEX_DATA);
const static cell_info_t VERTEX_Y(1, 1,  1, -1, 0, 0, VERTEX_DATA);
const static cell_info_t X_FACE(  1, 0, -1,  1, 1, 0, X_FACE_DATA);
const static cell_info_t Y_FACE(  0, 1,  1, -1, 0, 1, Y_FACE_DATA);

class CloverleafCudaChunk
{
private:
    // work arrays
    double* volume;
    double* soundspeed;
    double* pressure;
    double* viscosity;

    double* density0;
    double* density1;
    double* energy0;
    double* energy1;
    double* xvel0;
    double* xvel1;
    double* yvel0;
    double* yvel1;
    double* xarea;
    double* yarea;
    double* vol_flux_x;
    double* vol_flux_y;
    double* mass_flux_x;
    double* mass_flux_y;

    double* cellx;
    double* celly;
    double* celldx;
    double* celldy;
    double* vertexx;
    double* vertexy;
    double* vertexdx;
    double* vertexdy;

    // used in calc_dt to retrieve values
    thrust::device_ptr< double > thr_cellx;
    thrust::device_ptr< double > thr_celly;
    thrust::device_ptr< double > thr_xvel0;
    thrust::device_ptr< double > thr_yvel0;
    thrust::device_ptr< double > thr_xvel1;
    thrust::device_ptr< double > thr_yvel1;
    thrust::device_ptr< double > thr_density0;
    thrust::device_ptr< double > thr_energy0;
    thrust::device_ptr< double > thr_pressure;
    thrust::device_ptr< double > thr_soundspeed;

    // holding temporary stuff like post_vol etc.
    double* work_array_1;
    double* work_array_2;
    double* work_array_3;
    double* work_array_4;
    double* work_array_5;
    double* work_array_6;

    // buffers used in mpi transfers
    double* dev_left_send_buffer;
    double* dev_right_send_buffer;
    double* dev_top_send_buffer;
    double* dev_bottom_send_buffer;
    double* dev_left_recv_buffer;
    double* dev_right_recv_buffer;
    double* dev_top_recv_buffer;
    double* dev_bottom_recv_buffer;

    // used for reductions in calc dt, pdv, field summary
    thrust::device_ptr< double > reduce_ptr_1;
    thrust::device_ptr< double > reduce_ptr_2;
    thrust::device_ptr< double > reduce_ptr_3;
    thrust::device_ptr< double > reduce_ptr_4;
    thrust::device_ptr< double > reduce_ptr_5;
    thrust::device_ptr< double > reduce_ptr_6;

    // number of blocks for work space
    unsigned int num_blocks;

    //as above, but for pdv kernel only
    int* pdv_reduce_array;
    thrust::device_ptr< int > reduce_pdv;

    // values used to control operation
    int x_min;
    int x_max;
    int y_min;
    int y_max;

    // if being profiled
    bool profiler_on;

    // mpi packing
    #define PACK_ARGS                                       \
        int chunk_1, int chunk_2, int external_face,        \
        int x_inc, int y_inc, int depth, int which_field,   \
        double *buffer_1, double *buffer_2
    int getBufferSize
    (int edge, int depth, int x_inc, int y_inc);
public:
    // kernels
    void calc_dt_kernel(double g_small, double g_big, double dtmin,
        double dtc_safe, double dtu_safe, double dtv_safe,
        double dtdiv_safe, double* dt_min_val, int* dtl_control,
        double* xl_pos, double* yl_pos, int* jldt, int* kldt, int* small);

    void field_summary_kernel(double* vol, double* mass,
        double* ie, double* ke, double* press, double* temp);

    void PdV_kernel(int* error_condition, int predict, double dbyt);

    void ideal_gas_kernel(int predict);

    void generate_chunk_kernel(const int number_of_states, 
        const double* state_density, const double* state_energy,
        const double* state_xvel, const double* state_yvel,
        const double* state_xmin, const double* state_xmax,
        const double* state_ymin, const double* state_ymax,
        const double* state_radius, const int* state_geometry,
        const int g_rect, const int g_circ, const int g_point);

    void initialise_chunk_kernel(double d_xmin, double d_ymin,
        double d_dx, double d_dy);

    void update_halo_kernel(const int* fields, int depth,
        const int* chunk_neighbours);

    void accelerate_kernel(double dbyt);

    void advec_mom_kernel(int which_vel, int sweep_number, int direction);

    void flux_calc_kernel(double dbyt);

    void advec_cell_kernel(int dr, int swp_nmbr);

    void revert_kernel();

    void set_field_kernel();
    void reset_field_kernel();

    void viscosity_kernel();

    // Tea leaf
    void tea_leaf_init_jacobi(int, double, double*, double*);
    void tea_leaf_kernel_jacobi(double, double, double*);

    void tea_leaf_init_cg(int, double, double*, double*, double*);
    void tea_leaf_kernel_cg_calc_w(double rx, double ry, double* pw);
    void tea_leaf_kernel_cg_calc_ur(double alpha, double* rrn);
    void tea_leaf_kernel_cg_calc_p(double beta);

    void tea_leaf_cheby_copy_u
    (double* rro);
    void tea_leaf_calc_2norm_kernel
    (int norm_array, double* norm);
    void tea_leaf_kernel_cheby_init
    (const double * ch_alphas, const double * ch_betas, int n_coefs,
     const double rx, const double ry, const double theta, double* error);
    void tea_leaf_kernel_cheby_iterate
    (const double * ch_alphas, const double * ch_betas, int n_coefs,
     const double rx, const double ry, const int cheby_calc_steps);

    void tea_leaf_finalise();

    void pack_left_right(PACK_ARGS);
    void unpack_left_right(PACK_ARGS);
    void pack_top_bottom(PACK_ARGS);
    void unpack_top_bottom(PACK_ARGS);

    CloverleafCudaChunk
    (INITIALISE_ARGS);

    CloverleafCudaChunk
    (void);
};

extern CloverleafCudaChunk chunk;

#endif


#include "cuda_common.hpp"

// same as in fortran
#define COEF_CONDUCTIVITY 1
#define COEF_RECIP_CONDUCTIVITY 2

#include "kernel_files/tea_block_jacobi.cuknl"
#include "kernel_files/tea_leaf_common.cuknl"
#include "kernel_files/tea_leaf_jacobi.cuknl"
#include "kernel_files/tea_leaf_cg.cuknl"
#include "kernel_files/tea_leaf_cheby.cuknl"
#include "kernel_files/tea_leaf_ppcg.cuknl"
#include "host_reductions_kernel_cuda.hpp"

#include <cassert>

// copy back dx/dy and calculate rx/ry
void TealeafCudaChunk::calcrxry
(double dt, double * rx, double * ry)
{
    double dx, dy;

    cudaMemcpy(&dx, halo_exchange_depth + celldx, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&dy, halo_exchange_depth + celldy, sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    CUDA_ERR_CHECK;

    *rx = dt/(dx*dx);
    *ry = dt/(dy*dy);
}

extern "C" void tea_leaf_calc_2norm_kernel_cuda_
(int* norm_array, double* norm)
{
    cuda_chunk.tea_leaf_calc_2norm_kernel(*norm_array, norm);
}

/********************/

extern "C" void tea_leaf_cheby_init_kernel_cuda_
(const double * ch_alphas, const double * ch_betas, int* n_coefs,
 const double * theta)
{
    cuda_chunk.tea_leaf_kernel_cheby_init(ch_alphas, ch_betas, *n_coefs,
        *theta);
}

extern "C" void tea_leaf_cheby_iterate_kernel_cuda_
(const int * cheby_calc_step)
{
    cuda_chunk.tea_leaf_kernel_cheby_iterate(*cheby_calc_step);
}

void TealeafCudaChunk::tea_leaf_calc_2norm_kernel
(int norm_array, double* norm)
{
    if (norm_array == 0)
    {
        // norm of u0
        CUDALAUNCH(device_tea_leaf_calc_2norm, u0, u0, reduce_buf_1);
    }
    else if (norm_array == 1)
    {
        // norm of r
        CUDALAUNCH(device_tea_leaf_calc_2norm, vector_r, vector_r, reduce_buf_1);
    }
    else if (norm_array == 2)
    {
        if (preconditioner_type != TL_PREC_NONE)
        {
            CUDALAUNCH(device_tea_leaf_calc_2norm, vector_r, vector_z, reduce_buf_1);
        }
        else
        {
            CUDALAUNCH(device_tea_leaf_calc_2norm, vector_r, vector_r, reduce_buf_1);
        }
    }
    else
    {
        DIE("Invalid value '%d' for norm_array passed, should be [0, 1, 2]", norm_array);
    }

    CUDA_ERR_CHECK;
    ReduceToHost<double>::sum(reduce_buf_1, norm, num_blocks);
}

void TealeafCudaChunk::upload_ch_coefs
(const double * ch_alphas, const double * ch_betas,
 const int n_coefs)
{
    size_t ch_buf_sz = n_coefs*sizeof(double);

    if (ch_alphas_device == NULL && ch_betas_device == NULL)
    {
        cudaMalloc((void**) &ch_alphas_device, ch_buf_sz);
        cudaMalloc((void**) &ch_betas_device, ch_buf_sz);
    }

    // upload to device
    cudaMemcpy(ch_alphas_device, ch_alphas, ch_buf_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(ch_betas_device, ch_betas, ch_buf_sz, cudaMemcpyHostToDevice);
}

void TealeafCudaChunk::tea_leaf_kernel_cheby_init
(const double * ch_alphas, const double * ch_betas, int n_coefs,
 const double theta)
{
    assert(tea_solver == TEA_ENUM_CHEBYSHEV);

    upload_ch_coefs(ch_alphas, ch_betas, n_coefs);

    CUDA_ERR_CHECK;

    CUDALAUNCH(device_tea_leaf_cheby_solve_init_p, u, u0,
        vector_p, vector_r, vector_w, tri_cp, tri_bfp,
        vector_Mi, vector_Kx, vector_Ky,
        theta);

    // update p
    CUDALAUNCH(device_tea_leaf_cheby_solve_calc_u, u, vector_p);
}

void TealeafCudaChunk::tea_leaf_kernel_cheby_iterate
(const int cheby_calc_step)
{
    CUDALAUNCH(device_tea_leaf_cheby_solve_calc_p, u, u0,
        vector_p, vector_r, vector_w, tri_cp, tri_bfp,
        vector_Mi, vector_Kx, vector_Ky,
        ch_alphas_device, ch_betas_device,
        cheby_calc_step-1);

    CUDALAUNCH(device_tea_leaf_cheby_solve_calc_u, u, vector_p);
}

/********************/

// CG solver functions
extern "C" void tea_leaf_cg_init_kernel_cuda_
(double * rro)
{
    cuda_chunk.tea_leaf_init_cg(rro);
}

extern "C" void tea_leaf_cg_calc_w_kernel_cuda_
(double * pw)
{
    cuda_chunk.tea_leaf_kernel_cg_calc_w(pw);
}
extern "C" void tea_leaf_cg_calc_ur_kernel_cuda_
(double * alpha, double * rrn)
{
    cuda_chunk.tea_leaf_kernel_cg_calc_ur(*alpha, rrn);
}
extern "C" void tea_leaf_cg_calc_p_kernel_cuda_
(double * beta)
{
    cuda_chunk.tea_leaf_kernel_cg_calc_p(*beta);
}

/********************/

void TealeafCudaChunk::tea_leaf_init_cg
(double * rro)
{
    assert(tea_solver == TEA_ENUM_CG || tea_solver == TEA_ENUM_CHEBYSHEV || tea_solver == TEA_ENUM_PPCG);

    if (preconditioner_type == TL_PREC_JAC_BLOCK)
    {
        CUDALAUNCH(device_tea_leaf_block_init, vector_r,
            vector_z, tri_cp, tri_bfp, vector_Kx, vector_Ky);
        CUDALAUNCH(device_tea_leaf_block_solve, vector_r,
            vector_z, tri_cp, tri_bfp, vector_Kx, vector_Ky);
    }
    else if (preconditioner_type == TL_PREC_JAC_DIAG)
    {
        CUDALAUNCH(device_tea_leaf_init_jac_diag, vector_Mi, vector_Kx, vector_Ky);
    }

    // init Kx, Ky
    CUDALAUNCH(device_tea_leaf_cg_solve_init_p, vector_p, vector_r,
        vector_z, vector_Mi, reduce_buf_2);

    ReduceToHost<double>::sum(reduce_buf_2, rro, num_blocks);
}

void TealeafCudaChunk::tea_leaf_kernel_cg_calc_w
(double* pw)
{
    CUDALAUNCH(device_tea_leaf_cg_solve_calc_w, reduce_buf_3,
        vector_p, vector_w, vector_Kx, vector_Ky);

    ReduceToHost<double>::sum(reduce_buf_3, pw, num_blocks);
}

void TealeafCudaChunk::tea_leaf_kernel_cg_calc_ur
(double alpha, double* rrn)
{
    CUDALAUNCH(device_tea_leaf_cg_solve_calc_ur, alpha, u, vector_p,
        vector_r, vector_w, vector_z, tri_cp, tri_bfp,
        vector_Mi, vector_Kx, vector_Ky, reduce_buf_4);

    ReduceToHost<double>::sum(reduce_buf_4, rrn, num_blocks);
}

void TealeafCudaChunk::tea_leaf_kernel_cg_calc_p
(double beta)
{
    CUDALAUNCH(device_tea_leaf_cg_solve_calc_p, beta, vector_p, vector_r, vector_z);
}

/********************/

extern "C" void tea_leaf_jacobi_solve_kernel_cuda_
(double * error)
{
    cuda_chunk.tea_leaf_kernel_jacobi(error);
}

void TealeafCudaChunk::tea_leaf_kernel_jacobi
(double* error)
{
    CUDALAUNCH(device_tea_leaf_jacobi_copy_u, u, vector_Mi);

    CUDALAUNCH(device_tea_leaf_jacobi_solve, vector_Kx, vector_Ky,
        u0, u, vector_Mi, reduce_buf_1);
    ReduceToHost<double>::sum(reduce_buf_1, error, num_blocks);
}

/********************/

extern "C" void tea_leaf_common_init_kernel_cuda_
(const int * coefficient, double * dt, double * rx, double * ry,
 int * zero_boundary, int * reflective_boundary)
{
    cuda_chunk.tea_leaf_common_init(*coefficient, *dt, rx, ry,
        zero_boundary, *reflective_boundary);
}

// used by both
extern "C" void tea_leaf_common_finalise_kernel_cuda_
(void)
{
    cuda_chunk.tea_leaf_finalise();
}

extern "C" void tea_leaf_calc_residual_cuda_
(void)
{
    cuda_chunk.tea_leaf_calc_residual();
}

void TealeafCudaChunk::tea_leaf_common_init
(int coefficient, double dt, double * rx, double * ry,
 int * zero_boundary, int reflective_boundary)
{
    if (coefficient != COEF_CONDUCTIVITY && coefficient != COEF_RECIP_CONDUCTIVITY)
    {
        DIE("Unknown coefficient %d passed to tea leaf\n", coefficient);
    }

    calcrxry(dt, rx, ry);

    CUDALAUNCH(device_tea_leaf_init_common, density, energy1,
        vector_Kx, vector_Ky, *rx, *ry, coefficient);

    if (!reflective_boundary)
    {
        int zero_left = zero_boundary[0];
        int zero_right = zero_boundary[1];
        int zero_bottom = zero_boundary[2];
        int zero_top = zero_boundary[3];

        CUDALAUNCH(device_tea_leaf_zero_boundaries, vector_Kx, vector_Ky,
            zero_left,
            zero_right,
            zero_bottom,
            zero_top);
    }

    generate_chunk_init_u(energy1);
}

// both
void TealeafCudaChunk::tea_leaf_finalise
(void)
{
    CUDALAUNCH(device_tea_leaf_finalise, density, u, energy1);
}

void TealeafCudaChunk::tea_leaf_calc_residual
(void)
{
    CUDALAUNCH(device_tea_leaf_calc_residual, u, u0, vector_r,
        vector_Kx, vector_Ky);
}

/********PPCG stuff********/

extern "C" void tea_leaf_ppcg_init_constants_cuda_
(const double * ch_alphas, const double * ch_betas,
 int* n_inner_steps)
{
    cuda_chunk.ppcg_init_constants(ch_alphas, ch_betas, *n_inner_steps);
}

/* A new initialisation routine */
extern "C" void tea_leaf_ppcg_init_kernel_cuda_
(const int * step, double * rro)
{
    cuda_chunk.ppcg_init(*step, rro);
}

extern "C" void tea_leaf_ppcg_init_sd_kernel_cuda_
(const double * theta)
{
    cuda_chunk.ppcg_init_sd(*theta);
}

/* Update to init_sd */
extern "C" void tea_leaf_ppcg_init_sd_new_kernel_cuda_
(const double * theta)
{
    cuda_chunk.ppcg_init_sd_new(*theta);
}


/* New store the residual for later use */ 
extern "C" void tea_leaf_ppcg_store_r_kernel_cuda_()
{
    cuda_chunk.ppcg_store_r();
}

/* New update_z */ 
extern "C" void tea_leaf_ppcg_update_z_kernel_cuda_()
{
    cuda_chunk.ppcg_update_z();
}

/* New calculate norm of (r-rstore)*z */
extern "C" void tea_leaf_ppcg_calc_rrn_kernel_cuda_(double* norm)
{
    cuda_chunk.tea_leaf_ppcg_calc_rrn_kernel(norm);
}


/* Main inner loop */
extern "C" void tea_leaf_ppcg_inner_kernel_cuda_
(int * ppcg_cur_step, int * bounds_extra,
 int * chunk_neighbours)
{
    cuda_chunk.ppcg_inner(*ppcg_cur_step, *bounds_extra, chunk_neighbours);
}


void TealeafCudaChunk::ppcg_init_constants
(const double * ch_alphas, const double * ch_betas,
 const int n_inner_steps)
{
    upload_ch_coefs(ch_alphas, ch_betas, n_inner_steps);
}


/* New initialisation */
void TealeafCudaChunk::ppcg_init
(const int step, double * rro)
{
    CUDALAUNCH(device_tea_leaf_ppcg_solve_init,
        vector_p, vector_r, vector_sd, vector_z, tri_cp, tri_bfp,
        vector_Mi, vector_Kx, vector_Ky, step, reduce_buf_1);

    CUDA_ERR_CHECK;
    ReduceToHost<double>::sum(reduce_buf_1, rro, num_blocks);        
}

void TealeafCudaChunk::ppcg_init_sd
(double theta)
{
    CUDALAUNCH(device_tea_leaf_ppcg_solve_init_sd,
        vector_r, vector_sd, vector_z, tri_cp, tri_bfp,
        vector_Mi, vector_Kx, vector_Ky, theta);
}

/* New update to init_sd */
void TealeafCudaChunk::ppcg_init_sd_new
(double theta)
{
    CUDALAUNCH(device_tea_leaf_ppcg_solve_init_sd_new,
        vector_r, vector_sd, vector_z, vector_rtemp, vector_utemp, theta);
}


/* New ppcg_store_r */ 
void TealeafCudaChunk::ppcg_store_r()
{
        CUDALAUNCH(device_tea_leaf_ppcg_store_r,
        vector_r, vector_r_store);
}

/* New ppcg_update_z */ 
void TealeafCudaChunk::ppcg_update_z()
{
        CUDALAUNCH(device_tea_leaf_ppcg_update_z, vector_z, vector_utemp);
}

/* New ppcg_calc_rrn */ 
void TealeafCudaChunk::tea_leaf_ppcg_calc_rrn_kernel(double* norm)
{

    // norm of (r-rstore)*z
    CUDALAUNCH(device_tea_leaf_calc_rrn, vector_r_store, vector_r, vector_z, reduce_buf_1);

    CUDA_ERR_CHECK;
    ReduceToHost<double>::sum(reduce_buf_1, norm, num_blocks);
}

void TealeafCudaChunk::ppcg_inner
(int ppcg_cur_step, int bounds_extra,
 int * chunk_neighbours)
{
    int step_depth = halo_exchange_depth - bounds_extra;

    int step_offset[2] = {step_depth, step_depth};
    int step_global_size[2] = {
        x_max + (halo_exchange_depth-step_depth)*2,
        y_max + (halo_exchange_depth-step_depth)*2};

    kernel_info_t kernel_info = kernel_info_map.at("device_tea_leaf_ppcg_solve_update_r");

    kernel_info.kernel_x_max = bounds_extra;
    kernel_info.kernel_y_max = bounds_extra;

    if (chunk_neighbours[CHUNK_LEFT - 1] == EXTERNAL_FACE)
    {
        step_offset[0] = halo_exchange_depth;
        step_global_size[0] -= (halo_exchange_depth-step_depth);
    }
    if (chunk_neighbours[CHUNK_RIGHT - 1] == EXTERNAL_FACE)
    {
        step_global_size[0] -= (halo_exchange_depth-step_depth);
        kernel_info.kernel_x_max = 0;
    }

    if (chunk_neighbours[CHUNK_BOTTOM - 1] == EXTERNAL_FACE)
    {
        step_offset[1] = halo_exchange_depth;
        step_global_size[1] -= (halo_exchange_depth-step_depth);
    }
    if (chunk_neighbours[CHUNK_TOP - 1] == EXTERNAL_FACE)
    {
        step_global_size[1] -= (halo_exchange_depth-step_depth);
        kernel_info.kernel_y_max = 0;
    }

    kernel_info.x_offset = step_offset[0];
    kernel_info.y_offset = step_offset[1];

    step_global_size[0] -= step_global_size[0] % LOCAL_X;
    step_global_size[0] += LOCAL_X;
    step_global_size[1] -= step_global_size[1] % LOCAL_Y;
    step_global_size[1] += LOCAL_Y;

    dim3 matrix_power_grid_dim = dim3(
        step_global_size[0]/LOCAL_X,
        step_global_size[1]/LOCAL_Y);

    TIME_KERNEL_BEGIN;
    device_tea_leaf_ppcg_solve_update_r
    <<<matrix_power_grid_dim, block_shape>>>
    (kernel_info, vector_rtemp, vector_Kx, vector_Ky, vector_sd);
    CUDA_ERR_CHECK;
    TIME_KERNEL_END(device_tea_leaf_ppcg_solve_update_r);
        
    TIME_KERNEL_BEGIN;
    device_tea_leaf_ppcg_solve_calc_sd_new
    <<<matrix_power_grid_dim, block_shape>>>
    (kernel_info,
        vector_r, vector_sd, vector_z, vector_rtemp, vector_utemp,
        tri_cp, tri_bfp,
        vector_Mi, vector_Kx, vector_Ky,
        ch_alphas_device, ch_betas_device, ppcg_cur_step - 1);
    CUDA_ERR_CHECK;        
    TIME_KERNEL_END(device_tea_leaf_ppcg_solve_calc_sd_new);
    
}

extern "C" void tea_leaf_ppcg_calc_2norm_kernel_cuda_
(double* norm)
{
    cuda_chunk.tea_leaf_ppcg_calc_2norm_kernel(norm);
}


void TealeafCudaChunk::tea_leaf_ppcg_calc_2norm_kernel
(double* norm)
{
    CUDALAUNCH(device_tea_leaf_calc_2norm, vector_r, vector_z, reduce_buf_1);
    CUDA_ERR_CHECK;
    ReduceToHost<double>::sum(reduce_buf_1, norm, num_blocks);
}

extern "C" void tea_leaf_ppcg_calc_p_kernel_cuda_
(double * beta)
{
    cuda_chunk.tea_leaf_kernel_ppcg_calc_p(*beta);
}

void TealeafCudaChunk::tea_leaf_kernel_ppcg_calc_p
(double beta)
{
    CUDALAUNCH(device_tea_leaf_ppcg_solve_calc_p, beta, vector_p, vector_z);
}




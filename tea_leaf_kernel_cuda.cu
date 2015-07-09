#include "cuda_common.hpp"

#include "kernel_files/tea_leaf_common.cuknl"
#include "kernel_files/tea_leaf_jacobi.cuknl"
#include "kernel_files/tea_leaf_cg.cuknl"
#include "kernel_files/tea_leaf_cheby.cuknl"
#include "kernel_files/tea_leaf_ppcg.cuknl"

#include <cassert>

// same as in fortran
#define CONDUCTIVITY 1
#define RECIP_CONDUCTIVITY 2

// copy back dx/dy and calculate rx/ry
void CloverleafCudaChunk::calcrxry
(double dt, double * rx, double * ry)
{
    double dx, dy;

    cudaMemcpy(&dx, celldx, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&dy, celldy, sizeof(double), cudaMemcpyDeviceToHost);

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
 const double * rx, const double * ry, const double * theta)
{
    cuda_chunk.tea_leaf_kernel_cheby_init(ch_alphas, ch_betas, *n_coefs,
        *rx, *ry, *theta);
}

extern "C" void tea_leaf_cheby_iterate_kernel_cuda_
(const double * rx, const double * ry, const int * cheby_calc_step)
{
    cuda_chunk.tea_leaf_kernel_cheby_iterate(*rx, *ry, *cheby_calc_step);
}

void CloverleafCudaChunk::tea_leaf_calc_2norm_kernel
(int norm_array, double* norm)
{
    if (norm_array == 0)
    {
        // norm of u0
        CUDALAUNCH(device_tea_leaf_common_calc_2norm, u0, u0, reduce_buf_1);
    }
    else if (norm_array == 1)
    {
        // norm of r
        CUDALAUNCH(device_tea_leaf_common_calc_2norm, vector_r, vector_r, reduce_buf_1);
    }
    else if (norm_array == 2)
    {
        CUDALAUNCH(device_tea_leaf_common_calc_2norm, vector_r, vector_z, reduce_buf_1);
    }
    else
    // TODO
    {
        DIE("Invalid value '%d' for norm_array passed, should be [1, 2]", norm_array);
    }

    CUDA_ERR_CHECK;

    *norm = thrust::reduce(reduce_ptr_1, reduce_ptr_1 + num_blocks, 0.0);
}

void CloverleafCudaChunk::upload_ch_coefs
(const double * ch_alphas, const double * ch_betas,
 const int n_coefs)
{
    size_t ch_buf_sz = n_coefs*sizeof(double);

    // upload to device
    cudaMalloc((void**) &ch_alphas_device, ch_buf_sz);
    cudaMalloc((void**) &ch_betas_device, ch_buf_sz);
    cudaMemcpy(ch_alphas_device, ch_alphas, ch_buf_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(ch_betas_device, ch_betas, ch_buf_sz, cudaMemcpyHostToDevice);
}

void CloverleafCudaChunk::tea_leaf_kernel_cheby_init
(const double * ch_alphas, const double * ch_betas, int n_coefs,
 const double rx, const double ry, const double theta)
{
    assert(tea_solver == TEA_ENUM_CHEBYSHEV);

    upload_ch_coefs(ch_alphas, ch_betas, n_coefs);

    CUDA_ERR_CHECK;

    CUDALAUNCH(device_tea_leaf_cheby_solve_init_p, u, u0,
        vector_p, vector_r, vector_w, vector_Mi,
        vector_Kx, vector_Ky,
        theta, rx, ry, preconditioner_on);

    // update p
    CUDALAUNCH(device_tea_leaf_cheby_solve_calc_u, u, vector_p);
}

void CloverleafCudaChunk::tea_leaf_kernel_cheby_iterate
(const double rx, const double ry, const int cheby_calc_step)
{
    CUDALAUNCH(device_tea_leaf_cheby_solve_calc_p, u, u0,
        vector_p, vector_r, vector_w, vector_Mi,
        vector_Kx, vector_Ky,
        ch_alphas_device, ch_betas_device,
        rx, ry, cheby_calc_step-1, preconditioner_on);

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

void CloverleafCudaChunk::tea_leaf_init_cg
(int coefficient, double dt, double * rx, double * ry, double * rro)
{
    assert(tea_solver == TEA_ENUM_CG || tea_solver == TEA_ENUM_CHEBYSHEV || tea_solver == TEA_ENUM_PPCG);

    // TODO preconditioners

    // init Kx, Ky
    CUDALAUNCH(device_tea_leaf_cg_init_p, vector_w, vector_Kx, vector_Ky);

    *rro = thrust::reduce(reduce_ptr_2, reduce_ptr_2 + num_blocks, 0.0);
}

void CloverleafCudaChunk::tea_leaf_kernel_cg_calc_w
(double rx, double ry, double* pw)
{
    CUDALAUNCH(device_tea_leaf_cg_solve_calc_w, reduce_buf_3,
        vector_p, vector_Mi, vector_Kx, vector_Ky, rx, ry);

    *pw = thrust::reduce(reduce_ptr_3, reduce_ptr_3 + num_blocks, 0.0);
}

void CloverleafCudaChunk::tea_leaf_kernel_cg_calc_ur
(double alpha, double* rrn)
{
    CUDALAUNCH(device_tea_leaf_cg_solve_calc_ur, alpha, reduce_buf_4, u, vector_p,
        vector_r, vector_Mi, z, vector_w, preconditioner_on);

    *rrn = thrust::reduce(reduce_ptr_4, reduce_ptr_4 + num_blocks, 0.0);
}

void CloverleafCudaChunk::tea_leaf_kernel_cg_calc_p
(double beta)
{
    CUDALAUNCH(device_tea_leaf_cg_solve_calc_p, beta, vector_p, vector_r, z,
        preconditioner_on);
}

/********************/

extern "C" void tea_leaf_jacobi_solve_kernel_cuda_
(double * error)
{
    cuda_chunk.tea_leaf_kernel_jacobi(error);
}

void CloverleafCudaChunk::tea_leaf_kernel_jacobi
(double* error)
{
    CUDALAUNCH(device_tea_leaf_jacobi_copy_u, u, vector_Mi);

    CUDALAUNCH(device_tea_leaf_jacobi_solve, rx, ry, vector_Kx, vector_Ky,
        vector_w, u, vector_Mi, reduce_buf_1);

    *error = *thrust::max_element(reduce_ptr_1, reduce_ptr_1 + num_blocks);
}

/********************/

extern "C" void tea_leaf_common_init_kernel_cuda_
(const int * coefficient, double * dt, double * rx, double * ry,
 int * chunk_neighbours, int * zero_boundary, int * reflective_boundary)
{
    cuda_chunk.tea_leaf_common_init(*coefficient, *dt, rx, ry,
        chunk_neighbours, zero_boundary, *reflective_boundary);
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

void CloverleafCudaChunk::tea_leaf_init_jacobi
(int coefficient, double dt, double * rx, double * ry)
{
    if (coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
    {
        DIE("Unknown coefficient %d passed to tea leaf\n", coefficient);
    }

    calcrxry(dt, rx, ry);

    CUDALAUNCH(device_tea_leaf_jacobi_init, density, energy1,
        vector_Kx, vector_Ky, vector_w, u, coefficient);
}

// both
void CloverleafCudaChunk::tea_leaf_finalise
(void)
{
    CUDALAUNCH(device_tea_leaf_finalise, density, u, energy1);
}

void CloverleafCudaChunk::tea_leaf_calc_residual
(void)
{
    CUDALAUNCH(device_tea_leaf_calc_residual, u, u0, vector_w,
        vector_Kx, vector_Ky);
}

/********************/

extern "C" void tea_leaf_kernel_ppcg_init_cuda_
(const double * ch_alphas, const double * ch_betas,
 int* n_inner_steps)
{
    cuda_chunk.ppcg_init(ch_alphas, ch_betas, *n_inner_steps);
}

extern "C" void tea_leaf_ppcg_init_sd_kernel_cuda_
(const double * theta)
{
    cuda_chunk.ppcg_init_sd(*theta);
}

extern "C" void tea_leaf_ppcg_inner_kernel_cuda
(int * ppcg_cur_step)
{
    cuda_chunk.ppcg_inner(*ppcg_cur_step);
}

void CloverleafCudaChunk::ppcg_init
(const double * ch_alphas, const double * ch_betas,
 const int n_inner_steps)
{
    upload_ch_coefs(ch_alphas, ch_betas, n_inner_steps);
}

void CloverleafCudaChunk::ppcg_init_sd
(double theta)
{
    CUDALAUNCH(device_tea_leaf_ppcg_solve_init_sd, vector_r,
        vector_Mi, vector_sd, theta);
}

void CloverleafCudaChunk::ppcg_inner
(int ppcg_cur_step)
{
    // TODO offsets
    CUDALAUNCH(device_tea_leaf_ppcg_solve_update_r, u, vector_r,
        vector_Kx, vector_Ky, vector_sd);

    CUDALAUNCH(device_tea_leaf_ppcg_solve_calc_sd, vector_r,
        vector_Mi, vector_sd, ch_alphas_device, ch_betas_device,
        ppcg_cur_step - 1);
}


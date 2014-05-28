#include "cuda_common.hpp"

#define CG_DO_PRECONDITION

#include "kernel_files/tea_leaf_common.cuknl"
#include "kernel_files/tea_leaf_jacobi.cuknl"
#include "kernel_files/tea_leaf_cg.cuknl"
#include "kernel_files/tea_leaf_cheby.cuknl"

#include <cassert>

// same as in fortran
#define CONDUCTIVITY 1
#define RECIP_CONDUCTIVITY 2

// copy back dx/dy and calculate rx/ry
void CloverleafCudaChunk::calcrxry
(double dt, double * rx, double * ry)
{
    static int initd = 0;
    if (!initd)
    {
        // FIXME remove this check - only relaly done once, one sync doesnt do much anyway
        // make sure intialise chunk has finished
        cudaDeviceSynchronize();
        // celldx doesnt change after that so check once
        initd = 1;
    }

    double dx, dy;

    cudaMemcpy(&dx, celldx, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&dy, celldy, sizeof(double), cudaMemcpyDeviceToHost);

    CUDA_ERR_CHECK;

    *rx = dt/(dx*dx);
    *ry = dt/(dy*dy);
}

/********************/

// Chebyshev solver
extern "C" void tea_leaf_kernel_cheby_copy_u_cuda_
(double* rro)
{
    cuda_chunk.tea_leaf_cheby_copy_u(rro);
}

extern "C" void tea_leaf_calc_2norm_kernel_cuda_
(int* norm_array, double* norm)
{
    cuda_chunk.tea_leaf_calc_2norm_kernel(*norm_array, norm);
}

extern "C" void tea_leaf_kernel_cheby_init_cuda_
(const double * ch_alphas, const double * ch_betas, int* n_coefs,
 const double * rx, const double * ry, const double * theta, double* error)
{
    cuda_chunk.tea_leaf_kernel_cheby_init(ch_alphas, ch_betas, *n_coefs,
        *rx, *ry, *theta, error);
}

extern "C" void tea_leaf_kernel_cheby_iterate_cuda_
(const double * ch_alphas, const double * ch_betas, int *n_coefs,
 const double * rx, const double * ry, const int * cheby_calc_step)
{
    cuda_chunk.tea_leaf_kernel_cheby_iterate(ch_alphas, ch_betas, *n_coefs,
        *rx, *ry, *cheby_calc_step);
}

void CloverleafCudaChunk::tea_leaf_cheby_copy_u
(double* rro)
{
    cudaDeviceSynchronize();
    cudaMemcpy(u0, u, BUFSZ2D(0, 0), cudaMemcpyDeviceToDevice);
    *rro = thrust::reduce(reduce_ptr_2, reduce_ptr_2 + num_blocks, 0.0);
}

void CloverleafCudaChunk::tea_leaf_calc_2norm_kernel
(int norm_array, double* norm)
{
    if (norm_array == 0)
    {
        // norm of u0
        device_tea_leaf_cheby_solve_calc_resid<<< num_blocks, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, u0, reduce_buf_1);
    }
    else if (norm_array == 1)
    {
        // norm of r
        device_tea_leaf_cheby_solve_calc_resid<<< num_blocks, BLOCK_SZ >>>
        (x_min, x_max, y_min, y_max, work_array_2, reduce_buf_1);
    }
    else
    {
        DIE("Invalid value '%d' for norm_array passed, should be [1, 2]", norm_array);
    }

    CUDA_ERR_CHECK;

    *norm = thrust::reduce(reduce_ptr_1, reduce_ptr_1 + num_blocks, 0.0);
}

void CloverleafCudaChunk::tea_leaf_kernel_cheby_init
(const double * ch_alphas, const double * ch_betas, int n_coefs,
 const double rx, const double ry, const double theta, double* error)
{
    assert(tea_solver == TEA_ENUM_CHEBYSHEV);

    size_t ch_buf_sz = n_coefs*sizeof(double);

    // upload to device
    cudaMalloc((void**) &ch_alphas_device, ch_buf_sz);
    cudaMalloc((void**) &ch_betas_device, ch_buf_sz);
    cudaMemcpy(ch_alphas_device, ch_alphas, ch_buf_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(ch_betas_device, ch_betas, ch_buf_sz, cudaMemcpyHostToDevice);

    CUDA_ERR_CHECK;

    // this will junk p but we don't need it anyway
    tea_leaf_kernel_cheby_iterate(ch_alphas, ch_betas, 0, rx, ry, 1);

    // then correct p
    CUDALAUNCH(device_tea_leaf_cheby_solve_init_p, work_array_1, z, theta);

    // do a step like in fortran
    tea_leaf_kernel_cheby_iterate(ch_alphas, ch_betas, 0, rx, ry, 1);

    // get norm of r
    tea_leaf_calc_2norm_kernel(1, error);
}

void CloverleafCudaChunk::tea_leaf_kernel_cheby_iterate
(const double * ch_alphas, const double * ch_betas, int n_coefs,
 const double rx, const double ry, const int cheby_calc_step)
{
    CUDALAUNCH(device_tea_leaf_cheby_solve_calc_u, u, work_array_1);

    CUDALAUNCH(device_tea_leaf_cheby_solve_calc_p, u, u0,
        work_array_1, work_array_2, work_array_3, work_array_4,
        z, work_array_5, work_array_6,
        ch_alphas_device, ch_betas_device,
        rx, ry, cheby_calc_step-1);
}

/********************/

// CG solver functions
extern "C" void tea_leaf_kernel_init_cg_cuda_
(const int * coefficient, double * dt, double * rx, double * ry, double * rro)
{
    cuda_chunk.tea_leaf_init_cg(*coefficient, *dt, rx, ry, rro);
}

extern "C" void tea_leaf_kernel_solve_cg_cuda_calc_w_
(const double * rx, const double * ry, double * pw)
{
    cuda_chunk.tea_leaf_kernel_cg_calc_w(*rx, *ry, pw);
}
extern "C" void tea_leaf_kernel_solve_cg_cuda_calc_ur_
(double * alpha, double * rrn)
{
    cuda_chunk.tea_leaf_kernel_cg_calc_ur(*alpha, rrn);
}
extern "C" void tea_leaf_kernel_solve_cg_cuda_calc_p_
(double * beta)
{
    cuda_chunk.tea_leaf_kernel_cg_calc_p(*beta);
}

/********************/

void CloverleafCudaChunk::tea_leaf_init_cg
(int coefficient, double dt, double * rx, double * ry, double * rro)
{
    if (coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
    {
        DIE("Unknown coefficient %d passed to tea leaf\n", coefficient);
    }

    assert(tea_solver == TEA_ENUM_CG || tea_solver == TEA_ENUM_CHEBYSHEV);

    calcrxry(dt, rx, ry);

    CUDALAUNCH(device_tea_leaf_cg_init_u, density1, energy1, u,
        work_array_1, work_array_2, work_array_3, coefficient);

    // init Kx, Ky
    CUDALAUNCH(device_tea_leaf_cg_init_directions, work_array_3, work_array_5, work_array_6);

    // premultiply Kx/Ky
    CUDALAUNCH(device_tea_leaf_init_diag, work_array_5, work_array_6, *rx, *ry);

    // get initial guess in w, r, etc
    CUDALAUNCH(device_tea_leaf_cg_init_others, reduce_buf_2, u,
        work_array_1, work_array_2, work_array_3, work_array_4, z,
        work_array_5, work_array_6, *rx, *ry);

    *rro = thrust::reduce(reduce_ptr_2, reduce_ptr_2 + num_blocks, 0.0);
}

void CloverleafCudaChunk::tea_leaf_kernel_cg_calc_w
(double rx, double ry, double* pw)
{
    CUDALAUNCH(device_tea_leaf_cg_solve_calc_w, reduce_buf_3,
        work_array_1, work_array_4, work_array_5, work_array_6, rx, ry);

    *pw = thrust::reduce(reduce_ptr_3, reduce_ptr_3 + num_blocks, 0.0);
}

void CloverleafCudaChunk::tea_leaf_kernel_cg_calc_ur
(double alpha, double* rrn)
{
    CUDALAUNCH(device_tea_leaf_cg_solve_calc_ur, alpha, reduce_buf_4, u, work_array_1,
        work_array_2, work_array_4, z, work_array_3);

    *rrn = thrust::reduce(reduce_ptr_4, reduce_ptr_4 + num_blocks, 0.0);
}

void CloverleafCudaChunk::tea_leaf_kernel_cg_calc_p
(double beta)
{
    CUDALAUNCH(device_tea_leaf_cg_solve_calc_p, beta, work_array_1, work_array_2, z);
}

/********************/

// jacobi solver functions
extern "C" void tea_leaf_kernel_init_cuda_
(const int * coefficient, double * dt, double * rx, double * ry)
{
    cuda_chunk.tea_leaf_init_jacobi(*coefficient, *dt, rx, ry);
}

extern "C" void tea_leaf_kernel_solve_cuda_
(const double * rx, const double * ry, double * error)
{
    cuda_chunk.tea_leaf_kernel_jacobi(*rx, *ry, error);
}

// jacobi
void CloverleafCudaChunk::tea_leaf_init_jacobi
(int coefficient, double dt, double * rx, double * ry)
{
    if (coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
    {
        DIE("Unknown coefficient %d passed to tea leaf\n", coefficient);
    }

    calcrxry(dt, rx, ry);

    CUDALAUNCH(device_tea_leaf_jacobi_init, density1, energy1,
        work_array_1, work_array_2, work_array_3, u, coefficient);
}

void CloverleafCudaChunk::tea_leaf_kernel_jacobi
(double rx, double ry, double* error)
{
    CUDALAUNCH(device_tea_leaf_jacobi_copy_u, u, work_array_4);

    CUDALAUNCH(device_tea_leaf_jacobi_solve, rx, ry, work_array_1, work_array_2,
        work_array_3, u, work_array_4, reduce_buf_1);

    *error = thrust::reduce(reduce_ptr_1, reduce_ptr_1 + num_blocks, 0.0);
}

/********************/

// used by both
extern "C" void tea_leaf_kernel_finalise_cuda_
(void)
{
    cuda_chunk.tea_leaf_finalise();
}

// both
void CloverleafCudaChunk::tea_leaf_finalise
(void)
{
    CUDALAUNCH(device_tea_leaf_finalise, density1, u, energy1);
}


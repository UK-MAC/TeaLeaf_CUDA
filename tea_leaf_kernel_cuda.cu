#include "cuda_common.hpp"
#include "kernel_files/tea_leaf_kernel.cuknl"

// same as in fortran
#define CONDUCTIVITY 1
#define RECIP_CONDUCTIVITY 2

// Chebyshev solver
extern "C" void tea_leaf_kernel_cheby_copy_u_cuda_
(double* rro)
{
    chunk.tea_leaf_cheby_copy_u(rro);
}

extern "C" void tea_leaf_calc_2norm_kernel_cuda_
(int* norm_array, double* norm)
{
    chunk.tea_leaf_calc_2norm_kernel(*norm_array, norm);
}

extern "C" void tea_leaf_kernel_cheby_init_cuda_
(const double * ch_alphas, const double * ch_betas, int* n_coefs,
 const double * rx, const double * ry, const double * theta, double* error)
{
    chunk.tea_leaf_kernel_cheby_init(ch_alphas, ch_betas, *n_coefs,
        *rx, *ry, *theta, error);
}

extern "C" void tea_leaf_kernel_cheby_iterate_cuda_
(const double * ch_alphas, const double * ch_betas, int *n_coefs,
 const double * rx, const double * ry, const int * cheby_calc_step)
{
    chunk.tea_leaf_kernel_cheby_iterate(ch_alphas, ch_betas, *n_coefs,
        *rx, *ry, *cheby_calc_step);
}

void CloverleafCudaChunk::tea_leaf_cheby_copy_u
(double* rro)
{
    cudaDeviceSynchronize();
    cudaMemcpy(u0, u, BUFSZ2D(1, 1), cudaMemcpyDeviceToDevice);
    cudaMemcpy(rro, work_array_2, sizeof(double), cudaMemcpyDeviceToHost);
}

void CloverleafCudaChunk::tea_leaf_calc_2norm_kernel
(int norm_array, double* norm)
{
    // TODO
    if (norm_array == 0)
    {
        // norm of u0
        tea_leaf_cheby_solve_calc_resid_device.setArg(0, u0);
    }
    else if (norm_array == 1)
    {
        // norm of r
        tea_leaf_cheby_solve_calc_resid_device.setArg(0, work_array_2);
    }
    else
    {
        DIE("Invalid value '%d' for norm_array passed, should be [1, 2]", norm_array);
    }

    ENQUEUE(tea_leaf_cheby_solve_calc_resid_device);
    *norm = reduceValue<double>(sum_red_kernels_double, reduce_buf_1);
}

void CloverleafCudaChunk::tea_leaf_kernel_cheby_init
(const double * ch_alphas, const double * ch_betas, int n_coefs,
 const double rx, const double ry, const double theta, double* error)
{
    // TODO
    size_t ch_buf_sz = n_coefs*sizeof(double);

    // upload to device
    ch_alphas_device = cl::Buffer(context, CL_MEM_READ_ONLY, ch_buf_sz);
    queue.enqueueWriteBuffer(ch_alphas_device, CL_TRUE, 0, ch_buf_sz, ch_alphas);
    ch_betas_device = cl::Buffer(context, CL_MEM_READ_ONLY, ch_buf_sz);
    queue.enqueueWriteBuffer(ch_betas_device, CL_TRUE, 0, ch_buf_sz, ch_betas);
    tea_leaf_cheby_solve_calc_p_device.setArg(7, ch_alphas_device);
    tea_leaf_cheby_solve_calc_p_device.setArg(8, ch_betas_device);
    tea_leaf_cheby_solve_calc_p_device.setArg(9, rx);
    tea_leaf_cheby_solve_calc_p_device.setArg(10, ry);

    tea_leaf_cheby_solve_init_p_device.setArg(2, theta);

    // this will junk p but we don't need it anyway
    chunk.tea_leaf_kernel_cheby_iterate(NULL, NULL, 0, rx, ry, 1);

    // get norm of r
    tea_leaf_calc_2norm_kernel(1, error);

    // then correct p
    ENQUEUE(tea_leaf_cheby_solve_init_p_device);
}

void CloverleafCudaChunk::tea_leaf_kernel_cheby_iterate
(const double * ch_alphas, const double * ch_betas, int n_coefs,
 const double rx, const double ry, const int cheby_calc_step)
{
    // TODO
    tea_leaf_cheby_solve_calc_p_device.setArg(11, cheby_calc_step-1);

    //ENQUEUE(tea_leaf_cheby_solve_calc_u_device);
    ENQUEUE_OFFSET(tea_leaf_cheby_solve_calc_u_device);
    //ENQUEUE(tea_leaf_cheby_solve_calc_p_device);
    ENQUEUE_OFFSET(tea_leaf_cheby_solve_calc_p_device);
}

/********************/

// CG solver functions
extern "C" void tea_leaf_kernel_init_cg_cuda_
(const int * coefficient, double * dt, double * rx, double * ry, double * rro)
{
    chunk.tea_leaf_init_cg(*coefficient, *dt, rx, ry, rro);
}

extern "C" void tea_leaf_kernel_solve_cg_cuda_calc_w_
(const double * rx, const double * ry, double * pw)
{
    chunk.tea_leaf_kernel_cg_calc_w(*rx, *ry, pw);
}
extern "C" void tea_leaf_kernel_solve_cg_cuda_calc_ur_
(double * alpha, double * rrn)
{
    chunk.tea_leaf_kernel_cg_calc_ur(*alpha, rrn);
}
extern "C" void tea_leaf_kernel_solve_cg_cuda_calc_p_
(double * beta)
{
    chunk.tea_leaf_kernel_cg_calc_p(*beta);
}

// copy back dx/dy and calculate rx/ry
void CloverleafCudaChunk::calcrxry
(double dt, double * rx, double * ry)
{
    static int initd = 0;
    if (!initd)
    {
        // make sure intialise chunk has finished
        queue.finish();
        // celldx doesnt change after that so check once
        initd = 1;
    }

    double dx, dy;

    try
    {
        // TODO
        // celldx/celldy never change, but done for consistency with fortran
        queue.enqueueReadBuffer(celldx, CL_TRUE,
            sizeof(double)*x_min, sizeof(double), &dx);
        queue.enqueueReadBuffer(celldy, CL_TRUE,
            sizeof(double)*y_min, sizeof(double), &dy);
    }
    catch (cl::Error e)
    {
        DIE("Error in copying back value from celldx/celldy (%d - %s)\n",
            e.err(), e.what());
    }

    *rx = dt/(dx*dx);
    *ry = dt/(dy*dy);
}

/********************/
#include <cassert>

void CloverleafCudaChunk::tea_leaf_init_cg
(int coefficient, double dt, double * rx, double * ry, double * rro)
{
    if (coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY)
    {
        DIE("Unknown coefficient %d passed to tea leaf\n", coefficient);
    }

    assert(tea_solver == TEA_ENUM_CG || tea_solver == TEA_ENUM_CHEBYSHEV);

    calcrxry(dt, rx, ry);

    // TODO
    // only needs to be set once
    tea_leaf_cg_solve_calc_w_device.setArg(5, *rx);
    tea_leaf_cg_solve_calc_w_device.setArg(6, *ry);
    tea_leaf_cg_init_others_device.setArg(8, *rx);
    tea_leaf_cg_init_others_device.setArg(9, *ry);
    tea_leaf_init_diag_device.setArg(2, *rx);
    tea_leaf_init_diag_device.setArg(3, *ry);

    // copy u, get density value modified by coefficient
    tea_leaf_cg_init_u_device.setArg(6, coefficient);
    //ENQUEUE(tea_leaf_cg_init_u_device);
    ENQUEUE_OFFSET(tea_leaf_cg_init_u_device);

    // init Kx, Ky
    //ENQUEUE(tea_leaf_cg_init_directions_device);
    ENQUEUE_OFFSET(tea_leaf_cg_init_directions_device);

    // premultiply Kx/Ky
    //ENQUEUE(tea_leaf_init_diag_device);
    ENQUEUE_OFFSET(tea_leaf_init_diag_device);

    // get initial guess in w, r, etc
    //ENQUEUE(tea_leaf_cg_init_others_device);
    ENQUEUE_OFFSET(tea_leaf_cg_init_others_device);

    *rro = reduceValue<double>(sum_red_kernels_double, reduce_buf_2);
}

void CloverleafCudaChunk::tea_leaf_kernel_cg_calc_w
(double rx, double ry, double* pw)
{
    // TODO
    //ENQUEUE(tea_leaf_cg_solve_calc_w_device);
    ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_w_device);
    *pw = reduceValue<double>(sum_red_kernels_double, reduce_buf_3);
}

void CloverleafCudaChunk::tea_leaf_kernel_cg_calc_ur
(double alpha, double* rrn)
{
    // TODO
    tea_leaf_cg_solve_calc_ur_device.setArg(0, alpha);

    //ENQUEUE(tea_leaf_cg_solve_calc_ur_device);
    ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_ur_device);
    *rrn = reduceValue<double>(sum_red_kernels_double, reduce_buf_4);
}

void CloverleafCudaChunk::tea_leaf_kernel_cg_calc_p
(double beta)
{
    // TODO
    tea_leaf_cg_solve_calc_p_device.setArg(0, beta);

    ENQUEUE(tea_leaf_cg_solve_calc_p_device);
    //ENQUEUE_OFFSET(tea_leaf_cg_solve_calc_p_device);
}

/********************/

// jacobi solver functions
extern "C" void tea_leaf_kernel_init_cuda_
(const int * coefficient, double * dt, double * rx, double * ry)
{
    chunk.tea_leaf_init_jacobi(*coefficient, *dt, rx, ry);
}

extern "C" void tea_leaf_kernel_solve_cuda_
(const double * rx, const double * ry, double * error)
{
    chunk.tea_leaf_kernel_jacobi(*rx, *ry, error);
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

    // TODO
    tea_leaf_jacobi_init_device.setArg(6, coefficient);
    //ENQUEUE(tea_leaf_jacobi_init_device);
    ENQUEUE_OFFSET(tea_leaf_jacobi_init_device);

    tea_leaf_jacobi_solve_device.setArg(0, *rx);
    tea_leaf_jacobi_solve_device.setArg(1, *ry);
}

void CloverleafCudaChunk::tea_leaf_kernel_jacobi
(double rx, double ry, double* error)
{
    // TODO
    //ENQUEUE(tea_leaf_jacobi_copy_u_device);
    ENQUEUE_OFFSET(tea_leaf_jacobi_copy_u_device);
    //ENQUEUE(tea_leaf_jacobi_solve_device);
    ENQUEUE_OFFSET(tea_leaf_jacobi_solve_device);

    *error = reduceValue<double>(max_red_kernels_double, reduce_buf_1);
}

/********************/

// used by both
extern "C" void tea_leaf_kernel_finalise_cuda_
(void)
{
    chunk.tea_leaf_finalise();
}

// both
void CloverleafCudaChunk::tea_leaf_finalise
(void)
{
    // TODO
    //ENQUEUE(tea_leaf_finalise_device);
    ENQUEUE_OFFSET(tea_leaf_finalise_device);
}


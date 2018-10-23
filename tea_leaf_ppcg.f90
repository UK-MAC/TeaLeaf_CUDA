
MODULE tea_leaf_ppcg_module

  USE tea_leaf_cheby_module
  USE definitions_module
  USE update_halo_module

  IMPLICIT NONE
  
CONTAINS

SUBROUTINE tea_leaf_ppcg_init_sd(theta)

  IMPLICIT NONE

  INTEGER :: t
  REAL(KIND=8) :: theta

  IF (use_cuda_kernels) THEN
    DO t=1,tiles_per_task
      CALL tea_leaf_ppcg_init_sd_kernel_cuda(theta)
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_ppcg_init_sd

SUBROUTINE tea_leaf_ppcg_init_sd_new(theta)

  IMPLICIT NONE

  INTEGER :: t
  REAL(KIND=8) :: theta

  IF (use_cuda_kernels) THEN
    DO t=1,tiles_per_task
      CALL tea_leaf_ppcg_init_sd_new_kernel_cuda(theta)
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_ppcg_init_sd_new


SUBROUTINE tea_leaf_ppcg_init(ppcg_inner_iters, ch_alphas, ch_betas, theta, step, rrn)

  IMPLICIT NONE

  INTEGER :: ppcg_inner_iters,step
  REAL(KIND=8) :: rrn,theta
  REAL(KIND=8), DIMENSION(ppcg_inner_iters) :: ch_alphas,ch_betas

  INTEGER :: t
  REAL(KIND=8) :: tile_rrn

  rrn = 0.0_8

  IF (use_cuda_kernels) THEN
    DO t=1,tiles_per_task
      tile_rrn = 0.0_8
      CALL tea_leaf_ppcg_init_kernel_cuda(step, tile_rrn)
      rrn = rrn + tile_rrn
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_ppcg_init


SUBROUTINE tea_leaf_ppcg_inner(ch_alphas, ch_betas, inner_step, bounds_extra)

  IMPLICIT NONE

  INTEGER :: t, inner_step, bounds_extra
  REAL(KIND=8), DIMENSION(:) :: ch_alphas, ch_betas

  IF (use_cuda_kernels) THEN
    DO t=1,tiles_per_task
      CALL tea_leaf_ppcg_inner_kernel_cuda(inner_step, bounds_extra, chunk%chunk_neighbours)
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_ppcg_inner

SUBROUTINE tea_leaf_ppcg_calc_zrnorm(rrn)

  IMPLICIT NONE

  INTEGER :: t
  REAL(KIND=8) :: rrn, tile_rrn

  rrn = 0.0_8

  IF (use_cuda_kernels) THEN
    DO t=1,tiles_per_task
      tile_rrn = 0.0_8

      CALL tea_leaf_ppcg_calc_2norm_kernel_cuda(tile_rrn)

      rrn = rrn + tile_rrn
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_ppcg_calc_zrnorm

! New: ppcg_store_r

SUBROUTINE tea_leaf_ppcg_store_r()

  IMPLICIT NONE

  INTEGER :: t

  IF (use_cuda_kernels) THEN
    DO t=1,tiles_per_task

      CALL tea_leaf_ppcg_store_r_kernel_cuda()

    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_ppcg_store_r

! New: update z

SUBROUTINE tea_leaf_ppcg_update_z()

  IMPLICIT NONE

  INTEGER :: t

  IF (use_cuda_kernels) THEN
    DO t=1,tiles_per_task
      CALL tea_leaf_ppcg_update_z_kernel_cuda()
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_ppcg_update_z

! New

SUBROUTINE tea_leaf_ppcg_calc_rrn(rrn)

  IMPLICIT NONE

  INTEGER :: t
  REAL(KIND=8) :: rrn, tile_rrn

  rrn = 0.0_8

  IF (use_cuda_kernels) THEN
    DO t=1,tiles_per_task
      tile_rrn = 0.0_8

      CALL tea_leaf_ppcg_calc_rrn_kernel_cuda(tile_rrn)

      rrn = rrn + tile_rrn
    ENDDO
  ENDIF

END SUBROUTINE


SUBROUTINE tea_calc_ls_coefs(ch_alphas, ch_betas, eigmin, eigmax, &
                             theta, ppcg_inner_steps)

  INTEGER :: ppcg_inner_steps
  REAL(KIND=8), DIMENSION(ppcg_inner_steps) :: ch_alphas, ch_betas
  REAL(KIND=8) :: eigmin, eigmax, theta

  ! TODO
  CALL tea_calc_ch_coefs(ch_alphas, ch_betas, eigmin, eigmax, &
                         theta, ppcg_inner_steps)

END SUBROUTINE

SUBROUTINE tea_leaf_run_ppcg_inner_steps(ch_alphas, ch_betas, theta, &
    tl_ppcg_inner_steps, solve_time)

  IMPLICIT NONE

  INTEGER :: fields(NUM_FIELDS)
  INTEGER :: tl_ppcg_inner_steps, ppcg_cur_step
  REAL(KIND=8) :: theta
  REAL(KIND=8) :: halo_time, timer, solve_time
  REAL(KIND=8), DIMENSION(max_iters) :: ch_alphas, ch_betas

  INTEGER(KIND=4) :: inner_step, bounds_extra

  fields = 0
  fields(FIELD_U) = 1

  IF (profiler_on) halo_time=timer()
  CALL update_halo(fields,1)
  IF (profiler_on) solve_time = solve_time + (timer() - halo_time)
  
  IF (tl_ppcg_inner_steps < 0) RETURN
  
  CALL tea_leaf_ppcg_init_sd_new(theta)

  ! inner steps
  DO ppcg_cur_step=1,tl_ppcg_inner_steps,halo_exchange_depth

    fields = 0
    fields(FIELD_SD) = 1
    !fields(FIELD_R) = 1

    IF (profiler_on) halo_time = timer()
    CALL update_halo(fields,halo_exchange_depth)
    IF (profiler_on) solve_time = solve_time + (timer()-halo_time)

    inner_step = ppcg_cur_step

    fields = 0
    fields(FIELD_SD) = 1

    DO bounds_extra = halo_exchange_depth-1, 0, -1
      CALL tea_leaf_ppcg_inner(ch_alphas, ch_betas, (ppcg_cur_step + halo_exchange_depth-1 - bounds_extra), bounds_extra)

      IF (profiler_on) halo_time = timer()
      CALL update_boundary(fields, 1)
      IF (profiler_on) solve_time = solve_time + (timer()-halo_time)
      !print*, (ppcg_cur_step + halo_exchange_depth-1 -bounds_extra)
      IF ((ppcg_cur_step + halo_exchange_depth-1 -bounds_extra)  .eq. tl_ppcg_inner_steps) EXIT
    ENDDO
  ENDDO
!stop
  fields = 0
  fields(FIELD_P) = 1
  
  CALL tea_leaf_ppcg_update_z()

END SUBROUTINE tea_leaf_run_ppcg_inner_steps

SUBROUTINE tea_leaf_ppcg_calc_p(beta)

  IMPLICIT NONE

  INTEGER :: t
  REAL(KIND=8) :: beta

  IF (use_cuda_kernels) THEN
    DO t=1,tiles_per_task
      CALL tea_leaf_ppcg_calc_p_kernel_cuda(beta)
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_ppcg_calc_p


END MODULE tea_leaf_ppcg_module


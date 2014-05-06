!Crown Copyright 2014 AWE.
!
! This file is part of TeaLeaf.
!
! TeaLeaf is free software: you can redistribute it and/or modify it under 
! the terms of the GNU General Public License as published by the 
! Free Software Foundation, either version 3 of the License, or (at your option) 
! any later version.
!
! TeaLeaf is distributed in the hope that it will be useful, but 
! WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
! FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
! details.
!
! You should have received a copy of the GNU General Public License along with 
! TeaLeaf. If not, see http://www.gnu.org/licenses/.

!>  @brief Driver for the heat conduction kernel
!>  @author David Beckingsale, Wayne Gaudin
!>  @details Invokes the user specified kernel for the heat conduction

MODULE tea_leaf_module
 
  USE report_module
  USE data_module
  USE tea_leaf_kernel_module
  USE tea_leaf_kernel_cg_module
  USE tea_leaf_kernel_cheby_module
  USE update_halo_module

  IMPLICIT NONE

  interface
    subroutine tea_leaf_kernel_cheby_copy_u_ocl(rro)
      real(kind=8) :: rro
    end subroutine
    subroutine tea_leaf_calc_2norm_kernel_ocl(initial, norm)
      integer :: initial
      real(kind=8) :: norm
    end subroutine
    subroutine tea_leaf_kernel_cheby_init_ocl(ch_alphas, ch_betas, n_coefs, &
        rx, ry, theta, error)
      real(kind=8) :: rx, ry, theta, error
      integer :: n_coefs
      real(kind=8), dimension(n_coefs) :: ch_alphas, ch_betas
    end subroutine
    subroutine tea_leaf_kernel_cheby_iterate_ocl(ch_alphas, ch_betas, n_coefs, &
        rx, ry, cheby_calc_step)
      real(kind=8) :: rx, ry
      integer :: cheby_calc_step
      integer :: n_coefs
      real(kind=8), dimension(n_coefs) :: ch_alphas, ch_betas
    end subroutine
    subroutine tqli(d, e, np, z, info)
      real(kind=8),dimension(np) :: d, e, z
      integer :: np, info
    end subroutine
  end interface

CONTAINS

SUBROUTINE tea_leaf()

!$ INTEGER :: OMP_GET_THREAD_NUM
  INTEGER :: c, n
  REAL(KIND=8) :: ry,rx, error, old_error

  INTEGER :: fields(NUM_FIELDS)

  REAL(KIND=8) :: kernel_time,timer

  ! For CG solver
  REAL(KIND=8) :: rro, pw, rrn, alpha, beta

  ! For chebyshev solver
  REAL(KIND=8), DIMENSION(tl_chebyshev_steps) :: cg_alphas, cg_betas
  REAL(KIND=8), DIMENSION(max_iters) :: ch_alphas, ch_betas
  REAL(KIND=8) :: eigmin, eigmax, theta
  REAL(KIND=8) :: it_alpha, cn, gamm, bb
  INTEGER :: est_itc, cheby_calc_steps, max_cheby_iters, info

  IF(coefficient .nE. RECIP_CONDUCTIVITY .and. coefficient .ne. conductivity) THEN
    CALL report_error('tea_leaf', 'unknown coefficient option')
  endif

  DO c=1,number_of_chunks

    IF(chunks(c)%task.EQ.parallel%task) THEN

      ! set old error to 0 initially
      old_error = 0.0

      fields=0
      fields(FIELD_ENERGY1) = 1
      fields(FIELD_DENSITY1) = 1
      CALL update_halo(fields,2)

      ! INIT
      IF(profiler_on) kernel_time=timer()

      if (use_fortran_kernels .or. use_c_kernels) then
        rx = dt/(chunks(c)%field%celldx(chunks(c)%field%x_min)**2)
        ry = dt/(chunks(c)%field%celldy(chunks(c)%field%y_min)**2)
      endif

      IF(tl_use_cg .or. tl_use_chebyshev) then
        IF(use_fortran_kernels) THEN
          CALL tea_leaf_kernel_init_cg_fortran(chunks(c)%field%x_min, &
              chunks(c)%field%x_max,                       &
              chunks(c)%field%y_min,                       &
              chunks(c)%field%y_max,                       &
              chunks(c)%field%density1,                    &
              chunks(c)%field%energy1,                     &
              chunks(c)%field%u,                           &
              chunks(c)%field%work_array1,                 &
              chunks(c)%field%work_array2,                 &
              chunks(c)%field%work_array3,                 &
              chunks(c)%field%work_array4,                 &
              chunks(c)%field%work_array5,                 &
              chunks(c)%field%work_array6,                 &
              chunks(c)%field%work_array7,                 &
              rx, ry, rro, coefficient)
        ELSEIF(use_opencl_kernels) THEN
          CALL tea_leaf_kernel_init_cg_ocl(coefficient, dt, rx, ry, rro)
        ELSEIF(use_C_kernels) THEN
          CALL tea_leaf_kernel_init_cg_c(chunks(c)%field%x_min, &
              chunks(c)%field%x_max,                       &
              chunks(c)%field%y_min,                       &
              chunks(c)%field%y_max,                       &
              chunks(c)%field%density1,                    &
              chunks(c)%field%energy1,                     &
              chunks(c)%field%u,                           &
              chunks(c)%field%work_array1,                 &
              chunks(c)%field%work_array2,                 &
              chunks(c)%field%work_array3,                 &
              chunks(c)%field%work_array4,                 &
              chunks(c)%field%work_array5,                 &
              chunks(c)%field%work_array6,                 &
              chunks(c)%field%work_array7,                 &
              rx, ry, rro, coefficient)
        ENDIF

        ! need to update p at this stage
        fields=0
        fields(FIELD_U) = 1
        fields(FIELD_P) = 1
        CALL update_halo(fields,2)

        ! and globally sum rro
        call clover_allsum(rro)
      ELSE
        IF (use_fortran_kernels) THEN
          CALL tea_leaf_kernel_init(chunks(c)%field%x_min, &
              chunks(c)%field%x_max,                       &
              chunks(c)%field%y_min,                       &
              chunks(c)%field%y_max,                       &
              chunks(c)%field%celldx,                      &
              chunks(c)%field%celldy,                      &
              chunks(c)%field%volume,                      &
              chunks(c)%field%density1,                    &
              chunks(c)%field%energy1,                     &
              chunks(c)%field%work_array1,                 &
              chunks(c)%field%u,                           &
              chunks(c)%field%work_array2,                 &
              chunks(c)%field%work_array3,                 &
              chunks(c)%field%work_array4,                 &
              chunks(c)%field%work_array5,                 &
              chunks(c)%field%work_array6,                 &
              chunks(c)%field%work_array7,                 &
              coefficient)
        ELSEIF(use_opencl_kernels) THEN
          CALL tea_leaf_kernel_init_ocl(coefficient, dt, rx, ry)
        ELSEIF(use_C_kernels) THEN
          CALL tea_leaf_kernel_init_c(chunks(c)%field%x_min, &
              chunks(c)%field%x_max,                       &
              chunks(c)%field%y_min,                       &
              chunks(c)%field%y_max,                       &
              chunks(c)%field%celldx,                      &
              chunks(c)%field%celldy,                      &
              chunks(c)%field%volume,                      &
              chunks(c)%field%density1,                    &
              chunks(c)%field%energy1,                     &
              chunks(c)%field%work_array1,                 &
              chunks(c)%field%u,                           &
              chunks(c)%field%work_array2,                 &
              chunks(c)%field%work_array3,                 &
              chunks(c)%field%work_array4,                 &
              chunks(c)%field%work_array5,                 &
              chunks(c)%field%work_array6,                 &
              chunks(c)%field%work_array7,                 &
              coefficient)
        ENDIF

      ENDIF

      fields=0
      fields(FIELD_U) = 1

      ! need the original value of u
      if(tl_use_chebyshev) then
        IF(use_fortran_kernels) then
          call tea_leaf_kernel_cheby_copy_u(chunks(c)%field%x_min,&
            chunks(c)%field%x_max,                       &
            chunks(c)%field%y_min,                       &
            chunks(c)%field%y_max,                       &
            chunks(c)%field%u0,                &
            chunks(c)%field%u)

          ! TODO find a smarter way to do this
          !
          ! If a preconditioner is used with the CG solver before launching
          ! into the chebyshev routine then the eigenvalues calculated are
          ! that of the preconditioned system and not the original one.
          ! Preconditioner is easily disabled in OpenCL, but here it's either
          ! do something like this or copy the functions but remove the Mi
          ! and z arrays (messy). This does mean extra memory bandwidth will
          ! be used, but it's not too much of an issue
          call tea_leaf_kernel_cheby_reset_Mi(chunks(c)%field%x_min,&
            chunks(c)%field%x_max,                       &
            chunks(c)%field%y_min,                       &
            chunks(c)%field%y_max,                       &
            chunks(c)%field%work_array1,                &
            chunks(c)%field%work_array2,                &
            chunks(c)%field%work_array3,                &
            chunks(c)%field%work_array5,                &
            rro)
        elseif(use_opencl_kernels) then
          write(*,*) rro
          call tea_leaf_kernel_cheby_copy_u_ocl(rro)
          write(*,*) rro
        endif

        call clover_allsum(rro)
      endif

      DO n=1,max_iters

        IF(tl_use_chebyshev .and. (n .gt. tl_chebyshev_steps)) then
          ! on the first chebyshev steps, find the eigenvalues, coefficients,
          ! and expected number of iterations
          IF (n .eq. tl_chebyshev_steps+1) then
            ! maximum number of iterations in chebyshev solver
            max_cheby_iters = max_iters - n + 2
            info = 0
            ! calculate eigenvalues
            call tea_calc_eigenvalues(cg_alphas, cg_betas, eigmin, eigmax,  &
                max_iters, n, info)

            if (info .ne. 0) then
                call report_error('tea_leaf', 'Error calculating eigenvalues')
            endif

            ! calculate chebyshev coefficients
            call tea_calc_ch_coefs(ch_alphas, ch_betas, eigmin, eigmax, &
                theta, max_cheby_iters)

            ! calculate 2 norm of u0
            IF(use_fortran_kernels) THEN
              call tea_leaf_calc_2norm_kernel(chunks(c)%field%x_min,        &
                    chunks(c)%field%x_max,                       &
                    chunks(c)%field%y_min,                       &
                    chunks(c)%field%y_max,                       &
                    chunks(c)%field%u0,                 &
                    bb)
            ELSEIF(use_opencl_kernels) THEN
              call tea_leaf_calc_2norm_kernel_ocl(0, bb)
            ENDIF
            call clover_allsum(bb)

            ! initialise 'p' array
            IF(use_fortran_kernels) THEN
              call tea_leaf_kernel_cheby_init(chunks(c)%field%x_min,&
                    chunks(c)%field%x_max,                       &
                    chunks(c)%field%y_min,                       &
                    chunks(c)%field%y_max,                       &
                    chunks(c)%field%u,                           &
                    chunks(c)%field%work_array1,                 &
                    chunks(c)%field%work_array2,                 &
                    chunks(c)%field%u0,                 &
                    chunks(c)%field%work_array5,                 &
                    chunks(c)%field%work_array6,                 &
                    chunks(c)%field%work_array7,                 &
                    ch_alphas, ch_betas, max_cheby_iters, &
                    rx, ry, theta, error)
            ELSEIF(use_opencl_kernels) THEN
              call tea_leaf_kernel_cheby_init_ocl(ch_alphas, ch_betas, &
                max_cheby_iters, rx, ry, theta, error)
            ENDIF
            call clover_allsum(error)

            ! FIXME not giving correct estimate
            it_alpha = eps*bb/(4.0_8*error)
            cn = eigmax/eigmin
            gamm = (sqrt(cn) - 1.0_8)/(sqrt(cn) + 1.0_8)
            est_itc = nint(log(it_alpha)/(2.0_8*log(gamm))) - n

            if (parallel%boss) then
              write(*,*) "switching after step", n
              write(*,*) "eigmin", eigmin
              write(*,*) "eigmax", eigmax
              write(*,*) "cn", cn
              write(*,*) "error", error
              write(*,*) "est itc", est_itc
            endif

            cheby_calc_steps = 2
          endif

          IF(use_fortran_kernels) THEN
              call tea_leaf_kernel_cheby_iterate(chunks(c)%field%x_min,&
                  chunks(c)%field%x_max,                       &
                  chunks(c)%field%y_min,                       &
                  chunks(c)%field%y_max,                       &
                  chunks(c)%field%u,                           &
                  chunks(c)%field%work_array1,                 &
                  chunks(c)%field%work_array2,                 &
                  chunks(c)%field%u0,                 &
                  chunks(c)%field%work_array5,                 &
                  chunks(c)%field%work_array6,                 &
                  chunks(c)%field%work_array7,                 &
                  ch_alphas, ch_betas, max_cheby_iters, &
                  rx, ry, cheby_calc_steps)
          ELSEIF(use_opencl_kernels) THEN
              call tea_leaf_kernel_cheby_iterate_ocl(ch_alphas, ch_betas, max_cheby_iters, &
                rx, ry, cheby_calc_steps)
          ENDIF

          ! after estimated number of iterations has passed, calc resid
          if (cheby_calc_steps .ge. est_itc) then
            IF(use_fortran_kernels) THEN
              call tea_leaf_calc_2norm_kernel(chunks(c)%field%x_min,        &
                    chunks(c)%field%x_max,                       &
                    chunks(c)%field%y_min,                       &
                    chunks(c)%field%y_max,                       &
                    chunks(c)%field%work_array2,                 &
                    error)
            ELSEIF(use_opencl_kernels) THEN
              call tea_leaf_calc_2norm_kernel_ocl(1, error)
            ENDIF
          else
            ! dummy to make it go smaller every time but not reach tolerance
            error = 1.0_8/(cheby_calc_steps)
          endif

          call clover_allsum(error)

          cheby_calc_steps = cheby_calc_steps + 1

        ELSEIF(tl_use_cg .or. tl_use_chebyshev) then
          fields(FIELD_P) = 1

          IF(use_fortran_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_fortran_calc_w(chunks(c)%field%x_min,&
                chunks(c)%field%x_max,                       &
                chunks(c)%field%y_min,                       &
                chunks(c)%field%y_max,                       &
                chunks(c)%field%work_array1,                 &
                chunks(c)%field%work_array4,                 &
                chunks(c)%field%work_array6,                 &
                chunks(c)%field%work_array7,                 &
                rx, ry, pw)
          ELSEIF(use_opencl_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_ocl_calc_w(rx, ry, pw)
          ELSEIF(use_c_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_c_calc_w(chunks(c)%field%x_min,&
                chunks(c)%field%x_max,                       &
                chunks(c)%field%y_min,                       &
                chunks(c)%field%y_max,                       &
                chunks(c)%field%work_array1,                 &
                chunks(c)%field%work_array4,                 &
                chunks(c)%field%work_array6,                 &
                chunks(c)%field%work_array7,                 &
                rx, ry, pw)
          ENDIF

          CALL clover_allsum(pw)
          alpha = rro/pw
          if(tl_use_chebyshev) cg_alphas(n) = alpha

          IF(use_fortran_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_fortran_calc_ur(chunks(c)%field%x_min,&
                chunks(c)%field%x_max,                       &
                chunks(c)%field%y_min,                       &
                chunks(c)%field%y_max,                       &
                chunks(c)%field%u,                           &
                chunks(c)%field%work_array1,                 &
                chunks(c)%field%work_array2,                 &
                chunks(c)%field%work_array3,                 &
                chunks(c)%field%work_array4,                 &
                chunks(c)%field%work_array5,                 &
                alpha, rrn)
          ELSEIF(use_opencl_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_ocl_calc_ur(alpha, rrn)
          ELSEIF(use_c_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_c_calc_ur(chunks(c)%field%x_min,&
                chunks(c)%field%x_max,                       &
                chunks(c)%field%y_min,                       &
                chunks(c)%field%y_max,                       &
                chunks(c)%field%u,                           &
                chunks(c)%field%work_array1,                 &
                chunks(c)%field%work_array2,                 &
                chunks(c)%field%work_array3,                 &
                chunks(c)%field%work_array4,                 &
                chunks(c)%field%work_array5,                 &
                alpha, rrn)
          ENDIF

          CALL clover_allsum(rrn)
          beta = rrn/rro
          if(tl_use_chebyshev) cg_betas(n) = beta

          IF(use_fortran_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_fortran_calc_p(chunks(c)%field%x_min,&
                chunks(c)%field%x_max,                       &
                chunks(c)%field%y_min,                       &
                chunks(c)%field%y_max,                       &
                chunks(c)%field%work_array1,                 &
                chunks(c)%field%work_array2,                 &
                chunks(c)%field%work_array5,                 &
                beta)
          ELSEIF(use_opencl_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_ocl_calc_p(beta)
          ELSEIF(use_c_kernels) THEN
            CALL tea_leaf_kernel_solve_cg_c_calc_p(chunks(c)%field%x_min,&
                chunks(c)%field%x_max,                       &
                chunks(c)%field%y_min,                       &
                chunks(c)%field%y_max,                       &
                chunks(c)%field%work_array1,                 &
                chunks(c)%field%work_array2,                 &
                chunks(c)%field%work_array5,                 &
                beta)
          ENDIF

          error = rrn
          rro = rrn
        ELSE
          IF(use_fortran_kernels) THEN
            CALL tea_leaf_kernel_solve(chunks(c)%field%x_min,&
                chunks(c)%field%x_max,                       &
                chunks(c)%field%y_min,                       &
                chunks(c)%field%y_max,                       &
                rx,                                          &
                ry,                                          &
                chunks(c)%field%work_array6,                 &
                chunks(c)%field%work_array7,                 &
                error,                                       &
                chunks(c)%field%work_array1,                 &
                chunks(c)%field%u,                           &
                chunks(c)%field%work_array2)
          ELSEIF(use_opencl_kernels) THEN
              CALL tea_leaf_kernel_solve_ocl(rx, ry, error)
          ELSEIF(use_C_kernels) THEN
              CALL tea_leaf_kernel_solve_c(chunks(c)%field%x_min,&
                  chunks(c)%field%x_max,                       &
                  chunks(c)%field%y_min,                       &
                  chunks(c)%field%y_max,                       &
                  rx,                                          &
                  ry,                                          &
                  chunks(c)%field%work_array6,                 &
                  chunks(c)%field%work_array7,                 &
                  error,                                       &
                  chunks(c)%field%work_array1,                 &
                  chunks(c)%field%u,                           &
                  chunks(c)%field%work_array2)
          ENDIF
        ENDIF

        ! updates u and possibly p
        CALL update_halo(fields,2)

        CALL clover_max(error)

        IF (abs(error) .LT. eps) EXIT

        ! if the error isn't getting any better, then exit - no point in going further
        !IF (abs(error - old_error) .LT. eps .or. (error .eq. old_error)) EXIT
        old_error = error

      ENDDO

      IF (parallel%boss) THEN
!$      IF(OMP_GET_THREAD_NUM().EQ.0) THEN
          WRITE(g_out,"('Conduction error ',e14.7)") error
          WRITE(g_out,"('Iteration count ',i8)") n-1
          WRITE(0,"('Conduction error ',e14.7)") error
          WRITE(0,"('Iteration count ', i8)") n-1
!$      ENDIF
      ENDIF

      if (tl_use_chebyshev) then
        write(0, "('Actually took ', i4)") cheby_calc_steps
      endif

      ! RESET
      IF(use_fortran_kernels) THEN
          CALL tea_leaf_kernel_finalise(chunks(c)%field%x_min, &
              chunks(c)%field%x_max,                           &
              chunks(c)%field%y_min,                           &
              chunks(c)%field%y_max,                           &
              chunks(c)%field%energy1,                         &
              chunks(c)%field%density1,                        &
              chunks(c)%field%u)
      ELSEIF(use_opencl_kernels) THEN
          CALL tea_leaf_kernel_finalise_ocl()
      ELSEIF(use_C_kernels) THEN
          CALL tea_leaf_kernel_finalise_c(chunks(c)%field%x_min, &
              chunks(c)%field%x_max,                           &
              chunks(c)%field%y_min,                           &
              chunks(c)%field%y_max,                           &
              chunks(c)%field%energy1,                         &
              chunks(c)%field%density1,                        &
              chunks(c)%field%u)
      ENDIF

      fields=0
      fields(FIELD_ENERGY1) = 1
      CALL update_halo(fields,1)

    ENDIF

  ENDDO
  IF(profiler_on) profiler%PdV=profiler%tea+(timer()-kernel_time)

END SUBROUTINE tea_leaf

END MODULE tea_leaf_module

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
    USE tea_leaf_kernel_ppcg_module
    USE tea_leaf_kernel_cheby_module
    USE update_halo_module

    IMPLICIT NONE

CONTAINS

    SUBROUTINE tea_leaf()

        IMPLICIT NONE

        !$ INTEGER :: OMP_GET_THREAD_NUM
        INTEGER :: c, n
        INTEGER :: fields(NUM_FIELDS)

        REAL(KIND=8) :: ry,rx, error, exact_error
        REAL(KIND=8) :: timer,halo_time,solve_time,init_time,reset_time
        REAL(KIND=8) :: rro, pw, rrn, alpha, beta
        REAL(KIND=8) :: eigmin, eigmax, theta, cn, ch_per_it, cg_per_it
        REAL(KIND=8) :: cg_time, ch_time, total_solve_time, iteration_time
        REAL(KIND=8), DIMENSION(max_iters) :: cg_alphas, cg_betas
        REAL(KIND=8), DIMENSION(max_iters) :: ch_alphas, ch_betas

        INTEGER :: cg_calc_steps
        INTEGER :: est_itc, cheby_calc_steps, max_cheby_iters, info, switch_step
        LOGICAL :: ch_switch_check

        time = 0.0_8
        cg_time = 0.0_8
        ch_time = 0.0_8
        cg_calc_steps = 0
        total_solve_time = 0.0_8
        init_time = 0.0_8
        halo_time = 0.0_8
        solve_time = 0.0_8

        IF(coefficient .NE. RECIP_CONDUCTIVITY .AND. coefficient .NE. conductivity) THEN
            CALL report_error('tea_leaf', 'unknown coefficient option')
        ENDIF

        error = 1e10
        cheby_calc_steps = 0
        cg_calc_steps = 0

        total_solve_time = timer()

        DO c=1,chunks_per_task

            IF(chunks(c)%task.EQ.parallel%task) THEN

                ! INIT

                IF (profiler_on) halo_time=timer()
                fields=0
                fields(FIELD_ENERGY1) = 1
                fields(FIELD_DENSITY) = 1
                CALL update_halo(fields,2)

                IF (profiler_on) profiler%halo_exchange = profiler%halo_exchange + &
                    (timer() - halo_time)
                IF (profiler_on) init_time=timer()

                IF (use_fortran_kernels) THEN
                    rx = dt/(chunks(c)%field%celldx(chunks(c)%field%x_min)**2)
                    ry = dt/(chunks(c)%field%celldy(chunks(c)%field%y_min)**2)
                ENDIF

                IF(tl_use_cg .OR. tl_use_chebyshev .OR. tl_use_ppcg) THEN
                    ! All 3 of these solvers use the CG kernels
                    IF(use_ext_kernels) THEN
                        CALL ext_cg_solver_init(c, coefficient, tl_preconditioner_on,&
                            dt, rx, ry, rro)
                    ELSEIF(use_fortran_kernels) THEN
                        CALL tea_leaf_kernel_init_cg_fortran(&
                            chunks(c)%field%x_min,      &
                            chunks(c)%field%x_max,      &
                            chunks(c)%field%y_min,      &
                            chunks(c)%field%y_max,      &
                            chunks(c)%field%density,    &
                            chunks(c)%field%energy1,    &
                            chunks(c)%field%u,          &
                            chunks(c)%field%vector_p,   &
                            chunks(c)%field%vector_r,   &
                            chunks(c)%field%vector_Mi,  &
                            chunks(c)%field%vector_w,   &
                            chunks(c)%field%vector_z,   &
                            chunks(c)%field%vector_Kx,  &
                            chunks(c)%field%vector_Ky,  &
                            rx, ry, rro, coefficient, tl_preconditioner_on)
                    ENDIF

                    ! need to update p when using CG due to matrix/vector multiplication
                    fields=0
                    fields(FIELD_U) = 1
                    fields(FIELD_P) = 1
                    IF (profiler_on) halo_time=timer()
                    CALL update_halo(fields,1)
                    IF (profiler_on) profiler%halo_exchange = profiler%halo_exchange + &
                        (timer() - halo_time)
                    init_time=init_time+(timer()-halo_time)

                    ! and globally sum rro
                    CALL tea_allsum(rro)
                ELSEIF(tl_use_jacobi) THEN
                    IF(use_ext_kernels) THEN
                        CALL ext_jacobi_kernel_init(c, coefficient, dt, rx, ry)
                    ELSEIF (use_fortran_kernels) THEN
                        CALL tea_leaf_kernel_init(&
                            chunks(c)%field%x_min,      &
                            chunks(c)%field%x_max,      &
                            chunks(c)%field%y_min,      &
                            chunks(c)%field%y_max,      &
                            chunks(c)%field%density,    &
                            chunks(c)%field%energy1,    &
                            chunks(c)%field%u0,         &
                            chunks(c)%field%u,          &
                            chunks(c)%field%vector_r,   &
                            chunks(c)%field%vector_Kx,  &
                            chunks(c)%field%vector_Ky,  &
                            coefficient)
                    ENDIF

                    fields=0
                    fields(FIELD_U) = 1
                ENDIF

                ! need the original value of u
                IF(use_ext_kernels) THEN
                    call ext_solver_copy_u(c)
                ELSEIF(use_fortran_kernels) THEN
                    call tea_leaf_kernel_cheby_copy_u(&
                        chunks(c)%field%x_min,        &
                        chunks(c)%field%x_max,        &
                        chunks(c)%field%y_min,        &
                        chunks(c)%field%y_max,        &
                        chunks(c)%field%u0,           &
                        chunks(c)%field%u)
                ENDIF

                IF (profiler_on) profiler%tea_init = profiler%tea_init + (timer() - init_time)
                IF (profiler_on) solve_time = timer()

                DO n=1,max_iters

                    iteration_time = timer()

                    IF (tl_ch_cg_errswitch) THEN
                        ! either the error has got below tolerance, or it's already going 
                        ! minimum 20 steps to converge eigenvalues
                        ch_switch_check = (cheby_calc_steps .GT. 0) .OR. &
                            (error .LE. tl_ch_cg_epslim) .AND. (n .GE. 20)
                    ELSE
                        ! enough steps have passed and error < 1 
                        ! otherwise it's nowhere near converging on eigenvalues
                        ch_switch_check = (n .GE. tl_ch_cg_presteps) .AND. (error .le. 1.0_8)
                    ENDIF

                    IF ((tl_use_chebyshev .OR. tl_use_ppcg) .AND. ch_switch_check) THEN
                        ! on the first chebyshev steps, find the eigenvalues, coefficients,
                        ! and expected number of iterations
                        IF (cheby_calc_steps .EQ. 0) THEN
                            ! maximum number of iterations in chebyshev solver
                            max_cheby_iters = max_iters - n + 2
                            rro = error

                            ! calculate eigenvalues
                            CALL tea_calc_eigenvalues(cg_alphas, cg_betas, eigmin, eigmax, &
                                max_iters, n-1, info)

                            IF (info .NE. 0) CALL report_error('tea_leaf', 'Error in calculating eigenvalues')

                            eigmin = eigmin*0.95
                            eigmax = eigmax*1.05

                            IF (tl_use_chebyshev) THEN
                                ! calculate chebyshev coefficients
                                CALL tea_calc_ch_coefs(ch_alphas, ch_betas, eigmin, eigmax, &
                                    theta, max_cheby_iters)

                                ! don't need to update p any more
                                fields = 0
                                fields(FIELD_U) = 1
                            ELSE IF (tl_use_ppcg) THEN
                                ! currently also calculate chebyshev coefficients
                                ! TODO least squares
                                CALL tea_calc_ch_coefs(ch_alphas, ch_betas, eigmin, eigmax, &
                                    theta, tl_ppcg_inner_steps)
                            ENDIF

                            cn = eigmax/eigmin

                            IF (parallel%boss) THEN
                                WRITE(g_out,'(a,i3,a,e15.7)') "Switching after ",n," steps, error ",rro
                                WRITE(g_out,'(4a11)')"eigmin", "eigmax", "cn", "error"
                                WRITE(g_out,'(2f11.5,2e11.4)')eigmin, eigmax, cn, error
                                WRITE(0,'(a,i3,a,e15.7)') "Switching after ",n," steps, error ",rro
                                WRITE(0,'(4a11)')"eigmin", "eigmax", "cn", "error"
                                WRITE(0,'(2f11.5,2e11.4)')eigmin, eigmax, cn, error
                            ENDIF
                        ENDIF

                        IF (tl_use_chebyshev) THEN
                            IF (cheby_calc_steps .EQ. 0) THEN
                                CALL tea_leaf_cheby_first_step(c, ch_alphas, ch_betas, fields, &
                                    error, rx, ry, theta, cn, max_cheby_iters, est_itc)

                                cheby_calc_steps = 1

                                switch_step = n
                            ELSE
                                IF(use_ext_kernels) THEN
                                    CALL ext_cheby_solver_iterate(c, cheby_calc_steps)
                                ELSE IF(use_fortran_kernels) THEN
                                    call tea_leaf_kernel_cheby_iterate(chunks(c)%field%x_min,&
                                        chunks(c)%field%x_max,                       &
                                        chunks(c)%field%y_min,                       &
                                        chunks(c)%field%y_max,                       &
                                        chunks(c)%field%u,                           &
                                        chunks(c)%field%u0,                          &
                                        chunks(c)%field%vector_p,                 &
                                        chunks(c)%field%vector_r,                 &
                                        chunks(c)%field%vector_Mi,                 &
                                        chunks(c)%field%vector_w,                 &
                                        chunks(c)%field%vector_z,                 &
                                        chunks(c)%field%vector_Kx,                 &
                                        chunks(c)%field%vector_Ky,                 &
                                        ch_alphas, ch_betas, max_cheby_iters,        &
                                        rx, ry, cheby_calc_steps, tl_preconditioner_on)
                                ENDIF

                                ! after estimated number of iterations has passed, calc resid.
                                ! Leaving 10 iterations between each global reduction won't affect
                                ! total time spent much if at all (number of steps spent in
                                ! chebyshev is typically O(300+)) but will greatly reduce global
                                ! synchronisations needed
                                IF ((n .GE. est_itc) .AND. (MOD(n, 10) .eq. 0)) THEN
                                    IF(use_ext_kernels) THEN
                                        CALL ext_calculate_2norm(c, 1, error)
                                    ELSE IF(use_fortran_kernels) THEN
                                        call tea_leaf_calc_2norm_kernel(chunks(c)%field%x_min,        &
                                            chunks(c)%field%x_max,                       &
                                            chunks(c)%field%y_min,                       &
                                            chunks(c)%field%y_max,                       &
                                            chunks(c)%field%vector_r,                 &
                                            error)
                                    ENDIF

                                    CALL tea_allsum(error)
                                ENDIF
                            ENDIF
                        ELSE IF (tl_use_ppcg) THEN
                            IF (cheby_calc_steps .EQ. 0) THEN
                                cheby_calc_steps = 1

                                IF(use_ext_kernels) THEN
                                    CALL ext_ppcg_init(&
                                        c, tl_preconditioner_on, ch_alphas, ch_betas, tl_ppcg_inner_steps)
                                ENDIF

                                IF(use_ext_kernels) THEN
                                    CALL ext_calculate_residual(c)
                                ELSE IF(use_fortran_kernels) THEN
                                    CALL tea_leaf_calc_residual(chunks(c)%field%x_min,&
                                        chunks(c)%field%x_max,                       &
                                        chunks(c)%field%y_min,                       &
                                        chunks(c)%field%y_max,                       &
                                        chunks(c)%field%u,                           &
                                        chunks(c)%field%u0,                 &
                                        chunks(c)%field%vector_r,                 &
                                        chunks(c)%field%vector_Kx,                 &
                                        chunks(c)%field%vector_Ky,                 &
                                        rx, ry)
                                ENDIF

                                IF (profiler_on) halo_time = timer()
                                ! update p
                                CALL update_halo(fields,1)
                                IF (profiler_on) profiler%halo_exchange = profiler%halo_exchange + (timer() - halo_time)
                                IF (profiler_on) solve_time = solve_time + (timer()-halo_time)

                                CALL tea_allsum(rro)
                            ENDIF

                            IF(use_ext_kernels) THEN
                                CALL ext_cg_calc_w(c, pw)
                            ELSE IF(use_fortran_kernels) THEN
                                CALL tea_leaf_kernel_solve_cg_fortran_calc_w(chunks(c)%field%x_min,&
                                    chunks(c)%field%x_max,                       &
                                    chunks(c)%field%y_min,                       &
                                    chunks(c)%field%y_max,                       &
                                    chunks(c)%field%vector_p,                 &
                                    chunks(c)%field%vector_w,                 &
                                    chunks(c)%field%vector_Kx,                 &
                                    chunks(c)%field%vector_Ky,                 &
                                    rx, ry, pw)
                            ENDIF

                            CALL tea_allsum(pw)
                            alpha = rro/pw

                            IF(use_ext_kernels) THEN
                                CALL ext_cg_calc_ur(c, alpha, rrn)
                            ELSE IF(use_fortran_kernels) THEN
                                call tea_leaf_kernel_solve_cg_fortran_calc_ur(chunks(c)%field%x_min,&
                                    chunks(c)%field%x_max,                       &
                                    chunks(c)%field%y_min,                       &
                                    chunks(c)%field%y_max,                       &
                                    chunks(c)%field%u,                           &
                                    chunks(c)%field%vector_p,                 &
                                    chunks(c)%field%vector_r,                 &
                                    chunks(c)%field%vector_Mi,                 &
                                    chunks(c)%field%vector_w,                 &
                                    chunks(c)%field%vector_z,                 &
                                    alpha, rrn, tl_preconditioner_on)
                            ENDIF

                            ! not using rrn, so don't do a tea_allsum
                            CALL tea_leaf_run_ppcg_inner_steps(ch_alphas, ch_betas, theta, &
                                rx, ry, tl_ppcg_inner_steps, c)

                            IF(use_ext_kernels) THEN
                                CALL ext_calculate_2norm(c, 1, rrn)
                            ELSE IF(use_fortran_kernels) THEN
                                call tea_leaf_calc_2norm_kernel(chunks(c)%field%x_min,        &
                                    chunks(c)%field%x_max,                       &
                                    chunks(c)%field%y_min,                       &
                                    chunks(c)%field%y_max,                       &
                                    chunks(c)%field%vector_r,                 &
                                    rrn)
                            ENDIF

                            CALL tea_allsum(rrn)

                            beta = rrn/rro

                            IF(use_ext_kernels) THEN
                                CALL ext_cg_calc_p(c, beta)
                            ELSE IF(use_fortran_kernels) THEN
                                CALL tea_leaf_kernel_solve_cg_fortran_calc_p(chunks(c)%field%x_min,&
                                    chunks(c)%field%x_max,                       &
                                    chunks(c)%field%y_min,                       &
                                    chunks(c)%field%y_max,                       &
                                    chunks(c)%field%vector_p,                 &
                                    chunks(c)%field%vector_r,                 &
                                    chunks(c)%field%vector_z,                 &
                                    beta, tl_preconditioner_on)
                            ENDIF

                            error = rrn
                            rro = rrn
                        ENDIF

                        cheby_calc_steps = cheby_calc_steps + 1
                    ELSEIF(tl_use_cg .OR. tl_use_chebyshev .OR. tl_use_ppcg) THEN
                        fields(FIELD_P) = 1
                        cg_calc_steps = cg_calc_steps + 1

                        IF(use_ext_kernels) THEN
                            CALL ext_cg_calc_w(c, pw)
                        ELSE IF(use_fortran_kernels) THEN
                            CALL tea_leaf_kernel_solve_cg_fortran_calc_w(chunks(c)%field%x_min,&
                                chunks(c)%field%x_max,                       &
                                chunks(c)%field%y_min,                       &
                                chunks(c)%field%y_max,                       &
                                chunks(c)%field%vector_p,                 &
                                chunks(c)%field%vector_w,                 &
                                chunks(c)%field%vector_Kx,                 &
                                chunks(c)%field%vector_Ky,                 &
                                rx, ry, pw)
                        ENDIF

                        CALL tea_allsum(pw)
                        alpha = rro/pw
                        IF(tl_use_chebyshev .OR. tl_use_ppcg) cg_alphas(n) = alpha

                        IF(use_ext_kernels) THEN
                            CALL ext_cg_calc_ur(c, alpha, rrn) 
                        ELSE IF(use_fortran_kernels) THEN
                            CALL tea_leaf_kernel_solve_cg_fortran_calc_ur(chunks(c)%field%x_min,&
                                chunks(c)%field%x_max,                       &
                                chunks(c)%field%y_min,                       &
                                chunks(c)%field%y_max,                       &
                                chunks(c)%field%u,                           &
                                chunks(c)%field%vector_p,                 &
                                chunks(c)%field%vector_r,                 &
                                chunks(c)%field%vector_Mi,                 &
                                chunks(c)%field%vector_w,                 &
                                chunks(c)%field%vector_z,                 &
                                alpha, rrn, tl_preconditioner_on)
                        ENDIF

                        CALL tea_allsum(rrn)
                        beta = rrn/rro
                        IF(tl_use_chebyshev .OR. tl_use_ppcg) cg_betas(n) = beta

                        IF(use_ext_kernels) THEN
                            CALL ext_cg_calc_p(c, beta)
                        ELSE IF(use_fortran_kernels) THEN
                            CALL tea_leaf_kernel_solve_cg_fortran_calc_p(chunks(c)%field%x_min,&
                                chunks(c)%field%x_max,                       &
                                chunks(c)%field%y_min,                       &
                                chunks(c)%field%y_max,                       &
                                chunks(c)%field%vector_p,                 &
                                chunks(c)%field%vector_r,                 &
                                chunks(c)%field%vector_z,                 &
                                beta, tl_preconditioner_on)
                        ENDIF

                        error = rrn
                        rro = rrn
                    ELSEIF(tl_use_jacobi) THEN
                        IF(use_ext_kernels) THEN
                            CALL ext_jacobi_kernel_solve(c, error)
                        ELSEIF(use_fortran_kernels) THEN
                            CALL tea_leaf_kernel_solve(chunks(c)%field%x_min,&
                                chunks(c)%field%x_max,                       &
                                chunks(c)%field%y_min,                       &
                                chunks(c)%field%y_max,                       &
                                rx,                                          &
                                ry,                                          &
                                chunks(c)%field%vector_Kx,                 &
                                chunks(c)%field%vector_Ky,                 &
                                error, &
                                chunks(c)%field%u0,                          &
                                chunks(c)%field%u,                           &
                                chunks(c)%field%vector_r)
                        ENDIF

                        ! error for jacobi is calculated recursively and is not very accurate,
                        ! so do this every so often to see whether it has actually converged
                        IF (mod(n, 50) .eq. 0) THEN
                            CALL update_halo(fields,1)

                            IF(use_ext_kernels) THEN
                                CALL ext_calculate_residual(c)
                                CALL ext_calculate_2norm(c, 1, error)
                            ELSEIF(use_fortran_kernels) THEN
                                CALL tea_leaf_calc_residual(chunks(c)%field%x_min,&
                                    chunks(c)%field%x_max,                       &
                                    chunks(c)%field%y_min,                       &
                                    chunks(c)%field%y_max,                       &
                                    chunks(c)%field%u,                           &
                                    chunks(c)%field%u0,                 &
                                    chunks(c)%field%vector_r,                 &
                                    chunks(c)%field%vector_Kx,                 &
                                    chunks(c)%field%vector_Ky,                 &
                                    rx, ry)
                                call tea_leaf_calc_2norm_kernel(chunks(c)%field%x_min,        &
                                    chunks(c)%field%x_max,                       &
                                    chunks(c)%field%y_min,                       &
                                    chunks(c)%field%y_max,                       &
                                    chunks(c)%field%vector_r,                 &
                                    error)
                            ENDIF
                        ENDIF

                        CALL tea_allsum(error)
                    ENDIF

                    ! updates u and possibly p
                    IF (profiler_on) halo_time = timer()
                    CALL update_halo(fields,1)
                    IF (profiler_on) profiler%halo_exchange = profiler%halo_exchange + (timer() - halo_time)
                    IF (profiler_on) solve_time = solve_time + (timer()-halo_time)

                    IF (profiler_on) THEN
                        IF (tl_use_chebyshev .AND. ch_switch_check) THEN
                            ch_time=ch_time+(timer()-iteration_time)
                        ELSE
                            cg_time=cg_time+(timer()-iteration_time)
                        ENDIF
                    ENDIF

                    IF (abs(error) .LT. eps) EXIT

                ENDDO

                IF (tl_check_result) THEN
                    IF(use_ext_kernels) THEN
                        CALL ext_calculate_residual(c)
                        CALL ext_calculate_2norm(c, 1, exact_error)
                    ELSEIF(use_fortran_kernels) THEN
                        CALL tea_leaf_calc_residual(chunks(c)%field%x_min,&
                            chunks(c)%field%x_max,                       &
                            chunks(c)%field%y_min,                       &
                            chunks(c)%field%y_max,                       &
                            chunks(c)%field%u,                           &
                            chunks(c)%field%u0,                 &
                            chunks(c)%field%vector_r,                 &
                            chunks(c)%field%vector_Kx,                 &
                            chunks(c)%field%vector_Ky,                 &
                            rx, ry)
                        call tea_leaf_calc_2norm_kernel(chunks(c)%field%x_min,        &
                            chunks(c)%field%x_max,                       &
                            chunks(c)%field%y_min,                       &
                            chunks(c)%field%y_max,                       &
                            chunks(c)%field%vector_r,                 &
                            exact_error)
                    ENDIF

                    CALL tea_allsum(exact_error)
                ENDIF

                IF (profiler_on) profiler%tea_solve = profiler%tea_solve + (timer() - solve_time)

                IF (parallel%boss) THEN
                    !$      IF(OMP_GET_THREAD_NUM().EQ.0) THEN
                    WRITE(g_out,"('Conduction error ',e14.7)") error
                    WRITE(0,"('Conduction error ',e14.7)") error

                    IF (tl_check_result) THEN
                        WRITE(0,"('EXACT error calculated as', e14.7)") exact_error
                        WRITE(g_out,"('EXACT error calculated as', e14.7)") exact_error
                    ENDIF

                    WRITE(g_out,"('Iteration count ',i8)") n-1
                    WRITE(0,"('Iteration count ', i8)") n-1
                    !$      ENDIF
                ENDIF

                ! RESET
                reset_time=timer()
                IF(use_ext_kernels) THEN
                    CALL ext_solver_finalise(c)
                ELSEIF(use_fortran_kernels) THEN
                    CALL tea_leaf_kernel_finalise(chunks(c)%field%x_min, &
                        chunks(c)%field%x_max,                           &
                        chunks(c)%field%y_min,                           &
                        chunks(c)%field%y_max,                           &
                        chunks(c)%field%energy1,                         &
                        chunks(c)%field%density,                        &
                        chunks(c)%field%u)
                ENDIF
                IF (profiler_on) profiler%tea_reset = profiler%tea_reset + (timer() - halo_time) 
                halo_time=timer()
                fields=0
                fields(FIELD_ENERGY1) = 1
                CALL update_halo(fields,1)
                IF (profiler_on) profiler%halo_exchange = profiler%halo_exchange + (timer() - halo_time)

            ENDIF

        ENDDO

        IF (profiler_on .AND. parallel%boss) THEN
            total_solve_time = (timer() - total_solve_time)
            WRITE(0, "(a16, a7, a16)") "Time", "Steps", "Per it"
            WRITE(0, "(f16.10, i7, f16.10, f7.2)") total_solve_time, n, total_solve_time/n
            WRITE(g_out, "(a16, a7, a16)") "Time", "Steps", "Per it"
            WRITE(g_out, "(f16.10, i7, f16.10, f7.2)") total_solve_time, n, total_solve_time/n
        ENDIF

        IF (profiler_on .AND. tl_use_chebyshev) THEN
            CALL tea_sum(ch_time)
            CALL tea_sum(cg_time)
            IF (parallel%boss) THEN
                cg_per_it = MERGE((cg_time/cg_calc_steps)/parallel%max_task, 0.0_8, cg_calc_steps .GT. 0)
                ch_per_it = MERGE((ch_time/cheby_calc_steps)/parallel%max_task, 0.0_8, cheby_calc_steps .GT. 0)

                WRITE(0, "(a3, a16, a7, a16, a7)") "", "Time", "Steps", "Per it", "Ratio"
                WRITE(0, "(a3, f16.10, i7, f16.10, f7.2)") &
                    "CG", cg_time + 0.0_8, cg_calc_steps, cg_per_it, 1.0_8
                WRITE(0, "(a3, f16.10, i7, f16.10, f7.2)") "CH", ch_time + 0.0_8, cheby_calc_steps, &
                    ch_per_it, MERGE(ch_per_it/cg_per_it, 0.0_8, cheby_calc_steps .GT. 0)
                WRITE(0, "('Chebyshev actually took ', i6, ' (' i6, ' off guess)')") &
                    cheby_calc_steps, cheby_calc_steps-est_itc

                WRITE(g_out, "(a3, a16, a7, a16, a7)") "", "Time", "Steps", "Per it", "Ratio"
                WRITE(g_out, "(a3, f16.10, i7, f16.10, f7.2)") &
                    "CG", cg_time + 0.0_8, cg_calc_steps, cg_per_it, 1.0_8
                WRITE(g_out, "(a3, f16.10, i7, f16.10, f7.2)") "CH", ch_time + 0.0_8, cheby_calc_steps, &
                    ch_per_it, MERGE(ch_per_it/cg_per_it, 0.0_8, cheby_calc_steps .GT. 0)
                WRITE(g_out, "('Chebyshev actually took ', i6, ' (' i6, ' off guess)')") &
                    cheby_calc_steps, cheby_calc_steps-est_itc
            ENDIF
        ENDIF

    END SUBROUTINE tea_leaf

    SUBROUTINE tea_leaF_run_ppcg_inner_steps(ch_alphas, ch_betas, theta, &
            rx, ry, tl_ppcg_inner_steps, c)

        INTEGER :: fields(NUM_FIELDS)
        INTEGER :: c, tl_ppcg_inner_steps, ppcg_cur_step
        REAL(KIND=8) :: rx, ry, theta
        REAL(KIND=8), DIMENSION(max_iters) :: ch_alphas, ch_betas

        IF(use_ext_kernels) THEN
            CALL ext_ppcg_init_sd(c, theta)
        ELSE IF(use_fortran_kernels) THEN
            call tea_leaf_kernel_ppcg_init_sd(chunks(c)%field%x_min,&
                chunks(c)%field%x_max,                       &
                chunks(c)%field%y_min,                       &
                chunks(c)%field%y_max,                       &
                chunks(c)%field%vector_r,                 &
                chunks(c)%field%vector_sd,                 &
                theta)
        ENDIF

        fields = 0
        fields(FIELD_SD) = 1

        ! inner steps
        DO ppcg_cur_step=1,tl_ppcg_inner_steps
            CALL update_halo(fields,1)

            IF(use_ext_kernels) THEN
                CALL ext_ppcg_inner(c, ppcg_cur_step)
            ELSE IF(use_fortran_kernels) THEN
                call tea_leaf_kernel_ppcg_inner(chunks(c)%field%x_min,&
                    chunks(c)%field%x_max,                       &
                    chunks(c)%field%y_min,                       &
                    chunks(c)%field%y_max,                       &
                    ppcg_cur_step, &
                    ch_alphas, ch_betas, &
                    rx, ry, &
                    chunks(c)%field%u,                           &
                    chunks(c)%field%vector_r,                 &
                    chunks(c)%field%vector_Kx,                 &
                    chunks(c)%field%vector_Ky,                 &
                    chunks(c)%field%vector_sd)
            ENDIF
        ENDDO

        fields = 0
        fields(FIELD_P) = 1

    END SUBROUTINE tea_leaF_run_ppcg_inner_steps

    SUBROUTINE tea_leaf_cheby_first_step(c, ch_alphas, ch_betas, fields, &
            error, rx, ry, theta, cn, max_cheby_iters, est_itc)

        IMPLICIT NONE

        integer :: c, est_itc, max_cheby_iters
        integer, dimension(:) :: fields
        REAL(KIND=8) :: it_alpha, cn, gamm, bb, error, rx, ry, theta
        REAL(KIND=8), DIMENSION(:) :: ch_alphas, ch_betas

        ! calculate 2 norm of u0
        IF(use_ext_kernels) THEN
            CALL ext_calculate_2norm(c, 0, bb)
        ELSE IF(use_fortran_kernels) THEN
            call tea_leaf_calc_2norm_kernel(chunks(c)%field%x_min,        &
                chunks(c)%field%x_max,                       &
                chunks(c)%field%y_min,                       &
                chunks(c)%field%y_max,                       &
                chunks(c)%field%u0,                 &
                bb)
        ENDIF

        CALL tea_allsum(bb)

        ! initialise 'p' array
        IF(use_ext_kernels) THEN
            CALL ext_cheby_solver_init(&
                c, ch_alphas, ch_betas, max_cheby_iters, theta,&
                tl_preconditioner_on)
        ELSE IF(use_fortran_kernels) THEN
            call tea_leaf_kernel_cheby_init(chunks(c)%field%x_min,&
                chunks(c)%field%x_max,                       &
                chunks(c)%field%y_min,                       &
                chunks(c)%field%y_max,                       &
                chunks(c)%field%u,                           &
                chunks(c)%field%u0,                 &
                chunks(c)%field%vector_p,                 &
                chunks(c)%field%vector_r,                 &
                chunks(c)%field%vector_Mi,                 &
                chunks(c)%field%vector_w,                 &
                chunks(c)%field%vector_z,                 &
                chunks(c)%field%vector_Kx,                 &
                chunks(c)%field%vector_Ky,                 &
                ch_alphas, ch_betas, max_cheby_iters, &
                rx, ry, theta, tl_preconditioner_on)
        ENDIF

        CALL update_halo(fields,1)

        IF(use_ext_kernels) THEN
            CALL ext_cheby_solver_iterate(c, 1)
        ELSE IF(use_fortran_kernels) THEN
            call tea_leaf_kernel_cheby_iterate(chunks(c)%field%x_min,&
                chunks(c)%field%x_max,                       &
                chunks(c)%field%y_min,                       &
                chunks(c)%field%y_max,                       &
                chunks(c)%field%u,                           &
                chunks(c)%field%u0,                          &
                chunks(c)%field%vector_p,                 &
                chunks(c)%field%vector_r,                 &
                chunks(c)%field%vector_Mi,                 &
                chunks(c)%field%vector_w,                 &
                chunks(c)%field%vector_z,                 &
                chunks(c)%field%vector_Kx,                 &
                chunks(c)%field%vector_Ky,                 &
                ch_alphas, ch_betas, max_cheby_iters,        &
                rx, ry, 1, tl_preconditioner_on)
        ENDIF

        IF(use_ext_kernels) THEN
            CALL ext_calculate_2norm(c, 1, error)
        ELSE IF(use_fortran_kernels) THEN
            call tea_leaf_calc_2norm_kernel(chunks(c)%field%x_min,        &
                chunks(c)%field%x_max,                       &
                chunks(c)%field%y_min,                       &
                chunks(c)%field%y_max,                       &
                chunks(c)%field%vector_r,                 &
                error)
        ENDIF

        CALL tea_allsum(error)

        it_alpha = EPSILON(1.0_8)*bb/(4.0_8*error)
        gamm = (SQRT(cn) - 1.0_8)/(SQRT(cn) + 1.0_8)
        est_itc = NINT(LOG(it_alpha)/(2.0_8*LOG(gamm)))

        IF (parallel%boss) THEN
            WRITE(g_out,'(a11)')"est itc"
            WRITE(g_out,'(11i11)')est_itc
            WRITE(0,'(a11)')"est itc"
            WRITE(0,'(11i11)')est_itc
        ENDIF

    END SUBROUTINE tea_leaf_cheby_first_step

END MODULE tea_leaf_module

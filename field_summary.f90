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

!>  @brief Driver for the field summary kernels
!>  @author David Beckingsale, Wayne Gaudin
!>  @details The user specified field summary kernel is invoked here. A summation
!>  across all mesh chunks is then performed and the information outputed.
!>  If the run is a test problem, the final result is compared with the expected
!>  result and the difference output.
!>  Note the reference solution is the value returned from an Intel compiler with
!>  ieee options set on a single core run.

SUBROUTINE field_summary()

USE tea_module
USE field_summary_kernel_module

IMPLICIT NONE

INTEGER      :: c
REAL(KIND=8) :: vol,mass,ie,temp
REAL(KIND=8) :: kernel_time,timer
REAL(KIND=8) :: qa_diff

!$ INTEGER :: OMP_GET_THREAD_NUM

IF(parallel%boss)THEN
    WRITE(g_out,*)
    WRITE(g_out,*) 'Time ',time
    WRITE(g_out,'(a13,5a16)')'           ','Volume','Mass','Density'       &
    ,'Energy','U'
ENDIF

IF(profiler_on) kernel_time=timer()

IF(use_fortran_kernels)THEN
    DO c=1,chunks_per_task
    IF(chunks(c)%task.EQ.parallel%task) THEN
        CALL field_summary_kernel(chunks(c)%field%x_min,                   &
        chunks(c)%field%x_max,                   &
        chunks(c)%field%y_min,                   &
        chunks(c)%field%y_max,                   &
        chunks(c)%field%volume,                  &
        chunks(c)%field%density,                 &
        chunks(c)%field%energy0,                 &
        chunks(c)%field%u,                       &
        vol,mass,ie,temp                         )
    ENDIF
    ENDDO
    ELSEIF(use_ext_kernels) THEN
    DO c=1,chunks_per_task
    IF(chunks(c)%task.EQ.parallel%task) THEN
        CALL ext_field_summary_kernel(c, vol, mass, ie, temp)
    ENDIF
    ENDDO
ENDIF

! For mpi I need a reduction here
CALL tea_sum(vol)
CALL tea_sum(mass)
CALL tea_sum(ie)
CALL tea_sum(temp)
IF(profiler_on) profiler%summary=profiler%summary+(timer()-kernel_time)

IF(parallel%boss) THEN
    !$ IF(OMP_GET_THREAD_NUM().EQ.0) THEN
    WRITE(g_out,'(a6,i7,5e16.7)')' step:',step,vol,mass,mass/vol,ie,temp
    WRITE(g_out,*)
    !$ ENDIF
ENDIF

!Check if this is the final call and if it is a test problem, check the result.
IF(complete) THEN
    IF(parallel%boss) THEN
        !$ IF(OMP_GET_THREAD_NUM().EQ.0) THEN
        IF(test_problem.GE.1) THEN

            ! Note that the "correct" solution is with IEEE switched on, 1 task, 1 thread, Intel compiler on Ivy Bridge
            IF(test_problem.EQ.1) qa_diff=ABS((100.0_8*(temp/157.550841832793_8))-100.0_8)
            IF(test_problem.EQ.2) qa_diff=ABS((100.0_8*(temp/116.067951160930_8))-100.0_8)
            IF(test_problem.EQ.3) qa_diff=ABS((100.0_8*(temp/95.4865103390698_8))-100.0_8)
            IF(test_problem.EQ.4) qa_diff=ABS((100.0_8*(temp/166.838315378708_8))-100.0_8)
            IF(test_problem.EQ.5) qa_diff=ABS((100.0_8*(temp/116.482111627676_8))-100.0_8)
            IF(test_problem.EQ.6) qa_diff=ABS((100.0_8*(temp/103.88639125996923_8))-100.0_8) ! 500x500 20 steps

            ! Test cases calculated without IEEE compliance

            ! 32x32
            IF(test_problem.EQ.7) qa_diff=ABS((100.0_8*(temp/177.00175203427580_8))-100.0_8)

            ! 64x64
            IF(test_problem.EQ.8) qa_diff=ABS((100.0_8*(temp/139.97102840522848_8))-100.0_8)

            ! 128x128
            IF(test_problem.EQ.9) qa_diff=ABS((100.0_8*(temp/118.33195538063963_8))-100.0_8)

            ! 256x256
            IF(test_problem.EQ.10) qa_diff=ABS((100.0_8*(temp/108.12687625327908_8))-100.0_8)

            ! 512x512
            IF(test_problem.EQ.11) qa_diff=ABS((100.0_8*(temp/103.46970921076068_8))-100.0_8)

            ! 1024x1024
            IF(test_problem.EQ.12) qa_diff=ABS((100.0_8*(temp/101.21009327612924_8))-100.0_8)

            ! 2048x2048
            IF(test_problem.EQ.13) qa_diff=ABS((100.0_8*(temp/100.16633923123922_8))-100.0_8)

            ! 4096x4096
            IF(test_problem.EQ.14) qa_diff=ABS((100.0_8*(temp/99.695059768963887_8))-100.0_8)

            ! 4000x4000
            IF(test_problem.EQ.15) qa_diff=ABS((100.0_8*(temp/99.756584330021866_8))-100.0_8)

            ! 316x316
            IF(test_problem.EQ.16) qa_diff=ABS((100.0_8*(temp/106.22536751207716_8))-100.0_8)

            ! 447x447
            IF(test_problem.EQ.17) qa_diff=ABS((100.0_8*(temp/103.75236286020100_8))-100.0_8)

            ! 548x548
            IF(test_problem.EQ.18) qa_diff=ABS((100.0_8*(temp/102.95322957172701_8))-100.0_8)

            ! 632x632
            IF(test_problem.EQ.19) qa_diff=ABS((100.0_8*(temp/102.61257379535277_8))-100.0_8)

            ! 707x707
            IF(test_problem.EQ.20) qa_diff=ABS((100.0_8*(temp/101.99792927438858_8))-100.0_8)

            ! 775x775
            IF(test_problem.EQ.21) qa_diff=ABS((100.0_8*(temp/101.93825682423312_8))-100.0_8)

            ! 837x837
            IF(test_problem.EQ.22) qa_diff=ABS((100.0_8*(temp/101.54636465546844_8))-100.0_8)

            ! 894x894
            IF(test_problem.EQ.23) qa_diff=ABS((100.0_8*(temp/101.51302189154072_8))-100.0_8)

            ! 949x949
            IF(test_problem.EQ.24) qa_diff=ABS((100.0_8*(temp/101.245719891081_8))-100.0_8)

            ! 1000x1000
            IF(test_problem.EQ.25) qa_diff=ABS((100.0_8*(temp/101.464437466587_8))-100.0_8)

            ! 1049x1049
            IF(test_problem.EQ.26) qa_diff=ABS((100.0_8*(temp/101.045345710857_8))-100.0_8)

            ! 1095x1095
            IF(test_problem.EQ.27) qa_diff=ABS((100.0_8*(temp/101.115716555869_8))-100.0_8)

            ! 1140x1140
            IF(test_problem.EQ.28) qa_diff=ABS((100.0_8*(temp/101.178529514050_8))-100.0_8)

            ! 1183x1183
            IF(test_problem.EQ.29) qa_diff=ABS((100.0_8*(temp/100.971769845516_8))-100.0_8)

            ! 1224x1224
            IF(test_problem.EQ.30) qa_diff=ABS((100.0_8*(temp/100.873263930372_8))-100.0_8)

            WRITE(*,'(a,i4,a,f16.7,a)')"Test problem", Test_problem," is within",qa_diff,"% of the expected solution"
            WRITE(g_out,'(a,i4,a,e16.7,a)')"Test problem", Test_problem," is within",qa_diff,"% of the expected solution"
            IF(qa_diff.LT.0.001) THEN
                WRITE(*,*)"This test is considered "//achar(27)//"[32m PASSED"//achar(27)//"[0m."
                WRITE(g_out,*)"This test is considered PASSED"
            ELSE
                WRITE(*,*)"This test is considered "//achar(27)//"[31m NOT PASSED"//achar(27)//"[0m."
                WRITE(g_out,*)"This test is considered NOT PASSED"
            ENDIF
        ENDIF
        !$ ENDIF
    ENDIF
ENDIF

END SUBROUTINE field_summary

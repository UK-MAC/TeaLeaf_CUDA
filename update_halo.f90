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

!>  @brief Driver for the halo updates
!>  @author David Beckingsale, Wayne Gaudin
!>  @details Invokes the kernels for the internal and external halo cells for
!>  the fields specified.

MODULE update_halo_module

  USE tea_module
  USE report_module

CONTAINS

SUBROUTINE update_halo(fields,depth)

  IMPLICIT NONE

  INTEGER :: fields(NUM_FIELDS),depth
  REAL(KIND=8) :: timer,halo_time

  IF (profiler_on) halo_time=timer()
  CALL tea_exchange(fields,depth)
  IF (profiler_on) profiler%halo_exchange = profiler%halo_exchange + (timer() - halo_time)

  CALL update_boundary(fields, depth)

  CALL update_tile_boundary(fields, depth)

END SUBROUTINE update_halo

SUBROUTINE update_boundary(fields,depth)

  IMPLICIT NONE

  INTEGER :: t,fields(NUM_FIELDS),depth
  REAL(KIND=8) :: timer,halo_time

  IF (profiler_on) halo_time=timer()

  IF (reflective_boundary .EQV. .TRUE. .AND. ANY(chunk%chunk_neighbours .EQ. EXTERNAL_FACE)) THEN
    IF (use_cuda_kernels)THEN
      DO t=1,tiles_per_task
        CALL update_halo_kernel_cuda(chunk%chunk_neighbours,     &
                                    fields,                         &
                                    depth                           )
      ENDDO
    ENDIF
  ENDIF

  IF (profiler_on) profiler%halo_update = profiler%halo_update + (timer() - halo_time)

END SUBROUTINE update_boundary

SUBROUTINE update_tile_boundary(fields, depth)

  IMPLICIT NONE

  INTEGER :: t,fields(NUM_FIELDS),depth, right_idx, up_idx
  REAL(KIND=8) :: timer,halo_time

  IF (profiler_on) halo_time=timer()

  IF (tiles_per_task .GT. 1) THEN
    CALL report_error("update_tile_boundary", "OpenCL (or CUDA) should never have >1 tile per task")
  ENDIF

  IF (profiler_on) profiler%internal_halo_update = profiler%internal_halo_update + (timer() - halo_time)

END SUBROUTINE update_tile_boundary

END MODULE update_halo_module

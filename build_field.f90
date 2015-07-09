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

!>  @brief  Allocates the data for each mesh chunk
!>  @author David Beckingsale, Wayne Gaudin
!>  @details The data fields for the mesh chunk are allocated based on the mesh
!>  size.

SUBROUTINE build_field()

  USE tea_module

  IMPLICIT NONE

  INTEGER :: j,k, t

  IF (use_cuda_kernels) THEN
    DO t=1,tiles_per_task
      chunk%tiles(t)%field%x_min=1
      chunk%tiles(t)%field%y_min=1

      chunk%tiles(t)%field%x_max=chunk%tiles(t)%x_cells
      chunk%tiles(t)%field%y_max=chunk%tiles(t)%y_cells

      CALL initialise_cuda(chunk%tiles(t)%field%x_min, &
                          chunk%tiles(t)%field%x_max, &
                          chunk%tiles(t)%field%y_min, &
                          chunk%tiles(t)%field%y_max)
    ENDDO
  ENDIF

END SUBROUTINE build_field


MODULE tea_leaf_common_module

  USE definitions_module

  IMPLICIT NONE

CONTAINS

SUBROUTINE tea_leaf_init_common()

  IMPLICIT NONE

  INTEGER :: t
  INTEGER :: zero_boundary(4)
  LOGICAL :: zero_boundary_mask(4)

  INTEGER :: reflective_boundary_int

  IF (reflective_boundary) THEN
    reflective_boundary_int = 1
  ELSE
    reflective_boundary_int = 0
  ENDIF

  IF (use_cuda_kernels) THEN
    DO t=1,tiles_per_task
      zero_boundary_mask = (chunk%chunk_neighbours .eq. external_face)

      ! CG never needs matrix defined outside of boundaries
      IF (tl_use_cg) THEN
        zero_boundary_mask = (chunk%tiles(t)%tile_neighbours .eq. external_face) .and. zero_boundary_mask
      ENDIF

      zero_boundary = 0

      WHERE (zero_boundary_mask .EQV. .TRUE.)
        zero_boundary = 1
      END WHERE

      CALL tea_leaf_common_init_kernel_cuda(coefficient, dt, &
        chunk%tiles(t)%field%rx, chunk%tiles(t)%field%ry, &
        zero_boundary, reflective_boundary_int)
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_init_common

SUBROUTINE tea_leaf_calc_residual()

  IMPLICIT NONE

  INTEGER :: t

  IF (use_cuda_kernels) THEN
    DO t=1,tiles_per_task
      CALL tea_leaf_calc_residual_cuda()
    ENDDO
  ENDIF

END SUBROUTINE

SUBROUTINE tea_leaf_calc_2norm(norm_array, norm)

  IMPLICIT NONE

  INTEGER :: t, norm_array
  REAL(KIND=8) :: norm, tile_norm

  norm = 0.0_8

  IF (use_cuda_kernels) THEN
    DO t=1,tiles_per_task
      tile_norm = 0.0_8

      CALL tea_leaf_calc_2norm_kernel_cuda(norm_array, tile_norm)

      norm = norm + tile_norm
    ENDDO
  ENDIF

END SUBROUTINE

SUBROUTINE tea_leaf_finalise()

  IMPLICIT NONE

  INTEGER :: t

  IF (use_cuda_kernels) THEN
    DO t=1,tiles_per_task
      CALL tea_leaf_common_finalise_kernel_cuda()
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_finalise

END MODULE


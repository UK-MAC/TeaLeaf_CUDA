
MODULE tea_leaf_jacobi_module

  USE definitions_module

  IMPLICIT NONE

CONTAINS

SUBROUTINE tea_leaf_jacobi_solve(error)

  IMPLICIT NONE

  INTEGER :: t
  REAL(KIND=8) :: error, tile_error

  error = 0.0_8

  IF (use_cuda_kernels) THEN
    DO t=1,tiles_per_task
      tile_error = 0.0_8

      CALL tea_leaf_jacobi_solve_kernel_cuda(tile_error)

      error = error + tile_error
    ENDDO
  ENDIF

END SUBROUTINE tea_leaf_jacobi_solve

END MODULE tea_leaf_jacobi_module


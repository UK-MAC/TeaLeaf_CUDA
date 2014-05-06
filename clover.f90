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

!>  @brief Communication Utilities
!>  @author David Beckingsale, Wayne Gaudin
!>  @details Contains all utilities required to run TeaLeaf in a distributed
!>  environment, including initialisation, mesh decompostion, reductions and
!>  halo exchange using explicit buffers.
!>
!>  Note the halo exchange is currently coded as simply as possible and no 
!>  optimisations have been implemented, such as post receives before sends or packing
!>  buffers with multiple data fields. This is intentional so the effect of these
!>  optimisations can be measured on large systems, as and when they are added.
!>
!>  Even without these modifications TeaLeaf weak scales well on moderately sized
!>  systems of the order of 10K cores.

MODULE clover_module

  USE data_module
  USE definitions_module
  !USE MPI

  IMPLICIT NONE
  include "mpif.h"

CONTAINS

SUBROUTINE clover_barrier

  INTEGER :: err

  CALL MPI_BARRIER(MPI_COMM_WORLD,err)

END SUBROUTINE clover_barrier

SUBROUTINE clover_abort

  INTEGER :: ierr,err

  CALL MPI_ABORT(MPI_COMM_WORLD,ierr,err)

END SUBROUTINE clover_abort

SUBROUTINE clover_finalize

  INTEGER :: err

  CLOSE(g_out)
  CALL FLUSH(0)
  CALL FLUSH(6)
  CALL FLUSH(g_out)
  CALL MPI_FINALIZE(err)

END SUBROUTINE clover_finalize

SUBROUTINE clover_init_comms

  IMPLICIT NONE

  INTEGER :: err,rank,size

  rank=0
  size=1

  CALL MPI_INIT(err) 

  CALL MPI_COMM_RANK(MPI_COMM_WORLD,rank,err) 
  CALL MPI_COMM_SIZE(MPI_COMM_WORLD,size,err) 

  parallel%parallel=.TRUE.
  parallel%task=rank

  IF(rank.EQ.0) THEN
    parallel%boss=.TRUE.
  ENDIF

  parallel%boss_task=0
  parallel%max_task=size

END SUBROUTINE clover_init_comms

SUBROUTINE clover_get_num_chunks(count)

  IMPLICIT NONE

  INTEGER :: count

! Should be changed so there can be more than one chunk per mpi task

  count=parallel%max_task

END SUBROUTINE clover_get_num_chunks

SUBROUTINE clover_decompose(x_cells,y_cells,left,right,bottom,top)

  ! This decomposes the mesh into a number of chunks.
  ! The number of chunks may be a multiple of the number of mpi tasks
  ! Doesn't always return the best split if there are few factors
  ! All factors need to be stored and the best picked. But its ok for now

  IMPLICIT NONE

  INTEGER :: x_cells,y_cells,left(:),right(:),top(:),bottom(:)
  INTEGER :: c,delta_x,delta_y

  REAL(KIND=8) :: mesh_ratio,factor_x,factor_y
  INTEGER  :: chunk_x,chunk_y,mod_x,mod_y,split_found

  INTEGER  :: cx,cy,chunk,add_x,add_y,add_x_prev,add_y_prev

  ! 2D Decomposition of the mesh

  mesh_ratio=real(x_cells)/real(y_cells)

  chunk_x=number_of_chunks
  chunk_y=1

  split_found=0 ! Used to detect 1D decomposition
  DO c=1,number_of_chunks
    IF (MOD(number_of_chunks,c).EQ.0) THEN
      factor_x=number_of_chunks/real(c)
      factor_y=c
      !Compare the factor ratio with the mesh ratio
      IF(factor_x/factor_y.LE.mesh_ratio) THEN
        chunk_y=c
        chunk_x=number_of_chunks/c
        split_found=1
        EXIT
      ENDIF
    ENDIF
  ENDDO

  IF(split_found.EQ.0.OR.chunk_y.EQ.number_of_chunks) THEN ! Prime number or 1D decomp detected
    IF(mesh_ratio.GE.1.0) THEN
      chunk_x=number_of_chunks
      chunk_y=1
    ELSE
      chunk_x=1
      chunk_y=number_of_chunks
    ENDIF
  ENDIF

  delta_x=x_cells/chunk_x
  delta_y=y_cells/chunk_y
  mod_x=MOD(x_cells,chunk_x)
  mod_y=MOD(y_cells,chunk_y)

  ! Set up chunk mesh ranges and chunk connectivity

  add_x_prev=0
  add_y_prev=0
  chunk=1
  DO cy=1,chunk_y
    DO cx=1,chunk_x
      add_x=0
      add_y=0
      IF(cx.LE.mod_x)add_x=1
      IF(cy.LE.mod_y)add_y=1
      left(chunk)=(cx-1)*delta_x+1+add_x_prev
      right(chunk)=left(chunk)+delta_x-1+add_x
      bottom(chunk)=(cy-1)*delta_y+1+add_y_prev
      top(chunk)=bottom(chunk)+delta_y-1+add_y
      chunks(chunk)%chunk_neighbours(chunk_left)=chunk_x*(cy-1)+cx-1
      chunks(chunk)%chunk_neighbours(chunk_right)=chunk_x*(cy-1)+cx+1
      chunks(chunk)%chunk_neighbours(chunk_bottom)=chunk_x*(cy-2)+cx
      chunks(chunk)%chunk_neighbours(chunk_top)=chunk_x*(cy)+cx
      IF(cx.EQ.1)chunks(chunk)%chunk_neighbours(chunk_left)=external_face
      IF(cx.EQ.chunk_x)chunks(chunk)%chunk_neighbours(chunk_right)=external_face
      IF(cy.EQ.1)chunks(chunk)%chunk_neighbours(chunk_bottom)=external_face
      IF(cy.EQ.chunk_y)chunks(chunk)%chunk_neighbours(chunk_top)=external_face
      IF(cx.LE.mod_x)add_x_prev=add_x_prev+1
      chunk=chunk+1
    ENDDO
    add_x_prev=0
    IF(cy.LE.mod_y)add_y_prev=add_y_prev+1
  ENDDO

  IF(parallel%boss)THEN
    WRITE(g_out,*)
    WRITE(g_out,*)"Mesh ratio of ",mesh_ratio
    WRITE(g_out,*)"Decomposing the mesh into ",chunk_x," by ",chunk_y," chunks"
    WRITE(g_out,*)
  ENDIF

END SUBROUTINE clover_decompose

SUBROUTINE clover_allocate_buffers(chunk)

  IMPLICIT NONE

  INTEGER      :: chunk
  
  ! Unallocated buffers for external boundaries caused issues on some systems so they are now
  !  all allocated
  IF(parallel%task.EQ.chunks(chunk)%task)THEN
    !IF(chunks(chunk)%chunk_neighbours(chunk_left).NE.external_face) THEN
      ALLOCATE(chunks(chunk)%left_snd_buffer(2*(chunks(chunk)%field%y_max+5)))
      ALLOCATE(chunks(chunk)%left_rcv_buffer(2*(chunks(chunk)%field%y_max+5)))
    !ENDIF
    !IF(chunks(chunk)%chunk_neighbours(chunk_right).NE.external_face) THEN
      ALLOCATE(chunks(chunk)%right_snd_buffer(2*(chunks(chunk)%field%y_max+5)))
      ALLOCATE(chunks(chunk)%right_rcv_buffer(2*(chunks(chunk)%field%y_max+5)))
    !ENDIF
    !IF(chunks(chunk)%chunk_neighbours(chunk_bottom).NE.external_face) THEN
      ALLOCATE(chunks(chunk)%bottom_snd_buffer(2*(chunks(chunk)%field%x_max+5)))
      ALLOCATE(chunks(chunk)%bottom_rcv_buffer(2*(chunks(chunk)%field%x_max+5)))
    !ENDIF
    !IF(chunks(chunk)%chunk_neighbours(chunk_top).NE.external_face) THEN
      ALLOCATE(chunks(chunk)%top_snd_buffer(2*(chunks(chunk)%field%x_max+5)))
      ALLOCATE(chunks(chunk)%top_rcv_buffer(2*(chunks(chunk)%field%x_max+5)))
    !ENDIF
  ENDIF

END SUBROUTINE clover_allocate_buffers

SUBROUTINE clover_exchange(fields,depth)

  IMPLICIT NONE

  INTEGER      :: fields(num_fields),depth

  ! Assuming 1 patch per task, this will be changed
  ! Also, not packing all fields for each communication, doing one at a time

  IF(fields(FIELD_P).EQ.1) THEN
    CALL clover_exchange_message(parallel%task+1,chunks(parallel%task+1)%field%work_array1,      &
                                 chunks(parallel%task+1)%left_snd_buffer,                      &
                                 chunks(parallel%task+1)%left_rcv_buffer,                      &
                                 chunks(parallel%task+1)%right_snd_buffer,                     &
                                 chunks(parallel%task+1)%right_rcv_buffer,                     &
                                 chunks(parallel%task+1)%bottom_snd_buffer,                    &
                                 chunks(parallel%task+1)%bottom_rcv_buffer,                    &
                                 chunks(parallel%task+1)%top_snd_buffer,                       &
                                 chunks(parallel%task+1)%top_rcv_buffer,                       &
                                 depth, &
                                 0, 0, &
                                 FIELD_P)
  ENDIF

  IF(fields(FIELD_U).EQ.1) THEN
    CALL clover_exchange_message(parallel%task+1,chunks(parallel%task+1)%field%u,      &
                                 chunks(parallel%task+1)%left_snd_buffer,                      &
                                 chunks(parallel%task+1)%left_rcv_buffer,                      &
                                 chunks(parallel%task+1)%right_snd_buffer,                     &
                                 chunks(parallel%task+1)%right_rcv_buffer,                     &
                                 chunks(parallel%task+1)%bottom_snd_buffer,                    &
                                 chunks(parallel%task+1)%bottom_rcv_buffer,                    &
                                 chunks(parallel%task+1)%top_snd_buffer,                       &
                                 chunks(parallel%task+1)%top_rcv_buffer,                       &
                                 depth, &
                                 0, 0, &
                                 FIELD_U)
  ENDIF

  IF(fields(FIELD_DENSITY0).EQ.1) THEN
    CALL clover_exchange_message(parallel%task+1,chunks(parallel%task+1)%field%density0,      &
                                 chunks(parallel%task+1)%left_snd_buffer,                      &
                                 chunks(parallel%task+1)%left_rcv_buffer,                      &
                                 chunks(parallel%task+1)%right_snd_buffer,                     &
                                 chunks(parallel%task+1)%right_rcv_buffer,                     &
                                 chunks(parallel%task+1)%bottom_snd_buffer,                    &
                                 chunks(parallel%task+1)%bottom_rcv_buffer,                    &
                                 chunks(parallel%task+1)%top_snd_buffer,                       &
                                 chunks(parallel%task+1)%top_rcv_buffer,                       &
                                 depth, &
                                 0, 0, &
                                 FIELD_DENSITY0)
  ENDIF

  IF(fields(FIELD_DENSITY1).EQ.1) THEN
    CALL clover_exchange_message(parallel%task+1,chunks(parallel%task+1)%field%density1,      &
                                 chunks(parallel%task+1)%left_snd_buffer,                      &
                                 chunks(parallel%task+1)%left_rcv_buffer,                      &
                                 chunks(parallel%task+1)%right_snd_buffer,                     &
                                 chunks(parallel%task+1)%right_rcv_buffer,                     &
                                 chunks(parallel%task+1)%bottom_snd_buffer,                    &
                                 chunks(parallel%task+1)%bottom_rcv_buffer,                    &
                                 chunks(parallel%task+1)%top_snd_buffer,                       &
                                 chunks(parallel%task+1)%top_rcv_buffer,                       &
                                 depth, &
                                 0, 0, &
                                 FIELD_density1)
  ENDIF

  IF(fields(FIELD_ENERGY0).EQ.1) THEN
    CALL clover_exchange_message(parallel%task+1,chunks(parallel%task+1)%field%energy0,       &
                                 chunks(parallel%task+1)%left_snd_buffer,                      &
                                 chunks(parallel%task+1)%left_rcv_buffer,                      &
                                 chunks(parallel%task+1)%right_snd_buffer,                     &
                                 chunks(parallel%task+1)%right_rcv_buffer,                     &
                                 chunks(parallel%task+1)%bottom_snd_buffer,                    &
                                 chunks(parallel%task+1)%bottom_rcv_buffer,                    &
                                 chunks(parallel%task+1)%top_snd_buffer,                       &
                                 chunks(parallel%task+1)%top_rcv_buffer,                       &
                                 depth, &
                                 0, 0, &
                                 FIELD_energy0)
  ENDIF

  IF(fields(FIELD_ENERGY1).EQ.1) THEN
    CALL clover_exchange_message(parallel%task+1,chunks(parallel%task+1)%field%energy1,       &
                                 chunks(parallel%task+1)%left_snd_buffer,                      &
                                 chunks(parallel%task+1)%left_rcv_buffer,                      &
                                 chunks(parallel%task+1)%right_snd_buffer,                     &
                                 chunks(parallel%task+1)%right_rcv_buffer,                     &
                                 chunks(parallel%task+1)%bottom_snd_buffer,                    &
                                 chunks(parallel%task+1)%bottom_rcv_buffer,                    &
                                 chunks(parallel%task+1)%top_snd_buffer,                       &
                                 chunks(parallel%task+1)%top_rcv_buffer,                       &
                                 depth, &
                                 0, 0, &
                                 FIELD_energy1)
  ENDIF

  IF(fields(FIELD_PRESSURE).EQ.1) THEN
    CALL clover_exchange_message(parallel%task+1,chunks(parallel%task+1)%field%pressure,      &
                                 chunks(parallel%task+1)%left_snd_buffer,                      &
                                 chunks(parallel%task+1)%left_rcv_buffer,                      &
                                 chunks(parallel%task+1)%right_snd_buffer,                     &
                                 chunks(parallel%task+1)%right_rcv_buffer,                     &
                                 chunks(parallel%task+1)%bottom_snd_buffer,                    &
                                 chunks(parallel%task+1)%bottom_rcv_buffer,                    &
                                 chunks(parallel%task+1)%top_snd_buffer,                       &
                                 chunks(parallel%task+1)%top_rcv_buffer,                       &
                                 depth, &
                                 0, 0, &
                                 FIELD_pressure)
  ENDIF

  IF(fields(FIELD_VISCOSITY).EQ.1) THEN
    CALL clover_exchange_message(parallel%task+1,chunks(parallel%task+1)%field%viscosity,     &
                                 chunks(parallel%task+1)%left_snd_buffer,                      &
                                 chunks(parallel%task+1)%left_rcv_buffer,                      &
                                 chunks(parallel%task+1)%right_snd_buffer,                     &
                                 chunks(parallel%task+1)%right_rcv_buffer,                     &
                                 chunks(parallel%task+1)%bottom_snd_buffer,                    &
                                 chunks(parallel%task+1)%bottom_rcv_buffer,                    &
                                 chunks(parallel%task+1)%top_snd_buffer,                       &
                                 chunks(parallel%task+1)%top_rcv_buffer,                       &
                                 depth, &
                                 0, 0, &
                                 FIELD_viscosity)
  ENDIF

  IF(fields(FIELD_SOUNDSPEED).EQ.1) THEN
    CALL clover_exchange_message(parallel%task+1,chunks(parallel%task+1)%field%soundspeed,    &
                                 chunks(parallel%task+1)%left_snd_buffer,                      &
                                 chunks(parallel%task+1)%left_rcv_buffer,                      &
                                 chunks(parallel%task+1)%right_snd_buffer,                     &
                                 chunks(parallel%task+1)%right_rcv_buffer,                     &
                                 chunks(parallel%task+1)%bottom_snd_buffer,                    &
                                 chunks(parallel%task+1)%bottom_rcv_buffer,                    &
                                 chunks(parallel%task+1)%top_snd_buffer,                       &
                                 chunks(parallel%task+1)%top_rcv_buffer,                       &
                                 depth, &
                                 0, 0, &
                                 FIELD_soundspeed)
  ENDIF

  IF(fields(FIELD_XVEL0).EQ.1) THEN
    CALL clover_exchange_message(parallel%task+1,chunks(parallel%task+1)%field%xvel0,         &
                                 chunks(parallel%task+1)%left_snd_buffer,                      &
                                 chunks(parallel%task+1)%left_rcv_buffer,                      &
                                 chunks(parallel%task+1)%right_snd_buffer,                     &
                                 chunks(parallel%task+1)%right_rcv_buffer,                     &
                                 chunks(parallel%task+1)%bottom_snd_buffer,                    &
                                 chunks(parallel%task+1)%bottom_rcv_buffer,                    &
                                 chunks(parallel%task+1)%top_snd_buffer,                       &
                                 chunks(parallel%task+1)%top_rcv_buffer,                       &
                                 depth, &
                                 1, 1, &
                                 FIELD_xvel0)
  ENDIF

  IF(fields(FIELD_XVEL1).EQ.1) THEN
    CALL clover_exchange_message(parallel%task+1,chunks(parallel%task+1)%field%xvel1,         &
                                 chunks(parallel%task+1)%left_snd_buffer,                      &
                                 chunks(parallel%task+1)%left_rcv_buffer,                      &
                                 chunks(parallel%task+1)%right_snd_buffer,                     &
                                 chunks(parallel%task+1)%right_rcv_buffer,                     &
                                 chunks(parallel%task+1)%bottom_snd_buffer,                    &
                                 chunks(parallel%task+1)%bottom_rcv_buffer,                    &
                                 chunks(parallel%task+1)%top_snd_buffer,                       &
                                 chunks(parallel%task+1)%top_rcv_buffer,                       &
                                 depth, &
                                 1, 1, &
                                 FIELD_xvel1)
  ENDIF

  IF(fields(FIELD_YVEL0).EQ.1) THEN
    CALL clover_exchange_message(parallel%task+1,chunks(parallel%task+1)%field%yvel0,         &
                                 chunks(parallel%task+1)%left_snd_buffer,                      &
                                 chunks(parallel%task+1)%left_rcv_buffer,                      &
                                 chunks(parallel%task+1)%right_snd_buffer,                     &
                                 chunks(parallel%task+1)%right_rcv_buffer,                     &
                                 chunks(parallel%task+1)%bottom_snd_buffer,                    &
                                 chunks(parallel%task+1)%bottom_rcv_buffer,                    &
                                 chunks(parallel%task+1)%top_snd_buffer,                       &
                                 chunks(parallel%task+1)%top_rcv_buffer,                       &
                                 depth, &
                                 1, 1, &
                                 FIELD_yvel0)
  ENDIF

  IF(fields(FIELD_YVEL1).EQ.1) THEN
    CALL clover_exchange_message(parallel%task+1,chunks(parallel%task+1)%field%yvel1,         &
                                 chunks(parallel%task+1)%left_snd_buffer,                      &
                                 chunks(parallel%task+1)%left_rcv_buffer,                      &
                                 chunks(parallel%task+1)%right_snd_buffer,                     &
                                 chunks(parallel%task+1)%right_rcv_buffer,                     &
                                 chunks(parallel%task+1)%bottom_snd_buffer,                    &
                                 chunks(parallel%task+1)%bottom_rcv_buffer,                    &
                                 chunks(parallel%task+1)%top_snd_buffer,                       &
                                 chunks(parallel%task+1)%top_rcv_buffer,                       &
                                 depth, &
                                 1, 1, &
                                 FIELD_yvel1)
  ENDIF

  IF(fields(FIELD_VOL_FLUX_X).EQ.1) THEN
    CALL clover_exchange_message(parallel%task+1,chunks(parallel%task+1)%field%vol_flux_x,    &
                                 chunks(parallel%task+1)%left_snd_buffer,                      &
                                 chunks(parallel%task+1)%left_rcv_buffer,                      &
                                 chunks(parallel%task+1)%right_snd_buffer,                     &
                                 chunks(parallel%task+1)%right_rcv_buffer,                     &
                                 chunks(parallel%task+1)%bottom_snd_buffer,                    &
                                 chunks(parallel%task+1)%bottom_rcv_buffer,                    &
                                 chunks(parallel%task+1)%top_snd_buffer,                       &
                                 chunks(parallel%task+1)%top_rcv_buffer,                       &
                                 depth, &
                                 1, 0, &
                                 FIELD_vol_flux_x)
  ENDIF

  IF(fields(FIELD_VOL_FLUX_Y).EQ.1) THEN
    CALL clover_exchange_message(parallel%task+1,chunks(parallel%task+1)%field%vol_flux_y,    &
                                 chunks(parallel%task+1)%left_snd_buffer,                      &
                                 chunks(parallel%task+1)%left_rcv_buffer,                      &
                                 chunks(parallel%task+1)%right_snd_buffer,                     &
                                 chunks(parallel%task+1)%right_rcv_buffer,                     &
                                 chunks(parallel%task+1)%bottom_snd_buffer,                    &
                                 chunks(parallel%task+1)%bottom_rcv_buffer,                    &
                                 chunks(parallel%task+1)%top_snd_buffer,                       &
                                 chunks(parallel%task+1)%top_rcv_buffer,                       &
                                 depth, &
                                 0, 1, &
                                 FIELD_vol_flux_y)
  ENDIF

  IF(fields(FIELD_MASS_FLUX_X).EQ.1) THEN
    CALL clover_exchange_message(parallel%task+1,chunks(parallel%task+1)%field%mass_flux_x,   &
                                 chunks(parallel%task+1)%left_snd_buffer,                      &
                                 chunks(parallel%task+1)%left_rcv_buffer,                      &
                                 chunks(parallel%task+1)%right_snd_buffer,                     &
                                 chunks(parallel%task+1)%right_rcv_buffer,                     &
                                 chunks(parallel%task+1)%bottom_snd_buffer,                    &
                                 chunks(parallel%task+1)%bottom_rcv_buffer,                    &
                                 chunks(parallel%task+1)%top_snd_buffer,                       &
                                 chunks(parallel%task+1)%top_rcv_buffer,                       &
                                 depth, &
                                 1, 0, &
                                 FIELD_mass_flux_x)
  ENDIF

  IF(fields(FIELD_MASS_FLUX_Y).EQ.1) THEN
    CALL clover_exchange_message(parallel%task+1,chunks(parallel%task+1)%field%mass_flux_y,   &
                                 chunks(parallel%task+1)%left_snd_buffer,                      &
                                 chunks(parallel%task+1)%left_rcv_buffer,                      &
                                 chunks(parallel%task+1)%right_snd_buffer,                     &
                                 chunks(parallel%task+1)%right_rcv_buffer,                     &
                                 chunks(parallel%task+1)%bottom_snd_buffer,                    &
                                 chunks(parallel%task+1)%bottom_rcv_buffer,                    &
                                 chunks(parallel%task+1)%top_snd_buffer,                       &
                                 chunks(parallel%task+1)%top_rcv_buffer,                       &
                                 depth, &
                                 0, 1, &
                                 FIELD_MASS_FLUX_Y)
  ENDIF


END SUBROUTINE clover_exchange

SUBROUTINE clover_exchange_message(chunk,field,                            &
                                   left_snd_buffer,                        &
                                   left_rcv_buffer,                        &
                                   right_snd_buffer,                       &
                                   right_rcv_buffer,                       &
                                   bottom_snd_buffer,                      &
                                   bottom_rcv_buffer,                      &
                                   top_snd_buffer,                         &
                                   top_rcv_buffer,                         &
                                   depth,                                  &
                                   x_inc, y_inc, &
                                   which_field)

  USE pack_kernel_module

  IMPLICIT NONE

  INTEGER      :: x_min, x_max, y_min, y_max, x_inc, y_inc


  INTEGER      :: chunk,depth,field_type

  REAL(KIND=8) :: field(chunks(chunk)%field%x_min-2:chunks(chunk)%field%x_max+2+x_inc, &
                        chunks(chunk)%field%y_min-2:chunks(chunk)%field%y_max+2+y_inc) ! This seems to work for any type of mesh data

  REAL(KIND=8), dimension(2*(chunks(chunk)%field%y_max+5)) :: &
    left_snd_buffer, left_rcv_buffer, right_snd_buffer, right_rcv_buffer
  REAL(KIND=8), dimension(2*(chunks(chunk)%field%x_max+5)) :: &
    bottom_snd_buffer, bottom_rcv_buffer, top_snd_buffer, top_rcv_buffer

  INTEGER      :: size,err,request(8),tag,message_count,j,k,index
  INTEGER      :: status(MPI_STATUS_SIZE,8)
  INTEGER      :: receiver,sender

  ! pass the field so it can be passed to ocl
  INTEGER :: which_field

  ! Field type will either be cell, vertex, x_face or y_face to get the message limits correct

  ! I am packing my own buffers. I am sure this could be improved with MPI data types
  !  but this will do for now

  ! I am also sending buffers to chunks with the same task id for now.
  ! This can be improved in the future but at the moment there is just 1 chunk per task anyway

  ! The tag will be a function of the sending chunk and the face it is coming from
  !  like chunk 6 sending the left face

  ! No open mp in here either. May be beneficial will packing and unpacking in the future, though I am not sure.

  ! Change this so it will allow more than 1 chunk per task

  request=0
  message_count=0

  ! Pack and send

  ! Pack real data into buffers
  IF(parallel%task.EQ.chunks(chunk)%task) THEN
    size=(1+(chunks(chunk)%field%y_max+y_inc+depth)-(chunks(chunk)%field%y_min-depth))*depth
    IF(use_fortran_kernels) THEN
      CALL pack_left_right_buffers(chunks(chunk)%field%x_min,chunks(chunk)%field%x_max, &
                                   chunks(chunk)%field%y_min,chunks(chunk)%field%y_max, &
                                   chunks(chunk)%chunk_neighbours(chunk_left),          &
                                   chunks(chunk)%chunk_neighbours(chunk_right),         &
                                   external_face,                                       &
                                   x_inc,y_inc,depth,size,                              &
                                   field,left_snd_buffer,right_snd_buffer)
    ELSEIF(use_opencl_kernels)THEN
      CALL pack_left_right_buffers_ocl(chunks(chunk)%field%x_min,chunks(chunk)%field%x_max, &
                                       chunks(chunk)%field%y_min,chunks(chunk)%field%y_max, &
                                       chunks(chunk)%chunk_neighbours(chunk_left),          &
                                       chunks(chunk)%chunk_neighbours(chunk_right),         &
                                       external_face,                                       &
                                       x_inc,y_inc,depth,                                   &
                                       which_field,                                         &
                                       field,left_snd_buffer,right_snd_buffer)
    ELSEIF(use_C_kernels)THEN
      CALL pack_left_right_buffers_c(chunks(chunk)%field%x_min,chunks(chunk)%field%x_max, &
                                     chunks(chunk)%field%y_min,chunks(chunk)%field%y_max, &
                                     chunks(chunk)%chunk_neighbours(chunk_left),          &
                                     chunks(chunk)%chunk_neighbours(chunk_right),         &
                                     external_face,                                       &
                                     x_inc,y_inc,depth,size,                              &
                                     field,left_snd_buffer,right_snd_buffer)
    ENDIF

    ! Send/receive the data
    IF(chunks(chunk)%chunk_neighbours(chunk_left).NE.external_face) THEN
      tag=4*(chunk)+1 ! 4 because we have 4 faces, 1 because it is leaving the left face
      receiver=chunks(chunks(chunk)%chunk_neighbours(chunk_left))%task
      CALL MPI_ISEND(left_snd_buffer,size,MPI_DOUBLE_PRECISION,receiver,tag &
                    ,MPI_COMM_WORLD,request(message_count+1),err)
      tag=4*(chunks(chunk)%chunk_neighbours(chunk_left))+2 ! 4 because we have 4 faces, 1 because it is coming from the right face of the left neighbour
      sender=chunks(chunks(chunk)%chunk_neighbours(chunk_left))%task
      CALL MPI_IRECV(left_rcv_buffer,size,MPI_DOUBLE_PRECISION,sender,tag &
                    ,MPI_COMM_WORLD,request(message_count+2),err)
      message_count=message_count+2
    ENDIF

    IF(chunks(chunk)%chunk_neighbours(chunk_right).NE.external_face) THEN
      tag=4*chunk+2 ! 4 because we have 4 faces, 2 because it is leaving the right face
      receiver=chunks(chunks(chunk)%chunk_neighbours(chunk_right))%task
      CALL MPI_ISEND(right_snd_buffer,size,MPI_DOUBLE_PRECISION,receiver,tag &
                    ,MPI_COMM_WORLD,request(message_count+1),err)
      tag=4*(chunks(chunk)%chunk_neighbours(chunk_right))+1 ! 4 because we have 4 faces, 1 because it is coming from the left face of the right neighbour
      sender=chunks(chunks(chunk)%chunk_neighbours(chunk_right))%task
      CALL MPI_IRECV(right_rcv_buffer,size,MPI_DOUBLE_PRECISION,sender,tag, &
                     MPI_COMM_WORLD,request(message_count+2),err)
      message_count=message_count+2
    ENDIF
  ENDIF

  ! Wait for the messages
  CALL MPI_WAITALL(message_count,request,status,err)

  ! Unpack buffers in halo cells
  IF(parallel%task.EQ.chunks(chunk)%task) THEN
    IF(use_fortran_kernels) THEN
      CALL unpack_left_right_buffers(chunks(chunk)%field%x_min,chunks(chunk)%field%x_max, &
                                     chunks(chunk)%field%y_min,chunks(chunk)%field%y_max, &
                                     chunks(chunk)%chunk_neighbours(chunk_left),          &
                                     chunks(chunk)%chunk_neighbours(chunk_right),         &
                                     external_face,                                       &
                                     x_inc,y_inc,depth,size,                              &
                                     field,left_rcv_buffer,right_rcv_buffer)
    ELSEIF(use_opencl_kernels)THEN
      CALL unpack_left_right_buffers_ocl(chunks(chunk)%field%x_min,chunks(chunk)%field%x_max, &
                                         chunks(chunk)%field%y_min,chunks(chunk)%field%y_max, &
                                         chunks(chunk)%chunk_neighbours(chunk_left),          &
                                         chunks(chunk)%chunk_neighbours(chunk_right),         &
                                         external_face,                                       &
                                         x_inc,y_inc,depth,                                   &
                                         which_field,                                         &
                                         field,left_rcv_buffer,right_rcv_buffer)
    ELSEIF(use_C_kernels)THEN
      CALL unpack_left_right_buffers_c(chunks(chunk)%field%x_min,chunks(chunk)%field%x_max, &
                                       chunks(chunk)%field%y_min,chunks(chunk)%field%y_max, &
                                       chunks(chunk)%chunk_neighbours(chunk_left),          &
                                       chunks(chunk)%chunk_neighbours(chunk_right),         &
                                       external_face,                                       &
                                       x_inc,y_inc,depth,size,                              &
                                       field,left_rcv_buffer,right_rcv_buffer)
    ENDIF
  ENDIF

  request=0
  message_count=0

  ! Pack real data into buffers
  IF(parallel%task.EQ.chunks(chunk)%task) THEN
    size=(1+(chunks(chunk)%field%x_max+x_inc+depth)-(chunks(chunk)%field%x_min-depth))*depth
    IF(use_fortran_kernels) THEN
      CALL pack_top_bottom_buffers(chunks(chunk)%field%x_min,chunks(chunk)%field%x_max, &
                                   chunks(chunk)%field%y_min,chunks(chunk)%field%y_max, &
                                   chunks(chunk)%chunk_neighbours(chunk_bottom),        &
                                   chunks(chunk)%chunk_neighbours(chunk_top),           &
                                   external_face,                                       &
                                   x_inc,y_inc,depth,size,                              &
                                   field,bottom_snd_buffer,top_snd_buffer)
    ELSEIF(use_opencl_kernels)THEN
      CALL pack_top_bottom_buffers_ocl(chunks(chunk)%field%x_min,chunks(chunk)%field%x_max, &
                                       chunks(chunk)%field%y_min,chunks(chunk)%field%y_max, &
                                       chunks(chunk)%chunk_neighbours(chunk_left),          &
                                       chunks(chunk)%chunk_neighbours(chunk_right),         &
                                       external_face,                                       &
                                       x_inc,y_inc,depth,                                   &
                                       which_field,                                         &
                                       field,left_snd_buffer,right_snd_buffer)
    ELSEIF(use_C_kernels)THEN
      CALL pack_top_bottom_buffers_c(chunks(chunk)%field%x_min,chunks(chunk)%field%x_max, &
                                     chunks(chunk)%field%y_min,chunks(chunk)%field%y_max, &
                                     chunks(chunk)%chunk_neighbours(chunk_bottom),        &
                                     chunks(chunk)%chunk_neighbours(chunk_top),           &
                                     external_face,                                       &
                                     x_inc,y_inc,depth,size,                              &
                                     field,bottom_snd_buffer,top_snd_buffer)
    ENDIF

    ! Send/receive the data
    IF(chunks(chunk)%chunk_neighbours(chunk_bottom).NE.external_face) THEN
      tag=4*(chunk)+3 ! 4 because we have 4 faces, 3 because it is leaving the bottom face
      receiver=chunks(chunks(chunk)%chunk_neighbours(chunk_bottom))%task
      CALL MPI_ISEND(bottom_snd_buffer,size,MPI_DOUBLE_PRECISION,receiver,tag &
                    ,MPI_COMM_WORLD,request(message_count+1),err)
      tag=4*(chunks(chunk)%chunk_neighbours(chunk_bottom))+4 ! 4 because we have 4 faces, 1 because it is coming from the top face of the bottom neighbour
      sender=chunks(chunks(chunk)%chunk_neighbours(chunk_bottom))%task
      CALL MPI_IRECV(bottom_rcv_buffer,size,MPI_DOUBLE_PRECISION,sender,tag &
                    ,MPI_COMM_WORLD,request(message_count+2),err)
      message_count=message_count+2
    ENDIF

    IF(chunks(chunk)%chunk_neighbours(chunk_top).NE.external_face) THEN
      tag=4*(chunk)+4 ! 4 because we have 4 faces, 4 because it is leaving the top face
      receiver=chunks(chunks(chunk)%chunk_neighbours(chunk_top))%task
      CALL MPI_ISEND(top_snd_buffer,size,MPI_DOUBLE_PRECISION,receiver,tag &
                    ,MPI_COMM_WORLD,request(message_count+1),err)
      tag=4*(chunks(chunk)%chunk_neighbours(chunk_top))+3 ! 4 because we have 4 faces, 4 because it is coming from the left face of the top neighbour
      sender=chunks(chunks(chunk)%chunk_neighbours(chunk_top))%task
      CALL MPI_IRECV(top_rcv_buffer,size,MPI_DOUBLE_PRECISION,sender,tag, &
                     MPI_COMM_WORLD,request(message_count+2),err)
      message_count=message_count+2
    ENDIF

  ENDIF

  ! Wait for the messages
  CALL MPI_WAITALL(message_count,request,status,err)

  ! Unpack buffers in halo cells
  IF(parallel%task.EQ.chunks(chunk)%task) THEN
    IF(use_fortran_kernels) THEN
      CALL unpack_top_bottom_buffers(chunks(chunk)%field%x_min,chunks(chunk)%field%x_max, &
                                     chunks(chunk)%field%y_min,chunks(chunk)%field%y_max, &
                                     chunks(chunk)%chunk_neighbours(chunk_bottom),        &
                                     chunks(chunk)%chunk_neighbours(chunk_top),           &
                                     external_face,                                       &
                                     x_inc,y_inc,depth,size,                              &
                                     field,bottom_rcv_buffer,top_rcv_buffer)
    ELSEIF(use_opencl_kernels)THEN
      CALL unpack_top_bottom_buffers_ocl(chunks(chunk)%field%x_min,chunks(chunk)%field%x_max, &
                                       chunks(chunk)%field%y_min,chunks(chunk)%field%y_max, &
                                       chunks(chunk)%chunk_neighbours(chunk_left),          &
                                       chunks(chunk)%chunk_neighbours(chunk_right),         &
                                       external_face,                                       &
                                       x_inc,y_inc,depth,                                   &
                                       which_field,                                         &
                                       field,left_snd_buffer,right_snd_buffer)
    ELSEIF(use_C_kernels)THEN
      CALL unpack_top_bottom_buffers_c(chunks(chunk)%field%x_min,chunks(chunk)%field%x_max, &
                                       chunks(chunk)%field%y_min,chunks(chunk)%field%y_max, &
                                       chunks(chunk)%chunk_neighbours(chunk_bottom),        &
                                       chunks(chunk)%chunk_neighbours(chunk_top),           &
                                       external_face,                                       &
                                       x_inc,y_inc,depth,size,                              &
                                       field,bottom_rcv_buffer,top_rcv_buffer)
    ENDIF
  ENDIF

END SUBROUTINE clover_exchange_message

SUBROUTINE clover_sum(value)

  ! Only sums to the master

  IMPLICIT NONE

  REAL(KIND=8) :: value

  REAL(KIND=8) :: total

  INTEGER :: err

  total=value

  CALL MPI_REDUCE(value,total,1,MPI_DOUBLE_PRECISION,MPI_SUM,0,MPI_COMM_WORLD,err)

  value=total

END SUBROUTINE clover_sum

SUBROUTINE clover_allsum(value)

  ! Global reduction for CG solver

  IMPLICIT NONE

  REAL(KIND=8) :: value

  REAL(KIND=8) :: total

  INTEGER :: err

  total=value

  CALL MPI_ALLREDUCE(value,total,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD,err)

  value=total

END SUBROUTINE clover_allsum

SUBROUTINE clover_min(value)

  IMPLICIT NONE

  REAL(KIND=8) :: value

  REAL(KIND=8) :: minimum

  INTEGER :: err

  minimum=value

  CALL MPI_ALLREDUCE(value,minimum,1,MPI_DOUBLE_PRECISION,MPI_MIN,MPI_COMM_WORLD,err)

  value=minimum

END SUBROUTINE clover_min

SUBROUTINE clover_max(value)

  IMPLICIT NONE

  REAL(KIND=8) :: value

  REAL(KIND=8) :: maximum

  INTEGER :: err

  maximum=value

  CALL MPI_ALLREDUCE(value,maximum,1,MPI_DOUBLE_PRECISION,MPI_MAX,MPI_COMM_WORLD,err)

  value=maximum

END SUBROUTINE clover_max

SUBROUTINE clover_allgather(value,values)

  IMPLICIT NONE

  REAL(KIND=8) :: value

  REAL(KIND=8) :: values(parallel%max_task)

  INTEGER :: err

  values(1)=value ! Just to ensure it will work in serial

  CALL MPI_ALLGATHER(value,1,MPI_DOUBLE_PRECISION,values,1,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,err)

END SUBROUTINE clover_allgather

SUBROUTINE clover_check_error(error)

  IMPLICIT NONE

  INTEGER :: error

  INTEGER :: maximum

  INTEGER :: err

  maximum=error

  CALL MPI_ALLREDUCE(error,maximum,1,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,err)

  error=maximum

END SUBROUTINE clover_check_error


END MODULE clover_module

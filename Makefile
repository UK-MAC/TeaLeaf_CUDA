# Crown Copyright 2014 AWE.
#
# This file is part of TeaLeaf.
#
# TeaLeaf is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the 
# Free Software Foundation, either version 3 of the License, or (at your option) 
# any later version.
#
# TeaLeaf is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
# details.
#
# You should have received a copy of the GNU General Public License along with 
# TeaLeaf. If not, see http://www.gnu.org/licenses/.  
#
#  @brief Makefile for CloverLeaf
#  @author David Beckingsale, Wayne Gaudin, Matthew Martineau
#  @details Agnostic, platform independent makefile for the TeaLeaf benchmark code.
#
# To select a OpenMP compiler option, you can do this in the shell before typing make:-
#
#  		make COMPILER=INTEL       # to select the Intel flags
#  		make COMPILER=GNU       # to select the GNU flags
#  		make COMPILER=CRAY        # to select the Cray flags
#  		make COMPILER=PGI         # to select the PGI flags
#  		make COMPILER=XL          # to select the IBM Xlf flags
#
# or you could export, e.g. 'export COMPILER=INTEL', in a shell before making
#
# Don't forget to set the number of threads you want to use, like so
#
# 		export OMP_NUM_THREADS=16
#
# USAGE:
# 		make                     # Will make the binary
#       make clean               # Will clean up the directory
#       make DEBUG=yes           # Selects debugging flags
#       make IEEE=yes            # Selects IEEE compliance flags
#
# EXAMPLE:
#
# 		make COMPILER=INTEL MPI_F90=mpiifort MPI_C=mpiicc DEBUG=yes IEEE=yes 
#
# will compile with the intel compiler with intel debug and ieee flags included

# User defined parameters
COMPILER = GNU
MPI_F90	 = mpif90
MPI_C	 = mpicc
MPI_CPP  = mpic++
OPTIONS = #-DENABLE_PROFILING
NV_ARCH	 = KEPLER

# Compiler-specific flags
OMP_INTEL     = -openmp
OMP_GNU       = -fopenmp -cpp
OMP_CRAY      = -e Z
OMP_PGI       = -mp=nonuma
OMP_XL        = -qsmp=omp -qthreaded
OMP		      = $(OMP_$(COMPILER))

FLAGS_INTEL     = -O3 -no-prec-div -xhost
FLAGS_GNU       = -O3 -march=native -funroll-loops
FLAGS_CRAY      = -em -ra -h acc_model=fast_addr:no_deep_copy:auto_async_all
FLAGS_PGI       = -fastsse -gopt -Mipa=fast -Mlist
FLAGS_XL       = -O5 -qipa=partition=large -g -qfullpath -Q -qsigtrap \
				 -qextname=flush:ideal_gas_kernel_c:viscosity_kernel_c:pdv_kernel_c:\
				 revert_kernel_c:accelerate_kernel_c:flux_calc_kernel_c:\
				 advec_cell_kernel_c:advec_mom_kernel_c:reset_field_kernel_c:timer_c:\
				 unpack_top_bottom_buffers_c:pack_top_bottom_buffers_c:\
				 unpack_left_right_buffers_c:pack_left_right_buffers_c:\
				 field_summary_kernel_c:update_halo_kernel_c:generate_chunk_kernel_c:\
				 initialise_chunk_kernel_c:calc_dt_kernel_c -qlistopt -qattr=full -qlist \
				 -qreport -qxref=full -qsource -qsuppress=1506-224:1500-036
FLAGS_           = -O3
CFLAGS_INTEL     = -O3 -no-prec-div -restrict -fno-alias -xhost
CFLAGS_GNU       = -O3 -march=native -funroll-loops
CFLAGS_CRAY      = -em -h list=a
CFLAGS_PGI       = -fastsse -gopt -Mipa=fast -Mlist
CFLAGS_XL        = -O5 -qipa=partition=large -g -qfullpath -Q -qlistopt -qattr=full -qlist -qreport -qxref=full -qsource -qsuppress=1506-224:1500-036 -qsrcmsg
CFLAGS_          = -O3

ifeq ($(DEBUG),yes)
  FLAGS_INTEL     = -O0 -g -debug all -check all -traceback -check noarg_temp_created
  FLAGS_GNU       = -O0 -g -O -Wall -Wextra -fbounds-check
  FLAGS_CRAY      = -O0 -g -em -eD
  FLAGS_PGI       = -O0 -g -C -Mchkstk -Ktrap=fp -Mchkfpstk -Mchkptr
  FLAGS_XL        = -O0 -g -qfullpath -qcheck -qflttrap=ov:zero:invalid:en -qsource \
					-qinitauto=FF -qmaxmem=-1 -qinit=f90ptr -qsigtrap \
					-qextname=flush:ideal_gas_kernel_c:viscosity_kernel_c:pdv_kernel_c:\
					revert_kernel_c:accelerate_kernel_c:flux_calc_kernel_c:\
					advec_cell_kernel_c:advec_mom_kernel_c:reset_field_kernel_c:\
					timer_c:unpack_top_bottom_buffers_c:pack_top_bottom_buffers_c:\
					unpack_left_right_buffers_c:pack_left_right_buffers_c:\
					field_summary_kernel_c:update_halo_kernel_c:generate_chunk_kernel_c:\
					initialise_chunk_kernel_c:calc_dt_kernel_c
  FLAGS_           = -O0 -g
  CFLAGS_          = -O0 -g
  CFLAGS_INTEL     = -O0 -g -debug all -traceback
  CFLAGS_GNU       = -O0 -g -O -Wall -Wextra -fbounds-check
  CFLAGS_CRAY      = -O0 -g -em -eD
  CFLAGS_PGI       = -O0 -g -C -Mchkstk -Ktrap=fp -Mchkfpstk
  CFLAGS_XL        = -O0 -g -qfullpath -qcheck -qflttrap=ov:zero:invalid:en -qsource -qinitauto=FF -qmaxmem=-1 -qsrcmsg
endif

ifeq ($(IEEE), yes)
  I3E_INTEL     = -fp-model strict -fp-model source -prec-div -prec-sqrt
  I3E_GNU       = -ffloat-store
  I3E_CRAY      = -hflex_mp=intolerant
  I3E_PGI       = -Kieee
  I3E_XL        = -qfloat=nomaf
  I3E			= $(I3E_$(COMPILER))
endif

CODE_GEN_FERMI			 = -gencode arch=compute_20,code=sm_21
CODE_GEN_KEPLER			 = -gencode arch=compute_35,code=sm_35
CODE_GEN_KEPLER_CONSUMER = -gencode arch=compute_30,code=sm_30

LDLIBS	+= -lcudart -lstdc++
FLAGS	 = $(FLAGS_$(COMPILER)) $(OMP) $(I3E) $(OPTIONS)
CFLAGS	 = $(CFLAGS_$(COMPILER)) $(OMP) $(I3E) $(OPTIONS)

# Requires CUDA_PATH to be set - not the same on all machines
NV_FLAGS  = -I$(CUDA_PATH)/include $(CODE_GEN_$(NV_ARCH)) -restrict \
		 	-Xcompiler "$(CFLAGS_GNU)" -D MPI_HDR
NV_FLAGS += -DNO_ERR_CHK
LDFLAGS  += -L$(CUDA_PATH)/lib64

ifeq ($(DEBUG),yes)
  NV_FLAGS += -O0 -g -G
else
  NV_FLAGS += -O3 --ptxas-options="-v"
endif

FOBJ=\
			  data.o						\
			  definitions.o					\
			  pack_kernel.o					\
			  tea.o							\
			  report.o						\
			  timer.o						\
			  parse.o						\
			  read_input.o					\
			  initialise_chunk_kernel.o		\
			  initialise_chunk.o			\
			  build_field.o					\
			  update_halo_kernel.o			\
			  update_halo.o					\
			  start.o						\
			  generate_chunk_kernel.o		\
			  generate_chunk.o				\
			  initialise.o					\
			  field_summary_kernel.o		\
			  field_summary.o				\
			  calc_dt.o						\
			  timestep.o					\
			  set_field_kernel.o            \
			  set_field.o                   \
			  tea_leaf_jacobi.o             \
			  tea_leaf_cg.o             	\
			  tea_leaf_cheby.o             	\
			  tea_leaf_ppcg.o             	\
			  tea_solve.o                   \
			  visit.o						\
			  tea_leaf.o					\
			  diffuse.o

OBJ  = $(patsubst %.cu,%.o,$(wildcard *.cu))
OBJ += $(patsubst %.cpp,%.o,$(wildcard *.cpp))
OBJ += $(patsubst %.c,%.o,$(wildcard *.c))
OBJ += $(FOBJ)

tea_leaf: Makefile $(OBJ)
	$(MPI_F90) $(FLAGS) $(OBJ)	$(LDFLAGS) $(LDLIBS) -o tea_leaf
	@echo $(MESSAGE)

include make.deps

%.o: %.cu Makefile make.deps
	nvcc $(NV_FLAGS) -c $<
%_module.mod: %.f90 %.o
	@true
%.o: %.f90 Makefile make.deps
	$(MPI_F90) $(FLAGS) -c $<
%.o: %.c Makefile make.deps
	$(MPI_C) $(CFLAGS) -c $<
%.o: %.cpp Makefile make.deps
	$(MPI_CPP) $(CFLAGS) -c $<

.PHONY: clean

clean:
	rm -f *.o *.mod *genmod* *.lst *.cub *.ptx tea_leaf


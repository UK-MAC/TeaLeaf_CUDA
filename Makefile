#Crown Copyright 2014 AWE.
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

#  @brief Makefile for CloverLeaf
#  @author David Beckingsale, Wayne Gaudin
#  @details Agnostic, platform independent makefile for the TeaLeaf benchmark code.

# Agnostic, platform independent makefile for the TeaLeaf benchmark code.
# It is not meant to be clever in anyway, just a simple build out of the box script.
# Just make sure mpif90 is in your path. It uses mpif90 even for all builds because this abstracts the base
#  name of the compiler. If you are on a system that doesn't use mpif90, just replace mpif90 with the compiler name
#  of choice. The only mpi dependencies in this non-MPI version are mpi_wtime in timer.f90.

# There is no single way of turning OpenMP compilation on with all compilers.
# The known compilers have been added as a variable. By default the make
#  will use no options, which will work on Cray for example, but not on other
#  compilers.
# To select a OpenMP compiler option, do this in the shell before typing make:-
#
#  export COMPILER=INTEL       # to select the Intel flags
#  export COMPILER=SUN         # to select the Sun flags
#  export COMPILER=GNU         # to select the Gnu flags
#  export COMPILER=CRAY        # to select the Cray flags
#  export COMPILER=PGI         # to select the PGI flags
#  export COMPILER=PATHSCALE   # to select the Pathscale flags
#  export COMPILER=XL          # to select the IBM Xlf flags

# or this works as well:-
#
# make COMPILER=INTEL
# make COMPILER=SUN
# make COMPILER=GNU
# make COMPILER=CRAY
# make COMPILER=PGI
# make COMPILER=PATHSCALE
# make COMPILER=XL
#

# Don't forget to set the number of threads you want to use, like so
# export OMP_NUM_THREADS=4

# usage: make                     # Will make the binary
#        make clean               # Will clean up the directory
#        make DEBUG=1             # Will select debug options. If a compiler is selected, it will use compiler specific debug options
#        make IEEE=1              # Will select debug options as long as a compiler is selected as well
# e.g. make COMPILER=INTEL MPI_COMPILER=mpiifort C_MPI_COMPILER=mpiicc DEBUG=1 IEEE=1 # will compile with the intel compiler with intel debug and ieee flags included

ifndef COMPILER
  MESSAGE=select a compiler to compile in OpenMP, e.g. make COMPILER=INTEL
endif

OMP_INTEL     = -openmp
OMP_SUN       = -xopenmp=parallel -vpara
OMP_GNU       = -fopenmp
OMP_CRAY      =
OMP_PGI       = -mp=nonuma
OMP_PATHSCALE = -mp
OMP_XL        = -qsmp=omp -qthreaded
OMP=$(OMP_$(COMPILER))

FLAGS_INTEL     = -O3 -ipo -no-prec-div
FLAGS_SUN       = -fast -xipo=2 -Xlistv4
FLAGS_GNU       = -O3 -march=native -funroll-loops
FLAGS_CRAY      = -em -ra -h acc_model=fast_addr:no_deep_copy:auto_async_all
FLAGS_PGI       = -fastsse -gopt -Mipa=fast -Mlist
FLAGS_PATHSCALE = -O3
FLAGS_XL       = -O5 -qipa=partition=large -g -qfullpath -Q -qsigtrap -qextname=flush:ideal_gas_kernel_c:viscosity_kernel_c:pdv_kernel_c:revert_kernel_c:accelerate_kernel_c:flux_calc_kernel_c:advec_cell_kernel_c:advec_mom_kernel_c:reset_field_kernel_c:timer_c:unpack_top_bottom_buffers_c:pack_top_bottom_buffers_c:unpack_left_right_buffers_c:pack_left_right_buffers_c:field_summary_kernel_c:update_halo_kernel_c:generate_chunk_kernel_c:initialise_chunk_kernel_c:calc_dt_kernel_c -qlistopt -qattr=full -qlist -qreport -qxref=full -qsource -qsuppress=1506-224:1500-036
FLAGS_          = -O3
CFLAGS_INTEL     = -O3 -ipo -no-prec-div -restrict -fno-alias
CFLAGS_SUN       = -fast -xipo=2
CFLAGS_GNU       = -O3 -march=native -funroll-loops
CFLAGS_CRAY      = -em -h list=a
CFLAGS_PGI       = -fastsse -gopt -Mipa=fast -Mlist
CFLAGS_PATHSCALE = -O3
CFLAGS_XL       = -O5 -qipa=partition=large -g -qfullpath -Q -qlistopt -qattr=full -qlist -qreport -qxref=full -qsource -qsuppress=1506-224:1500-036 -qsrcmsg
CFLAGS_          = -O3

ifdef DEBUG
  FLAGS_INTEL     = -O0 -g -debug all -check all -traceback -check noarg_temp_created
  FLAGS_SUN       = -g -xopenmp=noopt -stackvar -u -fpover=yes -C -ftrap=common
  FLAGS_GNU       = -O0 -g -O -Wall -Wextra -fbounds-check
  FLAGS_CRAY      = -O0 -g -em -eD
  FLAGS_PGI       = -O0 -g -C -Mchkstk -Ktrap=fp -Mchkfpstk -Mchkptr
  FLAGS_PATHSCALE = -O0 -g
  FLAGS_XL       = -O0 -g -qfullpath -qcheck -qflttrap=ov:zero:invalid:en -qsource -qinitauto=FF -qmaxmem=-1 -qinit=f90ptr -qsigtrap -qextname=flush:ideal_gas_kernel_c:viscosity_kernel_c:pdv_kernel_c:revert_kernel_c:accelerate_kernel_c:flux_calc_kernel_c:advec_cell_kernel_c:advec_mom_kernel_c:reset_field_kernel_c:timer_c:unpack_top_bottom_buffers_c:pack_top_bottom_buffers_c:unpack_left_right_buffers_c:pack_left_right_buffers_c:field_summary_kernel_c:update_halo_kernel_c:generate_chunk_kernel_c:initialise_chunk_kernel_c:calc_dt_kernel_c
  FLAGS_          = -O0 -g
  CFLAGS_INTEL    = -O0 -g -debug all -traceback
  CFLAGS_SUN      = -g -O0 -xopenmp=noopt -stackvar -u -fpover=yes -C -ftrap=common
  CFLAGS_GNU       = -O0 -g -O -Wall -Wextra -fbounds-check
  CFLAGS_CRAY     = -O0 -g -em -eD
  CFLAGS_PGI      = -O0 -g -C -Mchkstk -Ktrap=fp -Mchkfpstk
  CFLAGS_PATHSCALE= -O0 -g
  CFLAGS_XL      = -O0 -g -qfullpath -qcheck -qflttrap=ov:zero:invalid:en -qsource -qinitauto=FF -qmaxmem=-1 -qsrcmsg
endif

ifdef IEEE
  I3E_INTEL     = -fp-model strict -fp-model source -prec-div -prec-sqrt
  I3E_SUN       = -fsimple=0 -fns=no
  I3E_GNU       = -ffloat-store
  I3E_CRAY      = -hflex_mp=intolerant
  I3E_PGI       = -Kieee
  I3E_PATHSCALE = -mieee-fp
  I3E_XL       = -qfloat=nomaf
  I3E=$(I3E_$(COMPILER))
endif

# flags for nvcc
# set NV_ARCH to select the correct one
NV_ARCH=KEPLER
CODE_GEN_FERMI=-gencode arch=compute_20,code=sm_21
CODE_GEN_KEPLER=-gencode arch=compute_35,code=sm_35

LDLIBS+=-lstdc++ -lcudart

FLAGS=$(FLAGS_$(COMPILER)) $(OMP) $(I3E) $(OPTIONS)
CFLAGS=$(CFLAGS_$(COMPILER)) $(OMP) $(I3E) $(C_OPTIONS) -c
MPI_COMPILER=mpif90
C_MPI_COMPILER=mpicc
CXX_MPI_COMPILER=mpiCC

CXXFLAGS+=$(CFLAGS)

# requires CUDA_HOME to be set - not the same on all machines
NV_FLAGS=-O2 -I$(CUDA_HOME)/include $(CODE_GEN_$(NV_ARCH)) -restrict -Xcompiler "$(CXXFLAGS)"
NV_FLAGS+=-DNO_ERR_CHK
#NV_FLAGS+=-DTIME_KERNELS

C_FILES=\
	accelerate_kernel_c.o           \
	pack_kernel_c.o \
	PdV_kernel_c.o                  \
	tqli.o			\
	timer_c.o                  \
	initialise_chunk_kernel_c.o                  \
	calc_dt_kernel_c.o                  \
	field_summary_kernel_c.o                  \
	update_halo_kernel_c.o                  \
	generate_chunk_kernel_c.o                  \
	flux_calc_kernel_c.o            \
	tea_leaf_kernel_c.o			\
	revert_kernel_c.o               \
	reset_field_kernel_c.o          \
	set_field_kernel_c.o          \
	ideal_gas_kernel_c.o            \
	viscosity_kernel_c.o            \
	advec_cell_kernel_c.o			\
	advec_mom_kernel_c.o

FORTRAN_FILES=\
	clover.o \
	pack_kernel.o \
	data.o			\
	definitions.o			\
	tea.o			\
	tea_leaf_jacobi.o			\
	tea_leaf_cg.o			\
	tea_leaf_cheby.o			\
	report.o			\
	timer.o			\
	parse.o			\
	read_input.o			\
	initialise_chunk_kernel.o	\
	initialise_chunk.o		\
	build_field.o			\
	update_halo_kernel.o		\
	update_halo.o			\
	ideal_gas_kernel.o		\
	ideal_gas.o			\
	start.o			\
	generate_chunk_kernel.o	\
	generate_chunk.o		\
	initialise.o			\
	field_summary_kernel.o	\
	field_summary.o		\
	viscosity_kernel.o		\
	viscosity.o			\
	calc_dt_kernel.o		\
	calc_dt.o			\
	timestep.o			\
	accelerate_kernel.o		\
	accelerate.o			\
	revert_kernel.o		\
	revert.o			\
	PdV_kernel.o			\
	PdV.o				\
	flux_calc_kernel.o		\
	flux_calc.o			\
	advec_cell_kernel.o		\
	advec_cell_driver.o		\
	advec_mom_kernel.o		\
	advec_mom_driver.o		\
	advection.o			\
	reset_field_kernel.o		\
	set_field_kernel.o		\
	reset_field.o			\
	set_field.o			\
	hydro.o			\
	visit.o			\
	tea_leaf.o

CUDA_FILES= \
	accelerate_kernel_cuda.o \
	advec_cell_kernel_cuda.o \
	advec_mom_kernel_cuda.o \
	calc_dt_kernel_cuda.o \
	field_summary_kernel_cuda.o \
	flux_calc_kernel_cuda.o \
	generate_chunk_kernel_cuda.o \
	ideal_gas_kernel_cuda.o \
	init_cuda.o \
	initialise_chunk_kernel_cuda.o \
	mpi_transfers_cuda.o \
	pack_buffer_kernels.o \
	PdV_kernel_cuda.o \
	reset_field_kernel_cuda.o \
	revert_kernel_cuda.o \
	update_halo_kernel_cuda.o \
	viscosity_kernel_cuda.o

tea_leaf: Makefile $(FORTRAN_FILES) $(C_FILES) $(CUDA_FILES)
	$(MPI_COMPILER) $(FLAGS)	\
	$(FORTRAN_FILES)	\
	$(C_FILES)	\
	$(CUDA_FILES) \
	$(LDFLAGS) \
	$(LDLIBS) \
	-o tea_leaf
	@echo $(MESSAGE)

include make.deps

%.o: %.cu Makefile make.deps
	nvcc $(NV_FLAGS) -c $< -o $*.o
%.mod %_module.mod %_leaf_module.mod: %.f90 %.o
	@true
%.o: %.f90 Makefile make.deps
	$(MPI_COMPILER) $(FLAGS) -c $< -o $*.o
%.o: %.c Makefile make.deps
	$(C_MPI_COMPILER) $(CFLAGS) -c $< -o $*.o

clean:
	rm -f *.o *.mod *genmod* *.lst *.cub *.ptx tea_leaf

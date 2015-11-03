# TeaLeaf - CUDA + MPI

TeaLeaf CUDA + MPI extendes the original TeaLeaf application with a set of CUDA kernels which allow the solvers to be run on NVIDIA GPUs.

## Compiler Support

This implementation has been tested against a number of compilers:

GNU   - 4.8.4, 4.9.3
Intel - 15.0.3
PGI   - 13.5, 14.7
Cray  - 8.2, 8.4

## Compiling

This implementation has been configured with default values, that should hopefully allow you to simply compile with `make`.

The `COMPILER` flag specifies which compiler suite to use, and is defaulted to use the GNU compilers.

The default MPI wrappers are `mpif90` and `mpicc`, if you want to change those you can simply type:

`make MPI_F90=<alternative> MPI_C=<alternative>`
`e.g. MPI_F90=mpiifort MPI_C=mpiicc`

### Other Flags

All other flags can be passed in to the `make` command or changed at the top of the Makefile. 

The default compilation with the COMPILER flag set chooses the optimal 
performing set of flags for the specified compiler, but with no hardware 
specific options or IEEE compatability.

To produce a version that has IEEE compatiblity a further flag has to be set on 
the compiler line.

`make IEEE=yes`

* INTEL: -fp-model strict -fp-model source -prec-div -prec-sqrt
* GNU  : -ffloat-store

Note that the MPI communications have been written to ensure bitwise identical 
answers independent of core count. However under some compilers this is not 
true unless the IEEE flags is set to be true. This is certainly true of the 
Intel and Cray compiler. Even with the IEEE options set, this is not guarantee 
that different compilers or platforms will produce the same answers. Indeed a 
Fortran run can give different answers from a C run with the same compiler, 
same options and same hardware.

Extra options can be added without modifying the makefile by setting the `OPTIONS` flag.

`make COMPILER=INTEL OPTIONS=-xavx`

Finally, a `DEBUG` flag can be set to use debug options for a specific compiler.

`make COMPILER=PGI DEBUG=yes`

These flags are also compiler specific, and so will depend on the `COMPILER` 
environment variable.

Optional function-level profiling can be enabled for all of the C components of the application by defining `ENABLE_PROFILING`.

`make COMPILER=INTEL OPTIONS=-DENABLE_PROFILING`

### File Input

The contents of tea.in defines the geometric and run time information, apart from task and thread counts.

A complete list of options is given below, where `<R>` shows the option takes a real number as an argument. Similarly `<I>` is an integer argument.

`initial_timestep <R>`

Set the initial time step for TeaLeaf. This time step stays constant through the entire simulation. The default value is 

`end_time <R>`

Sets the end time for the simulation. When the simulation time is greater than this number the simulation will stop.

`end_step <I>`

Sets the end step for the simulation. When the simulation step is equal to this then simulation will stop.

In the event that both the above options are set, the simulation will terminate on whichever completes first.

`xmin <R>`

`xmax <R>`

`ymin <R>`

`ymax <R>`

`zmin <R>` (for 3d)

`zmax <R>` (for 3d)

The above options set the size of the computational domain. The default domain size is a 10cm square/cubed. 

`x_cells <I>`

`y_cells <I>`

`z_cells <I>` (for 3d)

The options above set the cell count for each coordinate direction.

The geometric information and initial conditions are set using the following keywords with three possible variations. Note that state 1 is always the ambient material and any geometry information is ignored. Areas not covered by other defined states receive the energy and density of state 1.

2d:

`state <I> density <R> energy <R> geometry rectangle xmin <R> ymin <R> xmax <R> ymax <R> `

3d:

`state <I> density <R> energy <R> geometry cuboid xmin <R> ymin <R> zmin <R> xmax <R> ymax <R> zmax <R>`

Defines a rectangular region of the domain with the specified energy and density.

2d:
`state <I> density <R> energy <R> geometry circle xmin <R> ymin <R> radius <R>`

3d:
`state <I> density <R> energy <R> geometry circle xmin <R> ymin <R> zmin <R> radius <R>`

Defines a circular region of the domain with the specified energy and density.

2d:
`state <I> density <R> energy <R> geometry point xmin <R> ymin <R>`

3d:
`state <I> density <R> energy <R> geometry point xmin <R> ymin <R> zmax <R>`

Defines a cell in the domain with the specified energy and density.

Note that the generator is simple and the defined state completely fills a cell with which it intersects. In the case of over lapping regions, the last state takes priority. Hence a circular region will have a stepped interface and a point data will fill the cell it lies in with its defined energy and density.

`visit_frequency <I>`

This is the step frequency of visualisations dumps. The files produced are text base VTK files and are easily viewed in an application such as ViSit. The default is to output no graphical data. Note that the overhead of output is high, so should not be invoked when performance benchmarking is being carried out.

`summary_frequency <I>`

This is the step frequency of summary dumps. This requires a global reduction and associated synchronisation, so performance will be slightly affected as the frequency is increased. The default is for a summary dump to be produced every 10 steps and at the end of the simulation.

`tl_ch_cg_presteps  <I>`

This option specifies the number of Conjugate Gradient iterations completed before the Chebyshev method is started. This is necessary to provide approximate minimum and maximum eigen values to start the Chebyshev method. The default value is 30.

`tl_ppcg_inner_steps <I>`

Number of inner steps to run when using the PPCG solver. Please note that a large mesh size will require this parameter to be increased for optimal performance. For instance, a mesh size of 4096x4096 works best when the parameter is set to 350 or greater.

`use_ext_kernels`

Makes the application use the C++ Kokkos kernels at runtime.

`use_fortran_kernels`

Make the application use the Fortran kernels at runtime.

`tl_ch_cg_errswitch`

If enabled alongside Chebshev/PPCG solver, switch when a certain error is reached instead of when a certain number of steps is reached. The default for this is off.

`tl_ch_cg_epslim`

Default error to switch from CG to Chebyshev when using Chebyshev solver with the tl_cg_ch_errswitch option enabled. The default value is 1e-5.

`tl_check_result`

After the solver reaches convergence, calculate ||b-Ax|| to make sure the solver has actually converged. The default for this option is off.

`tl_use_jacobi`

This keyword selects the Jacobi method to solve the linear system. Note that this a very slowly converging method compared to other options. This is the default method is no method is explicitly selected.

`tl_use_cg`

This keyword selects the Conjugate Gradient method to solve the linear system.

`tl_use_ppcg`

This keyword selects the Conjugate Gradient method to solve the linear system.

`tl_use_chebyshev`

This keyword selects the Chebyshev method to solve the linear system.

`profiler_on`

This option turns the Fortran code's coarse grained internal profiler end. Timing information is reported at the end of the simulation in the tea.out file. The default is no profiling.

`verbose_on`

The option prints out extra information such as residual per iteration of a solve.

`tl_max_iters <I>`

This option provides an upper limit of the number of iterations used for the linear solve in a step. If this limit is reached, then the solution vector at this iteration is used as the solution, even if the convergence criteria has not been met. For this reason, care should be taken in the comparison of the performance of a slowly converging method, such as Jacobi, as the convergence criteria may not have been met for some of the steps. The default value is 1000.

`tl_eps <R>`

This option sets the convergence criteria for the selected solver. It uses a least squares measure of the residual. The default value is 1.0e-10.

`tl_coefficient_density

This option uses the density as the conduction coefficient. This is the default option.

`tl_coefficient_inverse_density

This option uses the inverse density as the conduction coefficient.

`test_problem <I>`

This keyword selects a standard test with a "known" solution. Test problem 1 is automatically generated if the tea.in file does not exist. Test problems 2-5 are shipped in the TeaLeaf repository. Note that the known solution for an iterative solver is not an analytic solution but is the solution for a single core simulation with IEEE options enabled with the Intel compiler and a strict convergence of 1.0e-15. The difference to the expected solution is reported at the end of the simulation in the tea.out file. There is no default value for this option.

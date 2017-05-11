# TeaLeaf

* Device selection is done by choosing the devices starting at the index given
  by `cuda_device` in tea.in. For example, if there are 2 devices on a system and
  `cuda_device` is 0, rank 0 will take device 0 and rank 1 will take device 1.

# Extra tea.in flags

Turn on cuda kernel use by putting `use_cuda_kernels` in tea.in.

## Solver flags

* `tl_max_iters` specifies the number of iterations to do before stopping
* `tl_eps` specifies the acceptable error level to stop at

Enabling these flags will turn on the relevant solver:

* `tl_use_jacobi` - use a simple jacobi iteration
* `tl_use_cg` - use the conjugate gradient method 
* `tl_use_ppcg` - use the polynomially preconditioned conjugate gradient method 
* `tl_use_chebyshev` - use chebyshev solver after running a few iterations of
  the conjugate gradient solver (not preconditioned) to approximate eigenvalues.
  The number of iterations of CG to run before switching to the chebyshev solver
  can be specified with the `tl_chebyshev_steps` flag (eg,
  `tl_chebyshev_steps=20`)
* `tl_use_preconditioner_type` - specify a preconditioner. Current options 'none', 'jac-diag' (Diagonal Jacobi) or
   'jac-block' (Block Jacobi)

## TODO

* Make preconditioner selectable from tea.in and not a compile time flag

## Compiling

- In many case just typing `make` in the required software version will work. 

If the MPI compilers have different names then the build process needs to 
notified of this by defining two environment variables, `MPI_COMPILER` and 
`C_MPI_COMPILER`. 

For example on some Intel systems:

`make MPI_COMPILER=mpiifort C_MPI_COMPILER=mpiicc`

Or on Cray systems:

`make MPI_COMPILER=ftn C_MPI_COMPILER=cc`

### OpenMP Build

All compilers use different arguments to invoke OpenMP compilation. A simple 
call to make will invoke the compiler with -O3. This does not usually include 
OpenMP by default. To build for OpenMP for a specific compiler a further 
variable must be defined, `COMPILER` that will then select the correct option 
for OpenMP compilation. 

For example with the Intel compiler:

`make COMPILER=INTEL`

Which then append the -openmp to the build flags.

Other supported compiler that will be recognise are:-

* CRAY
* SUN
* GNU
* IBM
* PATHSCALE
* PGI

The default flags for each of these is show below:-

* INTEL: -O3 -ipo
* SUN: -fast
* GNU: -ipo
* XL: -O5
* PATHSCLE: -O3
* PGI: -O3 -Minline
* CRAY: -em  _Note: that by default the Cray compiler with pick the optimum 
options for performance._

### Other Flags

The default compilation with the COMPILER flag set chooses the optimal 
performing set of flags for the specified compiler, but with no hardware 
specific options or IEEE compatability.

To produce a version that has IEEE compatiblity a further flag has to be set on 
the compiler line.

`make COMPILER=INTEL IEEE=1`

This flag has no effect if the compiler flag is not set because IEEE options 
are always compiler specific.

For each compiler the flags associated with IEEE are shown below:-

* INTEL: -fp-model strict –fp-model source –prec-div –prec-sqrt
* CRAY: -hpflex_mp=intolerant
* SUN: -fsimple=0 –fns=no
* GNU: -ffloat-store
* PGI: -Kieee
* PATHSCALE: -mieee-fp
* XL: -qstrict –qfloat=nomaf

Note that the MPI communications have been written to ensure bitwise identical 
answers independent of core count. However under some compilers this is not 
true unless the IEEE flags is set to be true. This is certainly true of the 
Intel and Cray compiler. Even with the IEEE options set, this is not guarantee 
that different compilers or platforms will produce the same answers. Indeed a 
Fortran run can give different answers from a C run with the same compiler, 
same options and same hardware.

Extra options can be added without modifying the makefile by adding two further 
flags, `OPTIONS` and `C_OPTIONS`, one for the Fortran and one for the C options.

`make COMPILER=INTEL OPTIONS=-xavx C_OPTIONS=-xavx`

Finally, a `DEBUG` flag can be set to use debug options for a specific compiler.

`make COMPILER=PGI DEBUG=1`

These flags are also compiler specific, and so will depend on the `COMPILER` 
environment variable.

So on a system without the standard MPI wrappers, for a build that requires 
OpenMP, IEEE and AVX this would look like so:-

```
make COMPILER=INTEL MPI_COMPILER=mpiifort C_MPI_COMPILER=mpiicc IEEE=1 \
OPTIONS="-xavx" C_OPTIONS="-xavx"
```

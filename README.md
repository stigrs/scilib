# Scilib [![Build Status](https://dev.azure.com/stigrs0020/stigrs/_apis/build/status/stigrs.scilib?branchName=main)](https://dev.azure.com/stigrs0020/stigrs/_build/latest?definitionId=8&branchName=main)

Scilib provides a C++ library for linear algebra and scientific computing.
BLAS and LAPACK are used for fast numerical performance. Currently, OpenBLAS
and Intel MKL are supported.

## Features

* N-dimensional dense matrices using std::experimental::mdspan for views 
  (row-major storage order)
* Linear algebra methods
* Integration methods
* Mathematical constants, metric prefixes, physical constants, and
  conversion factors

_Note: Some features are only available if Intel MKL is used._

## Licensing

Numlib is released under the [MIT](LICENSE) license.

## Usage of Third Party Libraries

This project makes use of [GoogleTest](https://https://github.com/google/googletest).
Please see the [ThirdPartyNotices.txt](ThirdPartyNotices.txt) file for details
regarding the licensing of GoogleTest.

## Quick Start

### Requirements

* [CMake](https://cmake.org) 3.14
* [mdspan](https://github.com/kokkos/mdspan)
* [OpenBLAS](https://www.openblas.net/) (Intel MKL is recommended)
* [Armadillo](http://arma.sourceforge.net) (for benchmarking)

### Supported Compilers

| Compiler      | Versions Tested |
|:--------------|----------------:|
| GCC           | 9               |
| Clang         | 11              |
| Visual Studio | VS2019          |
| XCode         | 13.0            |

### Obtaining the Source Code

The source code can be obtained from

        git clone git@github.com:stigrs/scilib.git

### Building the Software

These steps assumes that the source code of this repository has been cloned
into a directory called `scilib`.

1. Create a directory to contain the build outputs:

        cd scilib
        mkdir build
        cd build

2. Configure CMake to use the compiler of your choice (you can see a list by
   running `cmake --help`):

        cmake -G "Visual Studio 16 2019" ..

3. Build the software (in this case in the Release configuration):

        cmake --build . --config Release

4. Run the test suite:

        ctest -C Release

5. Install the software:

        cmake --build . --config Release --target install

   All tests should pass, indicating that your platform is fully supported.

6. Benchmarks can be built by setting the option BUILD_BENCH to ON. Please
   make sure BLAS run on the same number of threads in Armadillo and Scilib
   before comparing the benchmark results. If OpenBLAS is used, this can be
   controlled by setting the OPENBLAS_NUM_THREADS environmental variable.

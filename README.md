# Scilib 
[![CMake](https://github.com/stigrs/scilib/actions/workflows/cmake.yml/badge.svg?branch=main)](https://github.com/stigrs/scilib/actions/workflows/cmake.yml) [![codecov](https://codecov.io/gh/stigrs/scilib/branch/main/graph/badge.svg?token=IBOP66BJ5C)](https://codecov.io/gh/stigrs/scilib)[![CodeQL](https://github.com/stigrs/scilib/actions/workflows/codeql-analysis.yml/badge.svg?branch=main)](https://github.com/stigrs/scilib/actions/workflows/codeql-analysis.yml)

Scilib provides a C++ library for linear algebra and scientific computing.
BLAS and LAPACK are used for fast numerical performance. Currently, OpenBLAS
and Intel MKL are supported.

## Features

* Multidimensional dense arrays (row-major or column-major storage order; default is row-major)
* Linear algebra methods
* Integration methods
* Simple solver for initial value problems (Dormand-Prince)
* Common statistical methods
* Mathematical constants, metric prefixes, physical constants, and conversion factors

## Licensing

Scilib is released under the [MIT](LICENSE) license.

## Usage of Third Party Libraries

This project makes use of the following third-party libraries:
* [GoogleTest](https://github.com/google/googletest) 
* [mdspan](https://github.com/kokkos/mdspan)
* [stdBLAS](https://github.com/kokkos/stdBLAS)
* [Microsoft.GSL](https://github.com/microsoft/GSL)

Please see their websites for details regarding licensing terms.

## Quick Start

### Requirements

* [CMake](https://cmake.org) 3.13
* [OpenBLAS](https://www.openblas.net/) (Intel MKL is recommended)

### Supported Compilers

| Compiler      | Versions Tested |
|:--------------|----------------:|
| GCC           | 9, 10           |
| Clang         | 10, 11, 12      |
| Visual Studio | VS2019, VS2022  |
| XCode         | 13.0            |
| Intel         | 2022            |

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

        cmake -G "Visual Studio 17 2022" ..

3. Build the software (in this case in the Release configuration):

        cmake --build . --config Release

4. Run the test suite:

        ctest -C Release

5. Install the software:

        cmake --build . --config Release --target install

   All tests should pass, indicating that your platform is fully supported.

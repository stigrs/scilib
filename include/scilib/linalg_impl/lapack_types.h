// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_LAPACK_TYPES_H
#define SCILIB_LINALG_LAPACK_TYPES_H

//------------------------------------------------------------------------------
// Define integer type for use with BLAS, LAPACK and Intel MKL.

#ifdef USE_MKL
#include <mkl.h>
#define BLAS_INT MKL_INT
#else
#define BLAS_INT int
#endif

//------------------------------------------------------------------------------
// Define complex type for use with LAPACK.

#ifndef USE_MKL
#include <complex>
#ifndef lapack_complex_float
#define lapack_complex_float std::complex<float>
#endif
#ifndef lapack_complex_double
#define lapack_complex_double std::complex<double>
#endif
#endif

#endif // SCILIB_LINALG_LAPACK_TYPES_H

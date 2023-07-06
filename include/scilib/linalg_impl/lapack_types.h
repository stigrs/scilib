// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_LAPACK_TYPES_H
#define SCILIB_LINALG_LAPACK_TYPES_H

#include <complex>

#ifdef USE_MKL
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>
#include <mkl.h>
#define BLAS_INT MKL_INT
#else
#include <cblas.h>
#ifdef blasint
#define BLAS_INT blasint
#else
#define BLAS_INT int
#endif
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#include <lapacke.h>
#endif

#endif // SCILIB_LINALG_LAPACK_TYPES_H

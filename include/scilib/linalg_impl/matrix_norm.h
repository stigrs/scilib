// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_MATRIX_NORM_H
#define SCILIB_LINALG_MATRIX_NORM_H

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

#include <scilib/traits.h>
#include <scilib/mdarray.h>

namespace Scilib {
namespace Linalg {

namespace stdex = std::experimental;

// Matrix norm of a general rectangular matrix:
//
// Types of matrix norms:
// - M, m:       largest absolute value of the matrix
// - 1, O, o:    1-norm of the matrix (maximum column sum)
// - I, i:       infinity norm of the matrix (maximum row sum)
// - F, f, E, e: Frobenius norm of the matrix (square root of sum of squares)
//
inline double matrix_norm(Matrix_view<double> a, char norm)
{
    assert(norm == 'M' || norm == 'm' || norm == '1' || norm == 'O' ||
           norm == 'o' || norm == 'I' || norm == 'i' || norm == 'F' ||
           norm == 'f' || norm == 'E' || norm == 'e');

    BLAS_INT m = narrow_cast<BLAS_INT>(a.extent(0));
    BLAS_INT n = narrow_cast<BLAS_INT>(a.extent(1));
    BLAS_INT lda = n;

    return LAPACKE_dlange(LAPACK_ROW_MAJOR, norm, m, n, a.data(), lda);
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_MATRIX_NORM_H

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
#include <lapacke.h>
#endif

#include <scilib/mdarray.h>
#include <scilib/linalg_impl/lapack_types.h>
#include <type_traits>

namespace Sci {
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
template <class T, class Layout>
    requires(std::is_same_v<std::remove_cv_t<T>, double>)
inline auto matrix_norm(Sci::Matrix_view<T, Layout> a, char norm)
{
    static_assert(a.is_contiguous());

    assert(norm == 'M' || norm == 'm' || norm == '1' || norm == 'O' ||
           norm == 'o' || norm == 'I' || norm == 'i' || norm == 'F' ||
           norm == 'f' || norm == 'E' || norm == 'e');

    auto matrix_layout = LAPACK_ROW_MAJOR;
    BLAS_INT m = static_cast<BLAS_INT>(a.extent(0));
    BLAS_INT n = static_cast<BLAS_INT>(a.extent(1));
    BLAS_INT lda = n;

    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = LAPACK_COL_MAJOR;
        lda = m;
    }
    return LAPACKE_dlange(matrix_layout, norm, m, n, a.data(), lda);
}

template <class Layout, class Allocator>
inline double matrix_norm(const Sci::Matrix<double, Layout, Allocator>& a,
                          char norm)
{
    return matrix_norm(a.view(), norm);
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_MATRIX_NORM_H

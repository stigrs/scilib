// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_MATRIX_NORM_H
#define SCILIB_LINALG_MATRIX_NORM_H

#include "lapack_types.h"
#include <type_traits>

namespace Sci {
namespace Linalg {


// Matrix norm of a general rectangular matrix:
//
// Types of matrix norms:
// - M, m:       largest absolute value of the matrix
// - 1, O, o:    1-norm of the matrix (maximum column sum)
// - I, i:       infinity norm of the matrix (maximum row sum)
// - F, f, E, e: Frobenius norm of the matrix (square root of sum of squares)
//
template <class T,
          class IndexType,
          std::size_t nrows,
          std::size_t ncols,
          class Layout,
          class Accessor>
    requires(
        (std::is_same_v<std::remove_cv_t<T>, double> ||
         std::is_same_v<std::remove_cv_t<T>, std::complex<double>>) &&std::is_integral_v<IndexType>)
inline double
matrix_norm(Kokkos::mdspan<T, Kokkos::extents<IndexType, nrows, ncols>, Layout, Accessor> a,
            char norm)
{
    Expects(norm == 'M' || norm == 'm' || norm == '1' || norm == 'O' || norm == 'o' ||
            norm == 'I' || norm == 'i' || norm == 'F' || norm == 'f' || norm == 'E' || norm == 'e');

    auto matrix_layout = LAPACK_ROW_MAJOR;
    BLAS_INT m = gsl::narrow_cast<BLAS_INT>(a.extent(0));
    BLAS_INT n = gsl::narrow_cast<BLAS_INT>(a.extent(1));
    BLAS_INT lda = n;

    if constexpr (std::is_same_v<Layout, Kokkos::layout_left>) {
        matrix_layout = LAPACK_COL_MAJOR;
        lda = m;
    }

    double res = 0.0;
    if constexpr (std::is_same_v<std::remove_cv_t<T>, double>) {
        res = LAPACKE_dlange(matrix_layout, norm, m, n, a.data_handle(), lda);
    }
    if constexpr (std::is_same_v<std::remove_cv_t<T>, std::complex<double>>) {
        res = LAPACKE_zlange(matrix_layout, norm, m, n, a.data_handle(), lda);
    }
    return res;
}

template <class T,
          class IndexType,
          std::size_t nrows,
          std::size_t ncols,
          class Layout,
          class Container>
    requires(
        (std::is_same_v<std::remove_cv_t<T>, double> ||
         std::is_same_v<std::remove_cv_t<T>, std::complex<double>>) &&std::is_integral_v<IndexType>)
inline double
matrix_norm(const Sci::MDArray<T, Kokkos::extents<IndexType, nrows, ncols>, Layout, Container>& a,
            char norm)
{
    return matrix_norm(a.to_mdspan(), norm);
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_MATRIX_NORM_H

// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_INV_H
#define SCILIB_LINALG_INV_H

#include "lapack_types.h"
#include <exception>
#include <gsl/gsl>
#include <type_traits>

namespace Sci {
namespace Linalg {

// Matrix inversion.
template <class T_a,
          class IndexType,
          std::size_t nrows,
          std::size_t ncols,
          class Layout,
          class Accessor_a,
          class T_res,
          class Accessor_res>
    requires(std::is_same_v<std::remove_cv_t<T_a>, double>&& std::is_integral_v<IndexType>)
inline void
inv(stdex::mdspan<T_a, stdex::extents<IndexType, nrows, ncols>, Layout, Accessor_a> a,
    stdex::mdspan<T_res, stdex::extents<IndexType, nrows, ncols>, Layout, Accessor_res> res)
{
    Expects(a.extent(0) == a.extent(1));

    if (det(a) == 0.0) {
        throw std::runtime_error("inv: matrix not invertible");
    }
    const BLAS_INT n = gsl::narrow_cast<BLAS_INT>(a.extent(0));

    auto matrix_layout = LAPACK_ROW_MAJOR;
    BLAS_INT lda = n;

    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = LAPACK_COL_MAJOR;
    }

    Sci::copy(a, res);

    Sci::Vector<BLAS_INT, Layout> ipiv(n);
    Sci::Linalg::lu(res, ipiv.to_mdspan()); // perform LU factorization

    BLAS_INT info = LAPACKE_dgetri(matrix_layout, n, res.data_handle(), lda, ipiv.container_data());
    if (info != 0) {
        throw std::runtime_error("dgetri: matrix inversion failed");
    }
}

template <class Layout>
inline Sci::Matrix<double, Layout> inv(const Sci::Matrix<double, Layout>& a)
{
    Sci::Matrix<double, Layout> res(a.extent(0), a.extent(1));
    inv(a.to_mdspan(), res.to_mdspan());
    return res;
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_INV_H

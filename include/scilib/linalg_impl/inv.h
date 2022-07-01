// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_INV_H
#define SCILIB_LINALG_INV_H

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif

#include "lapack_types.h"
#include <exception>
#include <cassert>
#include <type_traits>

namespace Sci {
namespace Linalg {

// Matrix inversion.
template <class T_a, class T_res, class Layout>
    requires(std::is_same_v<std::remove_cv_t<T_a>, double>)
inline void inv(Sci::Matrix_view<T_a, Layout> a, Sci::Matrix_view<T_res, Layout> res)
{
    namespace stdex = std::experimental;

    assert(a.extent(0) == a.extent(1));

    if (det(a) == 0.0) {
        throw std::runtime_error("inv: matrix not invertible");
    }
    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(0));

    auto matrix_layout = LAPACK_ROW_MAJOR;
    BLAS_INT lda = n;

    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = LAPACK_COL_MAJOR;
    }

    Sci::copy(a, res);

    Sci::Vector<BLAS_INT, Layout> ipiv(n);
    Sci::Linalg::lu(res, ipiv.view()); // perform LU factorization

    BLAS_INT info = LAPACKE_dgetri(matrix_layout, n, res.data_handle(), lda, ipiv.data());
    if (info != 0) {
        throw std::runtime_error("dgetri: matrix inversion failed");
    }
}

template <class Layout, class Allocator>
inline Sci::Matrix<double, Layout, Allocator> inv(const Sci::Matrix<double, Layout, Allocator>& a)
{
    Sci::Matrix<double, Layout, Allocator> res(a.extent(0), a.extent(1));
    inv(a.view(), res.view());
    return res;
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_INV_H

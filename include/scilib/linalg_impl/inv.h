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

#include <scilib/mdarray.h>
#include <scilib/linalg_impl/lapack_types.h>
#include <experimental/mdspan>
#include <exception>
#include <cassert>
#include <type_traits>

namespace Scilib {
namespace Linalg {

// Matrix inversion.
template <class Layout>
inline void inv(Scilib::Matrix_view<double, Layout> a,
                Scilib::Matrix_view<double, Layout> res)
{
    namespace stdex = std::experimental;

    static_assert(a.is_contiguous());
    static_assert(res.is_contiguous());

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

    Scilib::copy(a, res);

    Scilib::Vector<BLAS_INT, Layout> ipiv(n);
    lu(res, ipiv.view()); // perform LU factorization

    BLAS_INT info =
        LAPACKE_dgetri(matrix_layout, n, res.data(), lda, ipiv.data());
    if (info != 0) {
        throw std::runtime_error("dgetri: matrix inversion failed");
    }
}

template <class Layout>
inline Scilib::Matrix<double, Layout> inv(Scilib::Matrix_view<double, Layout> a)
{
    Scilib::Matrix<double, Layout> res(a.extent(0), a.extent(1));
    inv(a, res.view());
    return res;
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_INV_H

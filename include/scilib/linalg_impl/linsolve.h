// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_LINSOLVE_H
#define SCILIB_LINALG_LINSOLVE_H

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

// Solve linear system of equations.
template <class Layout>
inline void linsolve(Scilib::Matrix_view<double, Layout> a,
                     Scilib::Matrix_view<double, Layout> b)
{
    namespace stdex = std::experimental;

    static_assert(a.is_contiguous());
    static_assert(b.is_contiguous());

    assert(a.extent(0) == a.extent(1));
    assert(b.extent(0) == a.extent(1));

    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(1));
    const BLAS_INT nrhs = static_cast<BLAS_INT>(b.extent(1));
    const BLAS_INT lda = n;

    Scilib::Vector<BLAS_INT, Layout> ipiv(n);

    auto matrix_layout = LAPACK_ROW_MAJOR;
    BLAS_INT ldb = nrhs;

    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = LAPACK_COL_MAJOR;
        ldb = n;
    }
    BLAS_INT info = LAPACKE_dgesv(matrix_layout, n, nrhs, a.data(), lda,
                                  ipiv.data(), b.data(), ldb);
    if (info != 0) {
        throw std::runtime_error("dgesv: factor U is singular");
    }
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_LINSOLVE_H
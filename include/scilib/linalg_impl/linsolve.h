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

#include "lapack_types.h"
#include <cassert>
#include <exception>
#include <type_traits>

namespace Sci {
namespace Linalg {

// Solve linear system of equations.
template <class Layout>
inline void linsolve(Sci::Matrix_view<double, Layout> a, Sci::Matrix_view<double, Layout> b)
{
    namespace stdex = std::experimental;

    assert(a.extent(0) == a.extent(1));
    assert(b.extent(0) == a.extent(1));

    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(1));
    const BLAS_INT nrhs = static_cast<BLAS_INT>(b.extent(1));
    const BLAS_INT lda = n;

    Sci::Vector<BLAS_INT, Layout> ipiv(n);

    auto matrix_layout = LAPACK_ROW_MAJOR;
    BLAS_INT ldb = nrhs;

    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = LAPACK_COL_MAJOR;
        ldb = n;
    }
    BLAS_INT info = LAPACKE_dgesv(matrix_layout, n, nrhs, a.data_handle(), lda, ipiv.data(),
                                  b.data_handle(), ldb);
    if (info != 0) {
        throw std::runtime_error("dgesv: factor U is singular");
    }
}

template <class Layout, class Allocator>
inline void linsolve(Sci::Matrix<double, Layout, Allocator>& a,
                     Sci::Matrix<double, Layout, Allocator>& b)
{
    linsolve(a.view(), b.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_LINSOLVE_H
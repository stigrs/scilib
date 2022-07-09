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
template <std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Accessor_a,
          std::size_t nrows_b,
          std::size_t ncols_b,
          class Accessor_b>
inline void
linsolve(stdex::mdspan<double, stdex::extents<index, nrows_a, ncols_a>, Layout, Accessor_a> a,
         stdex::mdspan<double, stdex::extents<index, nrows_b, ncols_b>, Layout, Accessor_b> b)
{
    Expects(a.extent(0) == a.extent(1));
    Expects(b.extent(0) == a.extent(1));

    const BLAS_INT n = gsl::narrow<BLAS_INT>(a.extent(1));
    const BLAS_INT nrhs = gsl::narrow<BLAS_INT>(b.extent(1));
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

template <class Layout, class Container>
inline void linsolve(Sci::Matrix<double, Layout, Container>& a,
                     Sci::Matrix<double, Layout, Container>& b)
{
    linsolve(a.view(), b.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_LINSOLVE_H
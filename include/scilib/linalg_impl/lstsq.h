// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_LSTSQ_H
#define SCILIB_LINALG_LSTSQ_H

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif

#include "lapack_types.h"
#include <algorithm>
#include <exception>
#include <type_traits>

namespace Sci {
namespace Linalg {

// Compute the minimum norm-solution to a real linear least squares problem.
template <std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Accessor_a,
          std::size_t nrows_b,
          std::size_t ncols_b,
          class Accessor_b>
inline void
lstsq(stdex::mdspan<double, stdex::extents<index, nrows_a, ncols_a>, Layout, Accessor_a> a,
      stdex::mdspan<double, stdex::extents<index, nrows_b, ncols_b>, Layout, Accessor_b> b)
{
    BLAS_INT m = gsl::narrow_cast<BLAS_INT>(a.extent(0));
    BLAS_INT n = gsl::narrow_cast<BLAS_INT>(a.extent(1));
    BLAS_INT nrhs = gsl::narrow_cast<BLAS_INT>(b.extent(1));
    BLAS_INT rank;

    double rcond = -1.0;                           // use machine epsilon
    Sci::Vector<double, Layout> s(std::min(m, n)); // singular values of a

    auto matrix_layout = LAPACK_ROW_MAJOR;
    BLAS_INT lda = n;
    BLAS_INT ldb = nrhs;

    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = LAPACK_COL_MAJOR;
        lda = m;
        ldb = n;
    }
    BLAS_INT info = LAPACKE_dgelsd(matrix_layout, m, n, nrhs, a.data_handle(), lda, b.data_handle(),
                                   ldb, s.data(), rcond, &rank);
    if (info != 0) {
        throw std::runtime_error("dgelsd failed");
    }
}

template <class Layout, class Container>
inline void lstsq(Sci::Matrix<double, Layout, Container>& a,
                  Sci::Matrix<double, Layout, Container>& b)
{
    lstsq(a.view(), b.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_LSTSQ_H

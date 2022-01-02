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

#include <scilib/mdarray.h>
#include <scilib/linalg_impl/lapack_types.h>
#include <experimental/mdspan>
#include <exception>
#include <algorithm>
#include <type_traits>

namespace Sci {
namespace Linalg {

// Compute the minimum norm-solution to a real linear least squares problem.
template <class Layout>
inline void lstsq(Sci::Matrix_view<double, Layout> a,
                  Sci::Matrix_view<double, Layout> b)
{
    namespace stdex = std::experimental;

    static_assert(a.is_contiguous());
    static_assert(b.is_contiguous());

    BLAS_INT m = static_cast<BLAS_INT>(a.extent(0));
    BLAS_INT n = static_cast<BLAS_INT>(a.extent(1));
    BLAS_INT nrhs = static_cast<BLAS_INT>(b.extent(1));
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
    BLAS_INT info = LAPACKE_dgelsd(matrix_layout, m, n, nrhs, a.data(), lda,
                                   b.data(), ldb, s.data(), rcond, &rank);
    if (info != 0) {
        throw std::runtime_error("dgelsd failed");
    }
}

template <class Layout, class Allocator>
inline void lstsq(Sci::Matrix<double, Layout, Allocator>& a,
                  Sci::Matrix<double, Layout, Allocator>& b)
{
    lstsq(a.view(), b.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_LSTSQ_H

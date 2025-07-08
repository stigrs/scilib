// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_LSTSQ_H
#define SCILIB_LINALG_LSTSQ_H

#include "lapack_types.h"
#include <algorithm>
#include <exception>
#include <type_traits>

namespace Sci {
namespace Linalg {

// Compute the minimum norm-solution to a real linear least squares problem.
template <class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Accessor_a,
          class IndexType_b,
          std::size_t nrows_b,
          std::size_t ncols_b,
          class Accessor_b>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_b>)
inline void
lstsq(Kokkos::mdspan<double, Kokkos::extents<IndexType_a, nrows_a, ncols_a>, Layout, Accessor_a> a,
      Kokkos::mdspan<double, Kokkos::extents<IndexType_b, nrows_b, ncols_b>, Layout, Accessor_b> b)
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

    if constexpr (std::is_same_v<Layout, Kokkos::layout_left>) {
        matrix_layout = LAPACK_COL_MAJOR;
        lda = m;
        ldb = n;
    }
    BLAS_INT info = LAPACKE_dgelsd(matrix_layout, m, n, nrhs, a.data_handle(), lda, b.data_handle(),
                                   ldb, s.container_data(), rcond, &rank);
    if (info != 0) {
        throw std::runtime_error("dgelsd failed");
    }
}

template <class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Container_a,
          class IndexType_b,
          std::size_t nrows_b,
          std::size_t ncols_b,
          class Container_b>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_b>)
inline void
lstsq(Sci::MDArray<double, Kokkos::extents<IndexType_a, nrows_a, ncols_a>, Layout, Container_a>& a,
      Sci::MDArray<double, Kokkos::extents<IndexType_b, nrows_b, ncols_b>, Layout, Container_b>& b)
{
    lstsq(a.to_mdspan(), b.to_mdspan());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_LSTSQ_H

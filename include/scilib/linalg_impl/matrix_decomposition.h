// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_MATRIX_DECOMPOSITION_H
#define SCILIB_LINALG_MATRIX_DECOMPOSITION_H

#include "lapack_types.h"
#include <cassert>
#include <exception>
#include <experimental/linalg>
#include <type_traits>

namespace Sci {
namespace Linalg {

namespace stdex = std::experimental;

// LU factorization.
template <std::size_t nrows,
          std::size_t ncols,
          class Layout,
          class Accessor_a,
          std::size_t ext_ipiv,
          class Accessor_ipiv>
inline void lu(stdex::mdspan<double, stdex::extents<index, nrows, ncols>, Layout, Accessor_a> a,
               stdex::mdspan<BLAS_INT, stdex::extents<index, ext_ipiv>, Layout, Accessor_ipiv> ipiv)
{
    const BLAS_INT m = gsl::narrow_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = gsl::narrow_cast<BLAS_INT>(a.extent(1));

    Expects(gsl::narrow_cast<BLAS_INT>(ipiv.size()) >= std::min(m, n));

    auto matrix_layout = LAPACK_ROW_MAJOR;
    BLAS_INT lda = n;

    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = LAPACK_COL_MAJOR;
        lda = m;
    }

    BLAS_INT info = LAPACKE_dgetrf(matrix_layout, m, n, a.data_handle(), lda, ipiv.data_handle());
    if (info < 0) {
        throw std::runtime_error("dgetrf: illegal input parameter");
    }
    if (info > 0) {
        throw std::runtime_error("dgetrf: U matrix is singular");
    }
}

template <class Layout, class Container_a, class Container_ipiv>
inline void lu(Sci::Matrix<double, Layout, Container_a>& a,
               Sci::Vector<BLAS_INT, Layout, Container_ipiv>& ipiv)
{
    lu(a.view(), ipiv.view());
}

// QR factorization.
template <std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Accessor_a,
          std::size_t nrows_q,
          std::size_t ncols_q,
          class Accessor_q,
          std::size_t nrows_r,
          std::size_t ncols_r,
          class Accessor_r>
inline void qr(stdex::mdspan<double, stdex::extents<index, nrows_a, ncols_a>, Layout, Accessor_a> a,
               stdex::mdspan<double, stdex::extents<index, nrows_q, ncols_q>, Layout, Accessor_q> q,
               stdex::mdspan<double, stdex::extents<index, nrows_r, ncols_r>, Layout, Accessor_r> r)
{
    Expects(q.extent(0) == a.extent(0) && q.extent(1) == a.extent(1));
    Expects(r.extent(0) == a.extent(0) && r.extent(1) == a.extent(1));

    const BLAS_INT m = gsl::narrow_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = gsl::narrow_cast<BLAS_INT>(a.extent(1));

    auto matrix_layout = LAPACK_ROW_MAJOR;
    BLAS_INT lda = n;

    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = LAPACK_COL_MAJOR;
        lda = m;
    }
    Sci::copy(a, q);
    Sci::Vector<double, Layout> tau(std::min(m, n));

    // Compute QR factorization:

    BLAS_INT info = LAPACKE_dgeqrf(matrix_layout, m, n, q.data_handle(), lda, tau.data());
    if (info != 0) {
        throw std::runtime_error("dgeqrf failed");
    }

    // Compute Q:

    info = LAPACKE_dorgqr(matrix_layout, m, n, n, q.data_handle(), lda, tau.data());
    if (info != 0) {
        throw std::runtime_error("dorgqr failed");
    }

    // Compute R:

    matrix_product(std::experimental::linalg::transposed(q), a, r);
    std::experimental::linalg::transposed(q);
}

template <class Layout, class Container>
inline void qr(Sci::Matrix<double, Layout, Container>& a,
               Sci::Matrix<double, Layout, Container>& q,
               Sci::Matrix<double, Layout, Container>& r)
{
    qr(a.view(), q.view(), r.view());
}

// Singular value decomposition.
template <std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Accessor_a,
          std::size_t ext_s,
          class Accessor_s,
          std::size_t nrows_u,
          std::size_t ncols_u,
          class Accessor_u,
          std::size_t nrows_vt,
          std::size_t ncols_vt,
          class Accessor_vt>
inline void
svd(stdex::mdspan<double, stdex::extents<index, nrows_a, ncols_a>, Layout, Accessor_a> a,
    stdex::mdspan<double, stdex::extents<index, ext_s>, Layout, Accessor_s> s,
    stdex::mdspan<double, stdex::extents<index, nrows_u, ncols_u>, Layout, Accessor_u> u,
    stdex::mdspan<double, stdex::extents<index, nrows_vt, ncols_vt>, Layout, Accessor_vt> vt)
{
    const BLAS_INT m = gsl::narrow_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = gsl::narrow_cast<BLAS_INT>(a.extent(1));
    const BLAS_INT ldu = m;
    const BLAS_INT ldvt = n;

    Expects(gsl::narrow_cast<BLAS_INT>(s.extent(0)) == std::min(m, n));
    Expects(gsl::narrow_cast<BLAS_INT>(u.extent(0)) == m);
    Expects(gsl::narrow_cast<BLAS_INT>(u.extent(1)) == ldu);
    Expects(gsl::narrow_cast<BLAS_INT>(vt.extent(0)) == n);
    Expects(gsl::narrow_cast<BLAS_INT>(vt.extent(1)) == ldvt);

    auto matrix_layout = LAPACK_ROW_MAJOR;
    BLAS_INT lda = n;

    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = LAPACK_COL_MAJOR;
        lda = m;
    }

    Sci::Vector<double, Layout> superb(std::min(m, n) - 1);

    BLAS_INT info =
        LAPACKE_dgesvd(matrix_layout, 'A', 'A', m, n, a.data_handle(), lda, s.data_handle(),
                       u.data_handle(), ldu, vt.data_handle(), ldvt, superb.data());
    if (info != 0) {
        throw std::runtime_error("dgesvd failed");
    }
}

template <class Layout, class Container>
inline void svd(Sci::Matrix<double, Layout, Container>& a,
                Sci::Vector<double, Layout, Container>& s,
                Sci::Matrix<double, Layout, Container>& u,
                Sci::Matrix<double, Layout, Container>& vt)
{
    svd(a.view(), s.view(), u.view(), vt.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_MATRIX_DECOMPOSITION_H

// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_MATRIX_DECOMPOSITION_H
#define SCILIB_LINALG_MATRIX_DECOMPOSITION_H

#include "auxiliary.h"
#include "lapack_types.h"
#include <cassert>
#include <exception>
#include <experimental/linalg>
#include <type_traits>

namespace Sci {
namespace Linalg {

namespace stdex = std::experimental;

// Cholesky factorization.
template <class IndexType, std::size_t nrows, std::size_t ncols, class Layout, class Accessor>
    requires(std::is_integral_v<IndexType>)
inline void
cholesky(stdex::mdspan<double, stdex::extents<IndexType, nrows, ncols>, Layout, Accessor> a)
{
    Expects(a.extent(0) == a.extent(1));

    const BLAS_INT lda = gsl::narrow_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = gsl::narrow_cast<BLAS_INT>(a.extent(1));

    auto matrix_layout = LAPACK_ROW_MAJOR;
    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = LAPACK_COL_MAJOR;
    }
    to_lower_triangular(a);

    BLAS_INT info = LAPACKE_dpotrf(matrix_layout, 'L', n, a.data_handle(), lda);
    if (info < 0) {
        throw std::runtime_error("dgetrf: illegal input parameter");
    }
    if (info > 0) {
        throw std::runtime_error("dpotrf: A matrix is not positive-definitive");
    }
}

template <class IndexType, std::size_t nrows, std::size_t ncols, class Layout, class Container>
    requires(std::is_integral_v<IndexType>)
inline void
cholesky(Sci::MDArray<double, stdex::extents<IndexType, nrows, ncols>, Layout, Container>& a)
{
    cholesky(a.to_mdspan());
}

// LU factorization.
template <class IndexType_a,
          std::size_t nrows,
          std::size_t ncols,
          class Layout,
          class Accessor_a,
          class IndexType_ipiv,
          std::size_t ext_ipiv,
          class Accessor_ipiv>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_ipiv>)
inline void
lu(stdex::mdspan<double, stdex::extents<IndexType_a, nrows, ncols>, Layout, Accessor_a> a,
   stdex::mdspan<BLAS_INT, stdex::extents<IndexType_ipiv, ext_ipiv>, Layout, Accessor_ipiv> ipiv)
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

template <class IndexType_a,
          std::size_t nrows,
          std::size_t ncols,
          class Layout,
          class Container_a,
          class IndexType_ipiv,
          std::size_t ext_ipiv,
          class Container_ipiv>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_ipiv>)
inline void
lu(Sci::MDArray<double, stdex::extents<IndexType_a, nrows, ncols>, Layout, Container_a>& a,
   Sci::MDArray<BLAS_INT, stdex::extents<IndexType_ipiv, ext_ipiv>, Layout, Container_ipiv>& ipiv)
{
    lu(a.to_mdspan(), ipiv.to_mdspan());
}

// QR factorization.
template <class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Accessor_a,
          class IndexType_q,
          std::size_t nrows_q,
          std::size_t ncols_q,
          class Accessor_q,
          class IndexType_r,
          std::size_t nrows_r,
          std::size_t ncols_r,
          class Accessor_r>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_q>&&
                 std::is_integral_v<IndexType_r>)
inline void
qr(stdex::mdspan<double, stdex::extents<IndexType_a, nrows_a, ncols_a>, Layout, Accessor_a> a,
   stdex::mdspan<double, stdex::extents<IndexType_q, nrows_q, ncols_q>, Layout, Accessor_q> q,
   stdex::mdspan<double, stdex::extents<IndexType_r, nrows_r, ncols_r>, Layout, Accessor_r> r)
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

    BLAS_INT info = LAPACKE_dgeqrf(matrix_layout, m, n, q.data_handle(), lda, tau.container_data());
    if (info != 0) {
        throw std::runtime_error("dgeqrf failed");
    }

    // Compute Q:

    info = LAPACKE_dorgqr(matrix_layout, m, n, n, q.data_handle(), lda, tau.container_data());
    if (info != 0) {
        throw std::runtime_error("dorgqr failed");
    }

    // Compute R:

    matrix_product(std::experimental::linalg::transposed(q), a, r);
    std::experimental::linalg::transposed(q);
}

template <class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Container_a,
          class IndexType_q,
          std::size_t nrows_q,
          std::size_t ncols_q,
          class Container_q,
          class IndexType_r,
          std::size_t nrows_r,
          std::size_t ncols_r,
          class Container_r>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_q>&&
                 std::is_integral_v<IndexType_r>)
inline void
qr(Sci::MDArray<double, stdex::extents<IndexType_a, nrows_a, ncols_a>, Layout, Container_a>& a,
   Sci::MDArray<double, stdex::extents<IndexType_q, nrows_q, ncols_q>, Layout, Container_q>& q,
   Sci::MDArray<double, stdex::extents<IndexType_r, nrows_r, ncols_r>, Layout, Container_r>& r)
{
    qr(a.to_mdspan(), q.to_mdspan(), r.to_mdspan());
}

// Singular value decomposition.
template <class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Accessor_a,
          class IndexType_s,
          std::size_t ext_s,
          class Accessor_s,
          class IndexType_u,
          std::size_t nrows_u,
          std::size_t ncols_u,
          class Accessor_u,
          class IndexType_vt,
          std::size_t nrows_vt,
          std::size_t ncols_vt,
          class Accessor_vt>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_s>&&
                 std::is_integral_v<IndexType_u>&& std::is_integral_v<IndexType_vt>)
inline void
svd(stdex::mdspan<double, stdex::extents<IndexType_a, nrows_a, ncols_a>, Layout, Accessor_a> a,
    stdex::mdspan<double, stdex::extents<IndexType_s, ext_s>, Layout, Accessor_s> s,
    stdex::mdspan<double, stdex::extents<IndexType_u, nrows_u, ncols_u>, Layout, Accessor_u> u,
    stdex::mdspan<double, stdex::extents<IndexType_vt, nrows_vt, ncols_vt>, Layout, Accessor_vt> vt)
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
                       u.data_handle(), ldu, vt.data_handle(), ldvt, superb.container_data());
    if (info != 0) {
        throw std::runtime_error("dgesvd failed");
    }
}

template <class IndexType_a,
          std::size_t nrows_a,
          std::size_t ncols_a,
          class Layout,
          class Container_a,
          class IndexType_s,
          std::size_t ext_s,
          class Container_s,
          class IndexType_u,
          std::size_t nrows_u,
          std::size_t ncols_u,
          class Container_u,
          class IndexType_vt,
          std::size_t nrows_vt,
          std::size_t ncols_vt,
          class Container_vt>
    requires(std::is_integral_v<IndexType_a>&& std::is_integral_v<IndexType_s>&&
                 std::is_integral_v<IndexType_u>&& std::is_integral_v<IndexType_vt>)
inline void
svd(Sci::MDArray<double, stdex::extents<IndexType_a, nrows_a, ncols_a>, Layout, Container_a>& a,
    Sci::MDArray<double, stdex::extents<IndexType_s, ext_s>, Layout, Container_s>& s,
    Sci::MDArray<double, stdex::extents<IndexType_u, nrows_u, ncols_u>, Layout, Container_u>& u,
    Sci::MDArray<double, stdex::extents<IndexType_vt, nrows_vt, ncols_vt>, Layout, Container_vt>&
        vt)
{
    svd(a.to_mdspan(), s.to_mdspan(), u.to_mdspan(), vt.to_mdspan());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_MATRIX_DECOMPOSITION_H

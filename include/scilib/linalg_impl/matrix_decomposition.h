// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_MATRIX_DECOMPOSITION_H
#define SCILIB_LINALG_MATRIX_DECOMPOSITION_H

#ifdef USE_MKL
#include <mkl.h>
#else
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif
#include <lapacke.h>
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#endif

#include "lapack_types.h"
#include <cassert>
#include <exception>
#include <type_traits>

namespace Sci {
namespace Linalg {

namespace stdex = std::experimental;

// LU factorization.
template <class Layout>
inline void lu(Sci::Matrix_view<double, Layout> a, Sci::Vector_view<BLAS_INT, Layout> ipiv)
{
    const BLAS_INT m = static_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(1));

    assert(static_cast<BLAS_INT>(ipiv.size()) >= std::min(m, n));

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

template <class Layout, class Allocator_a, class Allocator_ipiv>
inline void lu(Sci::Matrix<double, Layout, Allocator_a>& a,
               Sci::Vector<BLAS_INT, Layout, Allocator_ipiv>& ipiv)
{
    lu(a.view(), ipiv.view());
}

// QR factorization.
template <class Layout>
inline void qr(Sci::Matrix_view<double, Layout> a,
               Sci::Matrix_view<double, Layout> q,
               Sci::Matrix_view<double, Layout> r)
{
    assert(q.extent(0) == a.extent(0) && q.extent(1) == a.extent(1));
    assert(r.extent(0) == a.extent(0) && r.extent(1) == a.extent(1));

    const BLAS_INT m = static_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(1));

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

    matrix_product(transposed(q), a, r);
    transposed(q);
}

template <class Layout, class Allocator>
inline void qr(Sci::Matrix<double, Layout, Allocator>& a,
               Sci::Matrix<double, Layout, Allocator>& q,
               Sci::Matrix<double, Layout, Allocator>& r)
{
    qr(a.view(), q.view(), r.view());
}

// Singular value decomposition.
template <class Layout>
inline void svd(Sci::Matrix_view<double, Layout> a,
                Sci::Vector_view<double, Layout> s,
                Sci::Matrix_view<double, Layout> u,
                Sci::Matrix_view<double, Layout> vt)
{
    const BLAS_INT m = static_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(1));
    const BLAS_INT ldu = m;
    const BLAS_INT ldvt = n;

    assert(static_cast<BLAS_INT>(s.extent(0)) == std::min(m, n));
    assert(static_cast<BLAS_INT>(u.extent(0)) == m);
    assert(static_cast<BLAS_INT>(u.extent(1)) == ldu);
    assert(static_cast<BLAS_INT>(vt.extent(0)) == n);
    assert(static_cast<BLAS_INT>(vt.extent(1)) == ldvt);

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

template <class Layout, class Allocator>
inline void svd(Sci::Matrix<double, Layout, Allocator>& a,
                Sci::Vector<double, Layout, Allocator>& s,
                Sci::Matrix<double, Layout, Allocator>& u,
                Sci::Matrix<double, Layout, Allocator>& vt)
{
    svd(a.view(), s.view(), u.view(), vt.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_MATRIX_DECOMPOSITION_H

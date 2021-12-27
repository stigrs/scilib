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

namespace stdex = std::experimental;

// LU factorization.
template <class Layout>
inline void lu(Scilib::Matrix_view<double, Layout> a,
               Scilib::Vector_view<BLAS_INT, Layout> ipiv)
{
    static_assert(a.is_contiguous());
    static_assert(ipiv.is_contiguous());

    const BLAS_INT m = static_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(1));

    assert(static_cast<BLAS_INT>(ipiv.size()) >= std::min(m, n));

    auto matrix_layout = LAPACK_ROW_MAJOR;
    BLAS_INT lda = n;

    if constexpr (std::is_same_v<Layout, stdex::layout_left>) {
        matrix_layout = LAPACK_COL_MAJOR;
        lda = m;
    }

    BLAS_INT info =
        LAPACKE_dgetrf(matrix_layout, m, n, a.data(), lda, ipiv.data());
    if (info < 0) {
        throw std::runtime_error("dgetrf: illegal input parameter");
    }
    if (info > 0) {
        throw std::runtime_error("dgetrf: U matrix is singular");
    }
}

// QR factorization.
template <class Layout>
inline void qr(Scilib::Matrix_view<double, Layout> a,
               Scilib::Matrix_view<double, Layout> q,
               Scilib::Matrix_view<double, Layout> r)
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
    Scilib::copy(a, q);
    Scilib::Vector<double, Layout> tau(std::min(m, n));

    // Compute QR factorization:

    BLAS_INT info =
        LAPACKE_dgeqrf(matrix_layout, m, n, q.data(), lda, tau.data());
    if (info != 0) {
        throw std::runtime_error("dgeqrf failed");
    }

    // Compute Q:

    info = LAPACKE_dorgqr(matrix_layout, m, n, n, q.data(), lda, tau.data());
    if (info != 0) {
        throw std::runtime_error("dorgqr failed");
    }

    // Compute R:

    matrix_product(transposed(q), a, r);
    transposed(q);
}

// Singular value decomposition.
template <class Layout>
inline void svd(Scilib::Matrix_view<double, Layout> a,
                Scilib::Vector_view<double, Layout> s,
                Scilib::Matrix_view<double, Layout> u,
                Scilib::Matrix_view<double, Layout> vt)
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

    Scilib::Vector<double, Layout> superb(std::min(m, n) - 1);

    BLAS_INT info =
        LAPACKE_dgesvd(matrix_layout, 'A', 'A', m, n, a.data(), lda, s.data(),
                       u.data(), ldu, vt.data(), ldvt, superb.data());
    if (info != 0) {
        throw std::runtime_error("dgesvd failed");
    }
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_MATRIX_DECOMPOSITION_H

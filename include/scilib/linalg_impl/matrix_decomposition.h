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
#include <exception>
#include <cassert>

namespace Scilib {
namespace Linalg {

// LU factorization.
inline void lu(Scilib::Matrix_view<double> a,
               Scilib::Vector_view<BLAS_INT> ipiv)
{
    static_assert(a.is_contiguous());
    static_assert(ipiv.is_contiguous());

    const BLAS_INT m = static_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(1));
    const BLAS_INT lda = n;

    assert(static_cast<BLAS_INT>(ipiv.size()) >= std::min(m, n));

    BLAS_INT info =
        LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, a.data(), lda, ipiv.data());
    if (info < 0) {
        throw std::runtime_error("dgetrf: illegal input parameter");
    }
    if (info > 0) {
        throw std::runtime_error("dgetrf: U matrix is singular");
    }
}

// QR factorization.
inline void qr(Scilib::Matrix_view<double> a,
               Scilib::Matrix_view<double> q,
               Scilib::Matrix_view<double> r)
{
    const BLAS_INT m = static_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(1));
    const BLAS_INT lda = n;

    assert(q.extent(0) == a.extent(0) && q.extent(1) == a.extent(1));
    assert(r.extent(0) == a.extent(0) && r.extent(1) == a.extent(1));

    // Compute QR factorization:

    Scilib::copy(a, q);
    Scilib::Vector<double> tau(std::min(m, n));

    BLAS_INT info =
        LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, q.data(), lda, tau.data());
    if (info != 0) {
        throw std::runtime_error("dgeqrf failed");
    }

    // Compute Q:

    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, n, n, q.data(), lda, tau.data());
    if (info != 0) {
        throw std::runtime_error("dorgqr failed");
    }

    // Compute R:

    matrix_product(transposed(q), a, r);
    transposed(q);
}

// Singular value decomposition.
inline void svd(Scilib::Matrix_view<double> a,
                Scilib::Vector_view<double> s,
                Scilib::Matrix_view<double> u,
                Scilib::Matrix_view<double> vt)
{
    BLAS_INT m = static_cast<BLAS_INT>(a.extent(0));
    BLAS_INT n = static_cast<BLAS_INT>(a.extent(1));
    BLAS_INT lda = n;
    BLAS_INT ldu = m;
    BLAS_INT ldvt = n;

    assert(static_cast<BLAS_INT>(s.extent(0)) == std::min(m, n));
    assert(static_cast<BLAS_INT>(u.extent(0)) == m);
    assert(static_cast<BLAS_INT>(u.extent(1)) == ldu);
    assert(static_cast<BLAS_INT>(vt.extent(0)) == n);
    assert(static_cast<BLAS_INT>(vt.extent(1)) == ldvt);

    Scilib::Vector<double> superb(std::min(m, n) - 1);

    BLAS_INT info =
        LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, a.data(), lda,
                       s.data(), u.data(), ldu, vt.data(), ldvt, superb.data());
    if (info != 0) {
        throw std::runtime_error("dgesvd failed");
    }
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_MATRIX_DECOMPOSITION_H

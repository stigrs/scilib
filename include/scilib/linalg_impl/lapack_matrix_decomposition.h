// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#pragma once

#include <scilib/linalg.h>
#include <scilib/traits.h>
#include <exception>
#include <cassert>
#include <iostream>

namespace Scilib {
namespace Linalg {

// LU factorization.
inline void lu(Matrix_view<double> a, Vector_view<BLAS_INT> ipiv)
{
    const BLAS_INT m = narrow_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = narrow_cast<BLAS_INT>(a.extent(1));
    const BLAS_INT lda = n;

    assert(ipiv.size() >= std::min(m, n));

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
inline void
qr(Matrix_view<double> a, Matrix_view<double> q, Matrix_view<double> r)
{
    const BLAS_INT m = narrow_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = narrow_cast<BLAS_INT>(a.extent(1));
    const BLAS_INT lda = n;

    assert(q.extent(0) == m && q.extent(1) == n);
    assert(r.extent(0) == m && r.extent(1) == n);

    // Compute QR factorization:

    Vector<double> tau(std::min(m, n));

    Matrix<double> qq(a); // work on a local copy

    BLAS_INT info =
        LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, qq.data(), lda, tau.data());
    if (info != 0) {
        throw std::runtime_error("dgeqrf failed");
    }

    // Compute Q:

    info =
        LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, n, n, qq.data(), lda, tau.data());
    if (info != 0) {
        throw std::runtime_error("dorgqr failed");
    }

    // Compute R:

    Matrix<double> qt(transposed(qq.view())); // need a deep copy
    matrix_matrix_product(qt.view(), a, r);
    copy(qq.view(), q); // copy result back to q
}

#if 0
// Singular value decomposition.
inline void svd(Mat<double>& a, Vec<double>& s, Mat<double>& u, Mat<double>& vt)
{
    BLAS_INT m = narrow_cast<BLAS_INT>(a.rows());
    BLAS_INT n = narrow_cast<BLAS_INT>(a.cols());
    BLAS_INT lda = n;
    BLAS_INT ldu = m;
    BLAS_INT ldvt = n;

    s.resize(std::min(m, n));
    u.resize(m, ldu);
    vt.resize(n, ldvt);

    Vec<double> superb(std::min(m, n) - 1);

    BLAS_INT info =
        LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, a.data(), lda,
                       s.data(), u.data(), ldu, vt.data(), ldvt, superb.data());
    if (info != 0) {
        throw Math_error("dgesvd failed");
    }
}
#endif
} // namespace Linalg
} // namespace Scilib

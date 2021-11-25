// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#pragma once

#include <scilib/mdarray.h>
#include <scilib/mdarray_impl/operations.h>
#include <scilib/traits.h>
#include <scilib/linalg_impl/blas3_matrix_product.h>
#include <scilib/linalg_impl/transposed.h>
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
    assert(q.extent(0) == m && q.extent(1) == n);
    assert(r.extent(0) == m && r.extent(1) == n);

    // Compute QR factorization:

    const BLAS_INT m = narrow_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT n = narrow_cast<BLAS_INT>(a.extent(1));
    const BLAS_INT lda = n;

    Vector<double> tau(std::min(m, n));

    q = Matrix<double>(a).view();

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
    Scilib::print(std::cout, q);

    // Compute R:

    matrix_matrix_product(transposed(q), a, r);
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

// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_EIGENVALUE_H
#define SCILIB_LINALG_EIGENVALUE_H

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif

#include <scilib/mdarray.h>
#include <scilib/linalg_impl/lapack_types.h>
#include <exception>
#include <cassert>
#include <complex>

namespace Scilib {
namespace Linalg {

// Compute eigenvalues and eigenvectors of a real symmetric matrix.
inline void eigs(Scilib::Matrix_view<double> a,
                 Scilib::Vector_view<double> w,
                 double abstol = -1.0 /* use default value */)
{
    static_assert(a.is_contiguous());
    static_assert(w.is_contiguous());

    assert(a.extent(0) == a.extent(1));
    assert(w.extent(0) == a.extent(0));

    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(0));
    const BLAS_INT nselect = n;
    const BLAS_INT lda = n;
    const BLAS_INT ldz = nselect;

    BLAS_INT il = 1;
    BLAS_INT iu = n;
    BLAS_INT m;
    BLAS_INT info;

    double vl = 0.0;
    double vu = 0.0;

    Scilib::Vector<BLAS_INT> isuppz(2 * n);
    Scilib::Matrix<double> z(ldz, n);

    info = LAPACKE_dsyevr(LAPACK_ROW_MAJOR, 'V', 'A', 'U', n, a.data(), lda, vl,
                          vu, il, iu, abstol, &m, w.data(), z.data(), ldz,
                          isuppz.data());
    if (info != 0) {
        throw std::runtime_error("dsyevr failed");
    }
    Scilib::copy(z.view(), a);
}

// Compute eigenvalues and eigenvectors of a real non-symmetric matrix.
void eig(Scilib::Matrix_view<double> a,
         Scilib::Matrix_view<std::complex<double>> evec,
         Scilib::Vector_view<std::complex<double>> eval);

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_EIGENVALUE_H

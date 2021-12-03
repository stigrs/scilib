// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_LINSOLVE_H
#define SCILIB_LINALG_LINSOLVE_H

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

// Solve linear system of equations.
inline void linsolve(Scilib::Matrix_view<double> a,
                     Scilib::Matrix_view<double> b)
{
    assert(a.extent(0) == a.extent(1));
    assert(b.extent(0) == a.extent(1));

    const BLAS_INT n = static_cast<BLAS_INT>(a.extent(1));
    const BLAS_INT nrhs = static_cast<BLAS_INT>(b.extent(1));
    const BLAS_INT lda = n;
    const BLAS_INT ldb = nrhs;

    Scilib::Vector<BLAS_INT> ipiv(n);

    BLAS_INT info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, a.data(), lda,
                                  ipiv.data(), b.data(), ldb);
    if (info != 0) {
        throw std::runtime_error("dgesv: factor U is singular");
    }
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_LINSOLVE_H
// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_LSTSQ_H
#define SCILIB_LINALG_LSTSQ_H

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif

#include <scilib/mdarray.h>
#include <scilib/linalg_impl/lapack_types.h>
#include <exception>
#include <algorithm>

namespace Scilib {
namespace Linalg {

// Compute the minimum norm-solution to a real linear least squares problem.
inline void lstsq(Scilib::Matrix_view<double> a, Scilib::Matrix_view<double> b)
{
    static_assert(a.is_contiguous());
    static_assert(b.is_contiguous());

    BLAS_INT m = static_cast<BLAS_INT>(a.extent(0));
    BLAS_INT n = static_cast<BLAS_INT>(a.extent(1));
    BLAS_INT nrhs = static_cast<BLAS_INT>(b.extent(1));
    BLAS_INT lda = n;
    BLAS_INT ldb = nrhs;
    BLAS_INT rank;

    double rcond = -1.0;                      // use machine epsilon
    Scilib::Vector<double> s(std::min(m, n)); // singular values of a

    BLAS_INT info = LAPACKE_dgelsd(LAPACK_ROW_MAJOR, m, n, nrhs, a.data(), lda,
                                   b.data(), ldb, s.data(), rcond, &rank);
    if (info != 0) {
        throw std::runtime_error("dgelsd failed");
    }
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_LSTSQ_H

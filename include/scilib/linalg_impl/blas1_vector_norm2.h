// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_VECTOR_NORM2_H
#define SCILIB_LINALG_BLAS1_VECTOR_NORM2_H

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

#include <scilib/traits.h>

namespace Scilib {
namespace Linalg {

inline double norm2(Vector_view<double> x)
{
    const BLAS_INT n = narrow_cast<BLAS_INT>(x.size());
    const BLAS_INT incx = narrow_cast<BLAS_INT>(x.stride(0));

    return cblas_dnrm2(n, x.data(), incx);
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_BLAS1_VECTOR_NORM2_H

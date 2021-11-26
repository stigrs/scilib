// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#pragma once

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

#include <scilib/mdarray_impl/type_aliases.h>
#include <scilib/traits.h>
#include <exception>
#include <cassert>

namespace Scilib {
namespace Linalg {

// Compute eigenvalues and eigenvectors of a real symmetric matrix.
inline void eigs(Matrix_view<double> a, Vector_view<double> w)
{
    assert(a.extent(0) == a.extent(1));
    assert(w.extent(0) == a.extent(0));

    const BLAS_INT n = narrow_cast<BLAS_INT>(a.extent(0));

    BLAS_INT info =
        LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', n, a.data(), n, w.data());
    if (info != 0) {
        throw std::runtime_error("dsyevd failed");
    }
}

} // namespace Linalg
} // namespace Scilib

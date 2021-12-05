// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_SORT_H
#define SCILIB_LINALG_SORT_H

#ifdef USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif

#include <experimental/mdspan>
#include <scilib/linalg_impl/lapack_types.h>
#include <exception>

namespace Scilib {
namespace Linalg {
namespace stdex = std::experimental;

template <stdex::extents<>::size_type ext, class Layout, class Accessor>
inline void sort(stdex::mdspan<double, stdex::extents<ext>, Layout, Accessor> x,
                 char id = 'I')
{
    const BLAS_INT n = static_cast<BLAS_INT>(x.extent(0));

    BLAS_INT info = LAPACKE_dlasrt(id, n, x.data());
    if (info != 0) {
        throw std::runtime_error("bad input to dlasrt");
    }
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_SORT_H

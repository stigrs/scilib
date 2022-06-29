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

namespace Sci {
namespace Linalg {

namespace stdex = std::experimental;

template <std::size_t ext_x, class Layout_x, class Accessor_x>
inline double
norm2(stdex::mdspan<double, stdex::extents<std::size_t, ext_x>, Layout_x, Accessor_x> x)
{
    const BLAS_INT n = static_cast<BLAS_INT>(x.size());
    const BLAS_INT incx = static_cast<BLAS_INT>(x.stride(0));

    return cblas_dnrm2(n, x.data(), incx);
}

template <std::size_t ext_x, class Layout_x, class Accessor_x>
inline double
norm2(stdex::mdspan<const double, stdex::extents<std::size_t, ext_x>, Layout_x, Accessor_x> x)
{
    const BLAS_INT n = static_cast<BLAS_INT>(x.size());
    const BLAS_INT incx = static_cast<BLAS_INT>(x.stride(0));

    return cblas_dnrm2(n, x.data(), incx);
}

template <class Layout, class Allocator>
inline double norm2(const Sci::Vector<double, Layout, Allocator>& x)
{
    return norm2(x.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS1_VECTOR_NORM2_H

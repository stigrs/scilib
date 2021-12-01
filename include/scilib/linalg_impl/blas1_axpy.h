// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_AXPY_H
#define SCILIB_LINALG_BLAS1_AXPY_H

#include <experimental/mdspan>
#include <scilib/mdarray.h>

namespace Scilib {
namespace Linalg {

namespace stdex = std::experimental;

template <class T,
          stdex::extents<>::size_type ext_x,
          class Layout_x,
          class Accessor_x,
          stdex::extents<>::size_type ext_y,
          class Layout_y,
          class Accessor_y>
inline void
axpy(const T& scalar,
     stdex::mdspan<T, stdex::extents<ext_x>, Layout_x, Accessor_x> x,
     stdex::mdspan<T, stdex::extents<ext_y>, Layout_y, Accessor_y> y)
{
    static_assert(x.static_extent(0) == y.static_extent(0));

    using size_type = stdex::extents<>::size_type;

    for (size_type i = 0; i < y.extent(0); ++i) {
        y(i) = scalar * x(i) + y(i);
    }
}

template <class T>
inline void axpy(const T& scalar, const Vector<T>& x, Vector<T>& y)
{
    axpy(scalar, x.view(), y.view());
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_BLAS1_AXPY_H

// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_AXPY_H
#define SCILIB_LINALG_BLAS1_AXPY_H

#include <experimental/mdspan>
#include <scilib/mdarray.h>

namespace Sci {
namespace Linalg {

namespace stdex = std::experimental;

// clang-format off
template <class T_scalar,
          class T_x,
          stdex::extents<>::size_type ext_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          stdex::extents<>::size_type ext_y,
          class Layout_y,
          class Accessor_y>
    requires (!std::is_const_v<T_y>)
inline void
axpy(const T_scalar& scalar,
     stdex::mdspan<T_x, stdex::extents<ext_x>, Layout_x, Accessor_x> x,
     stdex::mdspan<T_y, stdex::extents<ext_y>, Layout_y, Accessor_y> y)
// clang-format on
{
    static_assert(x.static_extent(0) == y.static_extent(0));

    using size_type = stdex::extents<>::size_type;

    for (size_type i = 0; i < y.extent(0); ++i) {
        y(i) = scalar * x(i) + y(i);
    }
}

// clang-format off
template <class T_scalar, 
          class T_x, 
          class Layout_x, 
          class Allocator_x, 
          class T_y, 
          class Layout_y,
          class Allocator_y>
    requires (!std::is_const_v<T_y>)
inline void axpy(const T_scalar& scalar,
                 const Sci::Vector<T_x, Layout_x, Allocator_x>& x,
                 Sci::Vector<T_y, Layout_y, Allocator_y>& y)
// clang-format on
{
    axpy(scalar, x.view(), y.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS1_AXPY_H

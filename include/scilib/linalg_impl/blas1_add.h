// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_ADD_H
#define SCILIB_LINALG_BLAS1_ADD_H

#include <experimental/mdspan>

namespace Scilib {
namespace Linalg {

namespace stdex = std::experimental;

// clang-format off
template <class T_x,
          stdex::extents<>::size_type ext_x,
          class Layout_x,
          class Accessor_x,
          stdex::extents<>::size_type ext_y,
          class T_y,
          class Layout_y,
          class Accessor_y,
          stdex::extents<>::size_type ext_z,
          class T_z,
          class Layout_z,
          class Accessor_z>
    requires (!std::is_const_v<T_z>)
inline void
add(stdex::mdspan<T_x, stdex::extents<ext_x>, Layout_x, Accessor_x> x,
    stdex::mdspan<T_y, stdex::extents<ext_y>, Layout_y, Accessor_y> y,
    stdex::mdspan<T_z, stdex::extents<ext_z>, Layout_z, Accessor_z> z)
// clang-format on
{
    static_assert(x.static_extent(0) == z.static_extent(0));
    static_assert(y.static_extent(0) == z.static_extent(0));
    static_assert(x.static_extent(0) == y.static_extent(0));

    using size_type = stdex::extents<>::size_type;

    for (size_type i = 0; i < z.extent(0); ++i) {
        z(i) = x(i) + y(i);
    }
}

// clang-format off
template <class T_x,
          class Layout_x,
          class T_y,
          class Layout_y,
          class T_z,
          class Layout_z>
    requires (!std::is_const_v<T_z>)
inline void add(const Scilib::Vector<T_x, Layout_x>& x,
                const Scilib::Vector<T_y, Layout_y>& y,
                Scilib::Vector<T_z, Layout_z>& z)
{
    add(x.view(), y.view(), z.view());
}

} // namespace Linalg
} // namespace Scilib

#endif // SCILIB_LINALG_BLAS1_ADD_H

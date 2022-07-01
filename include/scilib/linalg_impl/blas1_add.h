// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_ADD_H
#define SCILIB_LINALG_BLAS1_ADD_H

namespace Sci {
namespace Linalg {

namespace stdex = std::experimental;

template <class T_x,
          std::size_t ext_x,
          class Layout_x,
          class Accessor_x,
          std::size_t ext_y,
          class T_y,
          class Layout_y,
          class Accessor_y,
          std::size_t ext_z,
          class T_z,
          class Layout_z,
          class Accessor_z>
    requires(!std::is_const_v<T_z>)
inline void add(stdex::mdspan<T_x, stdex::extents<index, ext_x>, Layout_x, Accessor_x> x,
                stdex::mdspan<T_y, stdex::extents<index, ext_y>, Layout_y, Accessor_y> y,
                stdex::mdspan<T_z, stdex::extents<index, ext_z>, Layout_z, Accessor_z> z)
{
    static_assert(x.static_extent(0) == z.static_extent(0));
    static_assert(y.static_extent(0) == z.static_extent(0));
    static_assert(x.static_extent(0) == y.static_extent(0));

    using index_type = index;

    for (index_type i = 0; i < z.extent(0); ++i) {
        z(i) = x(i) + y(i);
    }
}

template <class T_x,
          class Layout_x,
          class Allocator_x,
          class T_y,
          class Layout_y,
          class Allocator_y,
          class T_z,
          class Layout_z,
          class Allocator_z>
    requires(!std::is_const_v<T_z>)
inline void add(const Sci::Vector<T_x, Layout_x, Allocator_x>& x,
                const Sci::Vector<T_y, Layout_y, Allocator_y>& y,
                Sci::Vector<T_z, Layout_z, Allocator_z>& z)
{
    add(x.view(), y.view(), z.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS1_ADD_H

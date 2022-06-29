// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_AXPY_H
#define SCILIB_LINALG_BLAS1_AXPY_H

namespace Sci {
namespace Linalg {

namespace stdex = std::experimental;

template <class T_scalar,
          class T_x,
          std::size_t ext_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          std::size_t ext_y,
          class Layout_y,
          class Accessor_y>
    requires(!std::is_const_v<T_y>)
inline void axpy(
    const T_scalar& scalar,
    stdex::mdspan<T_x, stdex::extents<std::size_t, ext_x>, Layout_x, Accessor_x>
        x,
    stdex::mdspan<T_y, stdex::extents<std::size_t, ext_y>, Layout_y, Accessor_y>
        y)
{
    static_assert(x.static_extent(0) == y.static_extent(0));

    using size_type = std::size_t;

    for (size_type i = 0; i < y.extent(0); ++i) {
        y(i) = scalar * x(i) + y(i);
    }
}

template <class T_scalar,
          class T_x,
          class Layout_x,
          class Allocator_x,
          class T_y,
          class Layout_y,
          class Allocator_y>
    requires(!std::is_const_v<T_y>)
inline void axpy(const T_scalar& scalar,
                 const Sci::Vector<T_x, Layout_x, Allocator_x>& x,
                 Sci::Vector<T_y, Layout_y, Allocator_y>& y)
{
    axpy(scalar, x.view(), y.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS1_AXPY_H

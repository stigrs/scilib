// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_ADD_H
#define SCILIB_LINALG_BLAS1_ADD_H

#include <experimental/linalg>

namespace Sci {
namespace Linalg {

template <class T_x,
          class IndexType_x,
          std::size_t ext_x,
          class Layout_x,
          class Container_x,
          class T_y,
          class IndexType_y,
          std::size_t ext_y,
          class Layout_y,
          class Container_y,
          class T_z,
          class IndexType_z,
          std::size_t ext_z,
          class Layout_z,
          class Container_z>
    requires(!std::is_const_v<T_z> && std::is_integral_v<IndexType_x> &&
             std::is_integral_v<IndexType_y> && std::is_integral_v<IndexType_z>)
inline void
add(const Sci::MDArray<T_x, stdex::extents<IndexType_x, ext_x>, Layout_x, Container_x>& x,
    const Sci::MDArray<T_y, stdex::extents<IndexType_y, ext_y>, Layout_y, Container_y>& y,
    Sci::MDArray<T_z, stdex::extents<IndexType_z, ext_z>, Layout_z, Container_z>& z)
{
    std::experimental::linalg::add(x.to_mdspan(), y.to_mdspan(), z.to_mdspan());
}

template <class T_x,
          class IndexType_x,
          std::size_t numrows_x,
          std::size_t numcols_x,
          class Layout_x,
          class Container_x,
          class T_y,
          class IndexType_y,
          std::size_t numrows_y,
          std::size_t numcols_y,
          class Layout_y,
          class Container_y,
          class T_z,
          class IndexType_z,
          std::size_t numrows_z,
          std::size_t numcols_z,
          class Layout_z,
          class Container_z>
    requires(!std::is_const_v<T_z> && std::is_integral_v<IndexType_x> &&
             std::is_integral_v<IndexType_y> && std::is_integral_v<IndexType_z>)
inline void
add(const Sci::
        MDArray<T_x, stdex::extents<IndexType_x, numrows_x, numcols_x>, Layout_x, Container_x>& x,
    const Sci::
        MDArray<T_y, stdex::extents<IndexType_y, numrows_y, numcols_y>, Layout_y, Container_y>& y,
    Sci::MDArray<T_z, stdex::extents<IndexType_z, numrows_z, numcols_z>, Layout_z, Container_z>& z)
{
    std::experimental::linalg::add(x.to_mdspan(), y.to_mdspan(), z.to_mdspan());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS1_ADD_H

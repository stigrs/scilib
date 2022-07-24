// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_DOT_H
#define SCILIB_LINALG_BLAS1_DOT_H

#include <experimental/linalg>
#include <type_traits>

namespace Sci {
namespace Linalg {

namespace stdex = std::experimental;

template <class T_x,
          class IndexType_x,
          std::size_t ext_x,
          class Layout_x,
          class Container_x,
          class T_y,
          class IndexType_y,
          std::size_t ext_y,
          class Layout_y,
          class Container_y>
    requires(std::is_integral_v<IndexType_x>&& std::is_integral_v<IndexType_y>)
inline auto
dot(const Sci::MDArray<T_x, stdex::extents<IndexType_x, ext_x>, Layout_x, Container_x>& x,
    const Sci::MDArray<T_y, stdex::extents<IndexType_y, ext_y>, Layout_y, Container_y>& y)
{
    return std::experimental::linalg::dot(x.view(), y.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS1_DOT_H

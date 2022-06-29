// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_LINALG_BLAS1_DOT_H
#define SCILIB_LINALG_BLAS1_DOT_H

#include <type_traits>

namespace Sci {
namespace Linalg {

namespace stdex = std::experimental;

template <class T,
          std::size_t ext_x,
          class Layout_x,
          class Accessor_x,
          std::size_t ext_y,
          class Layout_y,
          class Accessor_y>
inline auto dot(stdex::mdspan<T, stdex::extents<std::size_t, ext_x>, Layout_x, Accessor_x> x,
                stdex::mdspan<T, stdex::extents<std::size_t, ext_y>, Layout_y, Accessor_y> y)
{
    static_assert(x.static_extent(0) == y.static_extent(0));

    using size_type = std::size_t;
    using value_type = std::remove_cv_t<T>;

    value_type result = 0;
    for (size_type i = 0; i < x.extent(0); ++i) {
        result += x(i) * y(i);
    }
    return result;
}

template <class T, class Layout_x, class Allocator_x, class Layout_y, class Allocator_y>
inline T dot(const Sci::Vector<T, Layout_x, Allocator_x>& x,
             const Sci::Vector<T, Layout_y, Allocator_y>& y)
{
    return dot(x.view(), y.view());
}

} // namespace Linalg
} // namespace Sci

#endif // SCILIB_LINALG_BLAS1_DOT_H

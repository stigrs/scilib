// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_COPY_N_H
#define SCILIB_MDARRAY_COPY_N_H

#include <cassert>
#include <type_traits>

namespace Sci {

namespace stdex = std::experimental;

template <class T_x,
          std::size_t ext_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          std::size_t ext_y,
          class Layout_y,
          class Accessor_y>
    requires(!std::is_const_v<T_y>)
inline void copy_n(stdex::mdspan<T_x, stdex::extents<index, ext_x>, Layout_x, Accessor_x> x,
                   std::size_t count,
                   stdex::mdspan<T_y, stdex::extents<index, ext_y>, Layout_y, Accessor_y> y,
                   std::size_t offset = 0)
{
    Expects(count <= x.extent(0));
    Expects(offset >= 0 && offset < count);
    Expects(count > 0 && offset + count <= y.extent(0));
    using index_type = index;
    for (index_type i = 0; i < count; ++i) {
        y(i + offset) = x(i);
    }
}

template <class T, class Layout>
inline void
copy_n(const Vector<T, Layout>& x, std::size_t count, Vector<T, Layout>& y, std::size_t offset = 0)
{
    copy_n(x.view(), count, y.view(), offset);
}

} // namespace Sci

#endif // SCILIB_MDARRAY_COPY_N_H

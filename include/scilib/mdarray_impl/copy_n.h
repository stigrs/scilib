// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_COPY_N_H
#define SCILIB_MDARRAY_COPY_N_H

#include <experimental/mdspan>
#include <cassert>
#include <type_traits>

namespace Sci {

namespace stdex = std::experimental;

// clang-format off
template <class T_x,
          stdex::extents<>::size_type ext_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          stdex::extents<>::size_type ext_y,
          class Layout_y,
          class Accessor_y>
    requires (!std::is_const_v<T_y>)
inline void
copy_n(stdex::mdspan<T_x, stdex::extents<ext_x>, Layout_x, Accessor_x> x,
       stdex::extents<>::size_type count,
       stdex::mdspan<T_y, stdex::extents<ext_y>, Layout_y, Accessor_y> y,
       stdex::extents<>::size_type offset = 0)
// clang-format on
{
    assert(count <= x.extent(0));
    assert(offset >= 0 && offset < count);
    assert(count > 0 && offset + count <= y.extent(0));
    using size_type = stdex::extents<>::size_type;
    for (size_type i = 0; i < count; ++i) {
        y(i + offset) = x(i);
    }
}

template <class T, class Layout>
inline void copy_n(const Vector<T, Layout>& x,
                   stdex::extents<>::size_type count,
                   Vector<T, Layout>& y,
                   stdex::extents<>::size_type offset = 0)
{
    copy_n(x.view(), count, y.view(), offset);
}

} // namespace Sci

#endif // SCILIB_MDARRAY_COPY_N_H

// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#ifndef SCILIB_MDARRAY_COPY_N_H
#define SCILIB_MDARRAY_COPY_N_H

#include <experimental/mdspan>
#include <cassert>

namespace Scilib {

namespace stdex = std::experimental;

template <class T,
          stdex::extents<>::size_type ext_x,
          class Layout_x,
          class Accessor_x,
          stdex::extents<>::size_type ext_y,
          class Layout_y,
          class Accessor_y>
inline void
copy_n(stdex::mdspan<T, stdex::extents<ext_x>, Layout_x, Accessor_x> x,
       stdex::extents<>::size_type start,
       stdex::extents<>::size_type count,
       stdex::mdspan<T, stdex::extents<ext_y>, Layout_y, Accessor_y> y)
{
    static_assert(x.static_extent(0) == y.static_extent(0));
    assert(start >= 0 && start < count);
    assert(count > 0 && count <= y.extent(0));
    using size_type = stdex::extents<>::size_type;
    for (size_type i = start; i < count; ++i) {
        y(i) = x(i);
    }
}

} // namespace Scilib

#endif // SCILIB_MDARRAY_COPY_N_H

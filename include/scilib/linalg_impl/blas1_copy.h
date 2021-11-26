// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#pragma once

#include <experimental/mdspan>

namespace Scilib {
namespace Linalg {

namespace stdex = std::experimental;

template <class T,
          stdex::extents<>::size_type ext_x,
          class Layout_x,
          class Accessor_x,
          stdex::extents<>::size_type ext_y,
          class Layout_y,
          class Accessor_y>
inline void
copy(stdex::mdspan<T, stdex::extents<ext_x>, Layout_x, Accessor_x> x,
     stdex::mdspan<T, stdex::extents<ext_y>, Layout_y, Accessor_y> y)
{
    static_assert(x.static_extent(0) == y.static_extent(0));
    using size_type = stdex::extents<>::size_type;
    for (size_type i = 0; i < y.extent(0); ++i) {
        y(i) = x(i);
    }
}

template <class T,
          stdex::extents<>::size_type nrows_x,
          stdex::extents<>::size_type ncols_x,
          class Layout_x,
          class Accessor_x,
          stdex::extents<>::size_type nrows_y,
          stdex::extents<>::size_type ncols_y,
          class Layout_y,
          class Accessor_y>
inline void
copy(stdex::mdspan<T, stdex::extents<nrows_x, ncols_x>, Layout_x, Accessor_x> x,
     stdex::mdspan<T, stdex::extents<nrows_y, ncols_y>, Layout_y, Accessor_y> y)
{
    static_assert(x.static_extent(0) == y.static_extent(0));
    static_assert(x.static_extent(1) == y.static_extent(1));

    using size_type = stdex::extents<>::size_type;

    for (size_type i = 0; i < y.extent(0); ++i) {
        for (size_type j = 0; j < y.extent(1); ++j) {
            y(i, j) = x(i, j);
        }
    }
}

} // namespace Linalg
} // namespace Scilib

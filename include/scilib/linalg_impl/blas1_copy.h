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

template <class T_x,
          stdex::extents<>::size_type ext_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          stdex::extents<>::size_type ext_y,
          class Layout_y,
          class Accessor_y>
inline void
copy(stdex::mdspan<T_x, stdex::extents<ext_x>, Layout_x, Accessor_x> x,
     stdex::mdspan<T_y, stdex::extents<ext_y>, Layout_y, Accessor_y> y)
{
    static_assert(x.static_extent(0) == y.static_extent(0));
    for (stdex::extents<>::size_type i = 0; i < y.extent(0); ++i) {
        y(i) = x(i);
    }
}

template <class T_x,
          stdex::extents<>::size_type nrows_x,
          stdex::extents<>::size_type ncols_x,
          class Layout_x,
          class Accessor_x,
          class T_y,
          stdex::extents<>::size_type nrows_y,
          stdex::extents<>::size_type ncols_y,
          class Layout_y,
          class Accessor_y>
inline void
// clang-format off
copy(stdex::mdspan<T_x, stdex::extents<nrows_x, ncols_x>, Layout_x, Accessor_x> x,
     stdex::mdspan<T_y, stdex::extents<nrows_y, ncols_y>, Layout_y, Accessor_y> y)
// clang-format on
{
    static_assert(x.static_extent(0) == y.static_extent(0));
    static_assert(x.static_extent(1) == y.static_extent(1));
    for (std::size_t i = 0; i < y.extent(0); ++i) {
        for (std::size_t j = 0; j < y.extent(1); ++j) {
            y(i, j) = x(i, j);
        }
    }
}

} // namespace Linalg
} // namespace Scilib

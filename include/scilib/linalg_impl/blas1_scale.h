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
          class Accessor_x>
inline void
scale(const T& scalar,
      stdex::mdspan<T, stdex::extents<ext_x>, Layout_x, Accessor_x> x)
{
    using size_type = stdex::extents<>::size_type;
    for (size_type i = 0; i < x.extent(0); ++i) {
        x(i) *= scalar;
    }
}

template <class T,
          stdex::extents<>::size_type nrows,
          stdex::extents<>::size_type ncols,
          class Layout_m,
          class Accessor_m>
inline void
scale(const T& scalar,
      stdex::mdspan<T, stdex::extents<nrows, ncols>, Layout_m, Accessor_m> m)
{
    using size_type = stdex::extents<>::size_type;
    for (size_type i = 0; i < m.extent(0); ++i) {
        for (size_type j = 0; j < m.extent(1); ++j) {
            m(i, j) *= scalar;
        }
    }
}

} // namespace Linalg
} // namespace Scilib

// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#pragma once

#include <experimental/mdspan>
#include <cmath>

namespace Scilib {
namespace Linalg {

namespace stdex = std::experimental;

template <class T,
          stdex::extents<>::size_type ext_x,
          class Layout_x,
          class Accessor_x>
inline std::size_t
idx_abs_max(stdex::mdspan<T, stdex::extents<ext_x>, Layout_x, Accessor_x> x)
{
    using magn_type = decltype(std::abs(x(0)));

    std::size_t max_idx = 0;
    magn_type max_val = std::abs(x(0));
    for (std::size_t i = 0; i < x.extent(0); ++i) {
        if (max_val < std::abs(x(i))) {
            max_val = std::abs(x(i));
            max_idx = i;
        }
    }
    return max_idx;
}

} // namespace Linalg
} // namespace Scilib

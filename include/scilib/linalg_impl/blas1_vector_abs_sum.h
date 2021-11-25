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
inline T
abs_sum(stdex::mdspan<T, stdex::extents<ext_x>, Layout_x, Accessor_x> x)
{
    using size_type = stdex::extents<>::size_type;

    T result = 0;
    for (size_type i = 0; i < x.extent(0); ++i) {
        result += std::abs(x(i));
    }
    return result;
}

} // namespace Linalg
} // namespace Scilib

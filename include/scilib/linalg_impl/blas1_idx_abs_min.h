// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#pragma once

#include <scilib/mdarray.h>
#include <cassert>
#include <cmath>

namespace Scilib {
namespace Linalg {

template <typename T>
inline std::size_t idx_abs_min(const Vector_view<T> v)
{
    using magn_type = decltype(std::abs(v(0)));

    std::size_t min_idx = 0;
    magn_type min_val = std::abs(v(0));
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (std::abs(v(i)) < min_val) {
            min_val = std::abs(v(i));
            min_idx = i;
        }
    }
    return min_idx;
}

} // namespace Linalg
} // namespace Scilib
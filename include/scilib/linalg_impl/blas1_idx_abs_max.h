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
inline std::size_t idx_abs_max(const Vector_view<T> v)
{
    using magn_type = decltype(std::abs(v(0)));

    std::size_t max_idx = 0;
    magn_type max_val = std::abs(v(0));
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (max_val < std::abs(v(i))) {
            max_val = std::abs(v(i));
            max_idx = i;
        }
    }
    return max_idx;
}

} // namespace Linalg
} // namespace Scilib

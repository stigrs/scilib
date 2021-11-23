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
inline T abs_sum(const Vector_view<T>& x)
{
    T result = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        result += std::abs(x(i));
    }
    return result;
}

} // namespace Linalg
} // namespace Scilib

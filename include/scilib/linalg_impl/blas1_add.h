// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#pragma once

#include <scilib/mdarray.h>
#include <cassert>

namespace Scilib {
namespace Linalg {

template <typename T>
inline void add(Vector_view<T> x, Vector_view<T> y, Vector_view<T> z)
{
    static_assert(x.static_extent(0) == z.static_extent(0));
    static_assert(y.static_extent(0) == z.static_extent(0));
    static_assert(x.static_extent(0) == y.static_extent(0));

    for (std::size_t i = 0; i < z.extent(0); ++i) {
        z(i) = x(i) + y(i);
    }
}

} // namespace Linalg
} // namespace Scilib

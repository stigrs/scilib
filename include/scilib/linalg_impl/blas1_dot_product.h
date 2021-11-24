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
inline T dot_product(Vector_view<T> x, Vector_view<T> y)
{
    assert(x.size() == y.size());

    T result = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        result += x(i) * y(i);
    }
    return result;
}

template <typename T>
inline T dot_product(const Vector<T>& x, const Vector<T>& y)
{
    return dot_product(x.view(), y.view());
}

} // namespace Linalg
} // namespace Scilib

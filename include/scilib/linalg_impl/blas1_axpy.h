// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#pragma once

#include <scilib/mdarray.h>
#include <cassert>
#include <algorithm>

namespace Scilib {
namespace Linalg {

template <typename T>
inline void axpy(const T& scalar, const Vector_view<T>& x, Vector_view<T>& y)
{
    assert(x.size() == y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        y(i) = scalar * x(i) + y(i);
    }
}

template <typename T>
inline void axpy(const T& scalar, const Vector<T>& x, Vector<T>& y)
{
    axpy(scalar, x.view(), y.view());
}

} // namespace Linalg
} // namespace Scilib

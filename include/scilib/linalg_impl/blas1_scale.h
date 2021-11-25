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
void scale(const T& scalar, Vector_view<T> v)
{
    for (std::size_t i = 0; i < v.size(); ++i) {
        v(i) *= scalar;
    }
}

template <typename T>
void scale(const T& scalar, Matrix_view<T> m)
{
    for (std::size_t i = 0; i < m.extent(0); ++i) {
        for (std::size_t j = 0; j < m.extent(1); ++j) {
            m(i, j) *= scalar;
        }
    }
}

template <typename T>
inline Vector_view<T> scaled(const T& scalar, Vector_view<T> v)
{
    scale(scalar, v);
    return v;
}

template <typename T>
inline Matrix_view<T> scaled(const T& scalar, Matrix_view<T> m)
{
    scale(scalar, m);
    return m;
}

} // namespace Linalg
} // namespace Scilib

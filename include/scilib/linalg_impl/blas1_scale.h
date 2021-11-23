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
void scale(Vector_view<T>& v, const T& scalar)
{
    for (std::size_t i = 0; i < v.size(); ++i) {
        v(i) *= scalar;
    }
}

template <typename T>
void scale(Matrix_view<T>& m, const T& scalar)
{
    for (std::size_t i = 0; i < m.extent(0); ++i) {
        for (std::size_t j = 0; j < m.extent(1); ++j) {
            m(i, j) *= scalar;
        }
    }
}

} // namespace Linalg
} // namespace Scilib

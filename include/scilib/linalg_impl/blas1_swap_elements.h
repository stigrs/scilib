// Copyright (c) 2021 Stig Rune Sellevag
//
// This file is distributed under the MIT License. See the accompanying file
// LICENSE.txt or http://www.opensource.org/licenses/mit-license.php for terms
// and conditions.

#pragma once

#include <scilib/mdarray.h>
#include <cassert>
#include <utility>

namespace Scilib {
namespace Linalg {

template <typename T>
inline void swap_elements(Vector_view<T> a, Vector_view<T> b)
{
    assert(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        std::swap(a(i), b(i));
    }
}

template <typename T>
inline void swap_elements(Matrix_view<T> a, Matrix_view<T> b)
{
    assert(a.extent(0) == b.extent(0));
    assert(a.extent(1) == b.extent(1));
    for (std::size_t i = 0; i < a.extent(0); ++i) {
        for (std::size_t j = 0; j < a.extent(1); ++j) {
            std::swap(a(i, j), b(i, j));
        }
    }
}

} // namespace Linalg
} // namespace Scilib
